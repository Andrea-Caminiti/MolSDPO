"""
rewards_v2.py
=============
Optimised, multi-GPU-aware reward stack for SDPO molecular diffusion.

Changes from v1
---------------
1. Symmetric: identical reward function applied to start, anchor and last
   with progressive RDKit weights [0.5, 0.75, 1.0].  The v1 asymmetry
   (rdkit bonus on last only) injected a constant +3.5 offset into every
   advantage, causing reward hacking where start degraded intentionally.

2. Fixed MW target: QM9 heavy-atom molecules average ~120-130 Da.
   The v1 target of 80 Da scored every generated molecule (~300-440 Da)
   at zero, making w_quality effectively dead for the entire training run.

3. Cross-GPU diversity: fingerprints are converted to float16 tensors
   and gathered via dist.all_gather when distributed training is active,
   giving a true global diversity signal instead of per-rank diversity.

4. Stronger atom surplus penalty: matches the deficit penalty (3x) with
   a tighter threshold (1.2x vs 1.5x), addressing the consistent 300-440 Da
   molecular weight observed across all previous runs.

5. Vectorised Tanimoto: binary fingerprint matrix computed via a single
   batched tensor operation instead of an O(n) Python loop.

6. Removed: proximity_reward (redundant with fragment + rg penalties),
   dead vdW parameter.

Drop-in compatible: same get_reward() signature, returns [3, B] tensor.
"""

import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds, Descriptors, DataStructs, AllChem
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PT             = Chem.GetPeriodicTable()
_ELEMENT_SYMBOLS = {z: _PT.GetElementSymbol(z) for z in [1, 6, 7, 8, 9]}

# QM9 heavy-atom target (mean 9, max 29).  Reward is maximised at n_target
# and falls off symmetrically with weight 3 in both directions.
_N_TARGET       = 9.0
_N_THRESHOLD    = _N_TARGET * 1.2   # 10.8 — surplus starts being penalised here
_ATOM_W_DEFICIT = 3.0               # symmetric with surplus
_ATOM_W_SURPLUS = 3.0               # was 1.0 in v1 — molecules were 300-440 Da

# RDKit tier: progressive weights per trajectory position.
# All three are evaluated; last gets full signal, anchor 75%, start 50%.
# This keeps the SDPO advantage properly signed while grounding start in chemistry.
_RDKIT_WEIGHTS  = {0: 0.5, 1: 0.75, 2: 1.0}   # indexed as [start, anchor, last]

# QM9 MW target.  v1 used 80 Da (wrong); actual QM9 range is ~16-200 Da, mean ~120.
_MW_TARGET      = 130.0
_MW_WIDTH       = 80.0     # Gaussian half-width: reward=0 outside [50, 210] Da

# Thread pool for RDKit CPU work — shared across all calls.
_N_WORKERS      = min(32, (os.cpu_count() or 4) * 2)
_THREAD_POOL    = ThreadPoolExecutor(max_workers=_N_WORKERS)


# ---------------------------------------------------------------------------
# Geometric reward  (fully GPU, vectorised over [M, B, N])
# ---------------------------------------------------------------------------

class GeometricReward:
    """
    Pure-tensor geometric reward over a batch of molecules.

    All operations are on-device.  No CPU crossing.  Call compute() once per
    training step; it handles the [M, B, N] batch dimension that comes from
    stacking start / anchor / last along the first axis.
    """
    _MAX_Z = 10

    def __init__(self, device: str = "cuda"):
        self.device = device

        vdw_defaults = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47}
        cov_defaults = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
        val_defaults = {1: 1,    6: 4,    7: 3,    8: 2,    9: 1   }

        pt      = Chem.GetPeriodicTable()
        vdw_tbl = torch.full((self._MAX_Z + 1,), 1.5)
        cov_tbl = torch.full((self._MAX_Z + 1,), 0.77)
        val_tbl = torch.full((self._MAX_Z + 1,), 4.0)

        for z in range(1, self._MAX_Z + 1):
            try:
                vdw_tbl[z] = pt.GetRvdw(z)
                cov_tbl[z] = pt.GetRcovalent(z)
                val_tbl[z] = pt.GetDefaultValence(z)
            except Exception:
                vdw_tbl[z] = vdw_defaults.get(z, 1.5)
                cov_tbl[z] = cov_defaults.get(z, 0.77)
                val_tbl[z] = val_defaults.get(z, 4.0)

        # Padding atom (index 0) has zero radii so it never contributes.
        vdw_tbl[0] = cov_tbl[0] = val_tbl[0] = 0.0

        self._vdw_tbl = vdw_tbl.to(device)
        self._cov_tbl = cov_tbl.to(device)
        self._val_tbl = val_tbl.to(device)

        # Cache identity matrix to avoid re-allocation.
        self._eye_N   = -1
        self._eye     = None

    def _eye_mask(self, N: int) -> torch.Tensor:
        if N != self._eye_N:
            self._eye_N = N
            self._eye   = torch.eye(N, dtype=torch.bool, device=self.device)
        return self._eye

    @torch.no_grad()
    def compute(
        self,
        pos:        torch.Tensor,   # [M, B, N, 3]  Angstroms, pre-centred
        atom_types: torch.Tensor,   # [M, B, N, 1]  atomic numbers (0 = pad)
    ) -> torch.Tensor:              # [M, B]
        M, B, N, _ = pos.shape
        eye        = self._eye_mask(N)

        z          = atom_types.squeeze(-1).long().clamp(0, self._MAX_Z)
        r_vdw      = self._vdw_tbl[z]                          # [M, B, N]
        r_cov      = self._cov_tbl[z]                          # [M, B, N]
        valences   = self._val_tbl[z]                          # [M, B, N]

        real_mask  = z > 0                                     # [M, B, N]
        real_f     = real_mask.float()
        n_real     = real_f.sum(-1).clamp(min=1.0)             # [M, B]

        dist_mat   = torch.cdist(pos, pos)                     # [M, B, N, N]
        real_pair  = real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)

        t_dist     = r_cov.unsqueeze(-1) + r_cov.unsqueeze(-2) # [M,B,N,N] cov sum
        bond_thr   = t_dist * 1.15

        # ── 1. Atom count ──────────────────────────────────────────────────
        deficit = (_N_TARGET - n_real).clamp(min=0.0)
        surplus = (n_real - _N_THRESHOLD).clamp(min=0.0)
        r_atom  = -_ATOM_W_DEFICIT * deficit - _ATOM_W_SURPLUS * surplus

        # ── 2. Radius of gyration ──────────────────────────────────────────
        real_f3d  = real_f.unsqueeze(-1)                        # [M,B,N,1]
        centroid  = (pos * real_f3d).sum(2, keepdim=True) / n_real[..., None, None]
        rg_sq     = ((pos - centroid).pow(2).sum(-1) * real_f).sum(-1) / n_real
        r_rg      = -0.15 * (rg_sq - 9.0).clamp(min=0.0)

        # ── 3. Bond identification ─────────────────────────────────────────
        bonded    = (dist_mat > 0.1) & (dist_mat < bond_thr) & ~eye & real_pair

        # ── 4. 1-3 / 1-4 exclusion (via batched matmul) ───────────────────
        A    = bonded.float().view(M * B, N, N)
        is13 = (torch.bmm(A, A) > 0).view(M, B, N, N)
        is14 = (torch.bmm(A, is13.float().view(M * B, N, N)) > 0).view(M, B, N, N)
        nb   = ~bonded & ~is13 & ~is14 & ~eye & real_pair

        # ── 5. Bond length ─────────────────────────────────────────────────
        n_bonds  = bonded.float().sum((-2, -1)).clamp(min=1.0)
        r_bond   = -(((dist_mat - t_dist).pow(2)) * bonded).sum((-2, -1)) / n_bonds

        # ── 6. Steric clash ────────────────────────────────────────────────
        vdw_sum  = (r_vdw.unsqueeze(-1) + r_vdw.unsqueeze(-2)) * 0.9
        r_clash  = -((vdw_sum - dist_mat).clamp(min=0.0).pow(2) * nb).sum((-2,-1)) / n_real

        # ── 7. Valence ─────────────────────────────────────────────────────
        cur_val  = bonded.float().sum(-1)
        r_val    = -((cur_val - valences).clamp(min=0.0).pow(2)).sum(-1) / n_real

        # ── 8. Fragment / isolation ────────────────────────────────────────
        isolated = real_mask & ~(bonded.any(-1) & real_mask)
        r_frag   = -(isolated.float().sum(-1) / n_real)

        # ── Total ──────────────────────────────────────────────────────────
        return (      r_atom
                + 2.0 * r_bond
                + 5.0 * r_clash
                + 2.0 * r_val
                + 3.0 * r_frag
                + 1.0 * r_rg)


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _mol_to_fp_tensor(mol) -> torch.Tensor | None:
    """Convert an RDKit mol to a float16 fingerprint tensor [1024]."""
    if mol is None:
        return None
    try:
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.from_numpy(arr).half()   # float16 saves memory during gather
    except Exception:
        return None


def _tanimoto_matrix(fp_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Tanimoto similarity in a single batched operation.

    fp_mat : [N, 1024]  binary float matrix (0 or 1, float16 or float32)
    Returns: [N, N]  symmetric Tanimoto matrix with zero diagonal.

    Tanimoto(a, b) = |a ∩ b| / |a ∪ b|
                   = dot(a,b) / (|a|² + |b|² - dot(a,b))
    """
    fp = fp_mat.float()
    inter = fp @ fp.T                                    # [N, N]  |a ∩ b|
    norms = fp.sum(1, keepdim=True)                      # [N, 1]  |a|
    union = norms + norms.T - inter                      # [N, N]  |a ∪ b|
    tani  = inter / union.clamp(min=1e-6)
    tani.fill_diagonal_(0.0)
    return tani


def _gather_fps_across_gpus(
    local_fps: list[torch.Tensor | None],
    device: torch.device,
) -> torch.Tensor:
    """
    All-gather fingerprint tensors across ranks.

    Pads None entries with zeros so every rank contributes a fixed-size
    [B_local, 1024] block.  Returns [B_global, 1024] on every rank.
    """
    B = len(local_fps)
    mat = torch.zeros(B, 1024, dtype=torch.float16, device=device)
    for i, fp in enumerate(local_fps):
        if fp is not None:
            mat[i] = fp.to(device)

    if dist.is_available() and dist.is_initialized():
        world  = dist.get_world_size()
        chunks = [torch.zeros_like(mat) for _ in range(world)]
        dist.all_gather(chunks, mat)
        mat    = torch.cat(chunks, dim=0)   # [B_global, 1024]

    return mat


# ---------------------------------------------------------------------------
# Per-molecule RDKit evaluation  (CPU, thread-pool)
# ---------------------------------------------------------------------------

def _parse_molecule(args):
    """
    Evaluate one molecule with RDKit.

    Returns (validity_score, quality_score, mol_or_None).
    """
    coords_np, atomic_nums_np, w_valid, w_quality = args
    real        = atomic_nums_np != 0
    real_coords = coords_np[real]
    real_nums   = atomic_nums_np[real]

    if len(real_nums) < 2:
        return -w_valid * 2.0, 0.0, None

    try:
        edit = Chem.RWMol()
        conf = Chem.Conformer(len(real_nums))
        for i, z in enumerate(real_nums):
            sym = _ELEMENT_SYMBOLS.get(int(z), 'C')
            idx = edit.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(idx, real_coords[i].tolist())
        mol = edit.GetMol()
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(mol)
        Chem.AssignRadicals(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return -w_valid * 0.5, 0.0, None

    n_frags   = len(Chem.GetMolFrags(mol))
    validity  = w_valid * (2.0 if n_frags == 1 else 0.2)
    valid_mol = mol if n_frags == 1 else None
    quality   = 0.0

    if valid_mol is not None and w_quality > 0.0:
        try:
            qed      = Descriptors.qed(mol)
            mw       = Descriptors.MolWt(mol)
            tpsa     = rdMolDescriptors.CalcTPSA(mol)

            # MW: Gaussian centred at _MW_TARGET, width _MW_WIDTH.
            # Scores 1.0 at target, 0.0 at ±_MW_WIDTH away.
            mw_score   = float(np.exp(-0.5 * ((mw - _MW_TARGET) / _MW_WIDTH) ** 2))

            # TPSA: drug-like range is 20-140 Å².  Score 1.0 inside, penalise outside.
            tpsa_score = float(np.clip(1.0 - max(0.0, tpsa - 140.0) / 100.0, 0.0, 1.0))

            quality = w_quality * (0.6 * qed + 0.3 * mw_score + 0.1 * tpsa_score)
        except Exception:
            pass

    return validity, quality, valid_mol


# ---------------------------------------------------------------------------
# RDKit batch evaluation  (called for each of start / anchor / last)
# ---------------------------------------------------------------------------

def _rdkit_batch(
    coords:      torch.Tensor,       # [B, N, 3]
    atom_types:  torch.Tensor,       # [B, N, A]
    vocab:       torch.Tensor,       # atom-type lookup
    scale:       float,
    w_valid:     float,
    w_quality:   float,
    w_diversity: float,
    device:      torch.device,
) -> torch.Tensor:                   # [B]  reward
    B = coords.shape[0]

    # Centre and scale coordinates to Angstroms.
    atomic_nums = vocab[atom_types.argmax(-1)]            # [B, N]
    real_f      = (atomic_nums != 0).float().unsqueeze(-1)
    n_real      = real_f.sum(1, keepdim=True).clamp(min=1)
    centroid    = (coords * real_f).sum(1, keepdim=True) / n_real
    coords_c    = (coords - centroid) * scale             # [B, N, 3] Angstroms

    coords_np  = coords_c.detach().cpu().float().numpy()
    atoms_np   = atomic_nums.detach().cpu().numpy()

    # Parallel RDKit evaluation.
    args    = [(coords_np[b], atoms_np[b], w_valid, w_quality) for b in range(B)]
    results = list(_THREAD_POOL.map(_parse_molecule, args))

    validity = torch.tensor([r[0] for r in results], dtype=torch.float32, device=device)
    quality  = torch.tensor([r[1] for r in results], dtype=torch.float32, device=device)
    mols     = [r[2] for r in results]

    # ── Diversity bonus ────────────────────────────────────────────────────
    # Build local fingerprint matrix, then gather across GPUs if distributed.
    local_fps = [_mol_to_fp_tensor(m) for m in mols]
    fp_mat    = _gather_fps_across_gpus(local_fps, device)    # [B_global, 1024]

    # Extract the local slice of the full similarity matrix.
    rank    = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    start_i = rank * B
    end_i   = start_i + B

    valid_global = (fp_mat.sum(-1) > 0)                       # [B_global]
    diversity    = torch.zeros(B, dtype=torch.float32, device=device)

    n_valid_global = valid_global.sum().item()
    if n_valid_global >= 2:
        tani      = _tanimoto_matrix(fp_mat)                  # [B_global, B_global]
        # Mean dissimilarity of each local molecule vs all global molecules.
        # Exclude self-similarity (already zeroed on diagonal).
        denom     = valid_global.float().sum().clamp(min=1.0) - 1.0
        mean_sim  = (tani[start_i:end_i] * valid_global.float()).sum(-1) / denom
        local_valid = valid_global[start_i:end_i]
        diversity = w_diversity * (1.0 - mean_sim) * local_valid.float()

    return validity + quality + diversity   # [B]


# ---------------------------------------------------------------------------
# Unified entry point  (drop-in replacement for v1 get_reward)
# ---------------------------------------------------------------------------

def get_reward(
    mols,
    rewarder:    GeometricReward,
    vocab:       torch.Tensor,
    vdW=None,               # kept for API compatibility; unused
    w_geom:      float = 1.0,
    w_valid:     float = 2.0,
    w_quality:   float = 1.5,
    w_diversity: float = 1.5,
    scale:       float = 2.2,
    debug:       bool  = False,
) -> torch.Tensor:          # [3, B]  — row 0=start, 1=anchor, 2=last
    """
    Compute rewards for start, anchor and last molecules.

    Returns [3, B] tensor matching the v1 interface so train.py needs no changes.

    Key difference from v1: the RDKit bonus (validity + quality + diversity) is
    now applied to ALL three molecules with progressive weights [0.5, 0.75, 1.0].
    In v1 it was applied to last only, creating a constant +3.5 advantage offset
    that caused start-reward degradation via reward hacking.
    """
    start, anchor, last = mols
    device = start[0].device

    # ── Geometric reward (GPU, [3, B]) ─────────────────────────────────────
    coords_all = torch.stack((start[0], anchor[0], last[0]))    # [3, B, N, 3]
    atoms_all  = torch.stack((start[1], anchor[1], last[1]))    # [3, B, N, A]
    atoms_num  = vocab[atoms_all.argmax(-1, keepdim=True)]      # [3, B, N, 1]

    real_f3d  = (atoms_num > 0).float()
    n_real    = real_f3d.sum(2, keepdim=True).clamp(min=1)
    centroid  = (coords_all * real_f3d).sum(2, keepdim=True) / n_real
    coords_c  = (coords_all - centroid) * scale                 # [3, B, N, 3]

    geom      = rewarder.compute(coords_c, atoms_num)           # [3, B]
    rewards   = w_geom * geom

    if debug:
        for i, label in enumerate(("start", "anchor", "last")):
            print(f"[geom/{label}] mean={rewards[i].mean():.3f}")

    # ── RDKit reward (CPU→GPU, per molecule, symmetric) ────────────────────
    for mol_idx, (mol_pair, w_rdkit) in enumerate(zip(
        [start, anchor, last],
        [_RDKIT_WEIGHTS[0], _RDKIT_WEIGHTS[1], _RDKIT_WEIGHTS[2]]
    )):
        bonus = _rdkit_batch(
            mol_pair[0], mol_pair[1], vocab,
            scale=scale, w_valid=w_valid,
            w_quality=w_quality, w_diversity=w_diversity,
            device=device,
        )
        rewards[mol_idx] += w_rdkit * bonus

        if debug:
            label = ("start", "anchor", "last")[mol_idx]
            print(f"[rdkit/{label}] w={w_rdkit:.2f}  bonus_mean={bonus.mean():.3f}  "
                  f"total_mean={rewards[mol_idx].mean():.3f}")

    return rewards   # [3, B]