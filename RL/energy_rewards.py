"""
rewards_energy.py  (v2 — buffer inversion fix + batch-size scaling)
====================================================================
Energy-landscape reward for SDPO molecular diffusion.

Changes from v1
---------------
1.  RepulsionBuffer capacity now scales with batch_size so the buffer
    fill-time is constant (~32 steps) regardless of batch size.
    At batch_size=16: capacity=1536.  At batch_size=60: capacity=5760.
    Previously a fixed capacity of 512 filled in 8 steps at batch_size=60,
    switching the diversity signal on too early at full force.

2.  Buffer is updated from ALL THREE trajectory positions (start, anchor,
    last) equally.  Previously only `last` was added, which progressively
    penalised the end of each trajectory while start accumulated no penalty
    — inverting the reward ordering to start > anchor > last by step ~600.

3.  Diversity reward is now a FLAT bonus (same weight for all three).
    It is a novelty signal, not a ranking signal.  Only validity/quality
    keeps the progressive weights [0.5, 0.75, 1.0] so `last` still has
    the strongest quality pressure — but diversity no longer distorts the
    SDPO advantage.

Batch size note
---------------
Going from batch_size=16 → 60 has three effects:
  a) Buffer fill speed: fixed by this version (scales automatically).
  b) Advantage normalisation: 2*60=120 samples → more stable std estimate.
     This is strictly better for SDPO; no fix needed.
  c) Mean reward gap |R1-R2| shrinks ~√(16/60) due to regression to the
     mean with more samples.  Expect gap_start ≈ 4-8 (vs 8-12 at B=16).
     This is not a problem — each step sees a richer molecule distribution.

Drop-in compatible:
    - Same get_reward() signature as v1
    - Returns [3, B] tensor (row 0=start, 1=anchor, 2=last)
    - EnergyRewarder replaces GeometricReward
    - Pass batch_size to EnergyRewarder():
        rewarder = EnergyRewarder(batch_size=args.batch_size)
"""

import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds, Descriptors

RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------------
# Physical constants — all distances in Angstroms
# ---------------------------------------------------------------------------

_MAX_Z    = 10

# Covalent radii  (index = atomic number, 0 = padding)
_R_COV = torch.tensor([0.00, 0.31, 0.76, 0.71, 0.66, 0.57,
                        0.77, 0.77, 0.77, 0.77, 0.77])

# Morse well depth D_e
_D_E   = torch.tensor([0.00, 0.40, 0.90, 0.85, 0.80, 0.70,
                        0.80, 0.80, 0.80, 0.80, 0.80])

# Lennard-Jones σ (collision diameter)
_LJ_S  = torch.tensor([0.00, 1.20, 1.70, 1.55, 1.52, 1.47,
                        1.50, 1.50, 1.50, 1.50, 1.50])

# Lennard-Jones ε
_LJ_E  = torch.tensor([0.00, 0.04, 0.09, 0.08, 0.08, 0.07,
                        0.08, 0.08, 0.08, 0.08, 0.08])

# Expected valence
_VAL   = torch.tensor([0.0,  1.0,  4.0,  3.0,  2.0,  1.0,
                        4.0,  4.0,  4.0,  4.0,  4.0])

_MORSE_A         = 2.0
_BOND_T          = 0.15     # sigmoid temperature (Angstroms)
_BOND_FAC        = 1.15     # bond threshold multiplier
_IDEAL_ANGLE_SP3 = math.radians(109.5)
_K_ANGLE         = 0.3
_K_CONNECT       = 2.0
_N_TARGET        = 9.0
_N_THRESH        = _N_TARGET * 1.2
_K_DEFICIT       = 3.0
_K_SURPLUS       = 3.0

# Progressive validity weights per trajectory position [start, anchor, last].
# Intentionally kept asymmetric — last should get stronger quality pressure.
_RDKIT_W         = [0.5, 0.75, 1.0]

# Flat diversity weight — same for all three positions.
# Diversity is a novelty bonus, not a ranking signal.
_DIV_W_FLAT      = 1.0

# Buffer fill-time in training steps (constant regardless of batch size).
_FILL_STEPS      = 32

# ---------------------------------------------------------------------------
# Thread pool for RDKit CPU work
# ---------------------------------------------------------------------------
_N_WORKERS   = min(32, (os.cpu_count() or 4) * 2)
_THREAD_POOL = ThreadPoolExecutor(max_workers=_N_WORKERS)
_PT_ELEM     = Chem.GetPeriodicTable()
_ELEM_SYM    = {z: _PT_ELEM.GetElementSymbol(z) for z in [1, 6, 7, 8, 9]}


# ---------------------------------------------------------------------------
# Repulsion buffer
# ---------------------------------------------------------------------------

class RepulsionBuffer:
    """
    Rolling circular buffer of distance-histogram fingerprints.

    Capacity scales with batch_size so the buffer takes a constant number of
    training steps to fill regardless of how many molecules are generated per
    step.  At batch_size=60 with three trajectory positions contributing per
    step, a fixed capacity of 512 filled in only ~8 steps — switching the
    diversity signal on immediately at full force, before the model has
    learned any meaningful structure.  Scaling capacity to
    _FILL_STEPS * batch_size * 3 restores the original ~32-step warm-up.

    Multi-GPU: fingerprints are gathered across all ranks before insertion so
    every rank's buffer reflects the global generation distribution.
    """

    def __init__(
        self,
        batch_size : int   = 16,
        n_bins     : int   = 64,
        r_max      : float = 12.0,
        device     : str   = "cuda",
    ):
        # ×3: start + anchor + last all contribute each step.
        capacity      = _FILL_STEPS * batch_size * 3

        self.capacity = capacity
        self.n_bins   = n_bins
        self.r_max    = r_max
        self.device   = device

        edges         = torch.linspace(0.0, r_max, n_bins + 1, device=device)
        self.edges    = edges
        self.centres  = 0.5 * (edges[:-1] + edges[1:])

        self.buf      = torch.zeros(capacity, n_bins, device=device)
        self.ptr      = 0
        self.filled   = 0

    @torch.no_grad()
    def fingerprint(
        self,
        pos  : torch.Tensor,   # [B, N, 3]  Angstroms
        mask : torch.Tensor,   # [B, N]     real-atom boolean mask
    ) -> torch.Tensor:         # [B, n_bins]
        """Soft distance histogram, rotation/translation/permutation invariant."""
        B, N, _ = pos.shape
        diff    = pos.unsqueeze(2) - pos.unsqueeze(1)
        dists   = diff.norm(dim=-1)                          # [B, N, N]

        rp      = mask.unsqueeze(2) & mask.unsqueeze(1)
        triu    = torch.ones(N, N, dtype=torch.bool, device=self.device).triu(1)
        rp      = rp & triu

        sigma   = (self.r_max / self.n_bins) * 0.5
        d_exp   = dists.unsqueeze(-1)
        c_exp   = self.centres.view(1, 1, 1, -1)
        kernel  = torch.exp(-0.5 * ((d_exp - c_exp) / sigma) ** 2)
        hist    = (kernel * rp.unsqueeze(-1).float()).sum(dim=(1, 2))  # [B, n_bins]

        n_pairs = rp.float().sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(-1)
        return hist / n_pairs

    @torch.no_grad()
    def update(self, fps: torch.Tensor) -> None:
        """
        Insert fingerprints into the buffer.

        If distributed, gathers across all ranks first so every rank's
        buffer reflects the global distribution.
        """
        if dist.is_available() and dist.is_initialized():
            world  = dist.get_world_size()
            chunks = [torch.zeros_like(fps) for _ in range(world)]
            dist.all_gather(chunks, fps)
            fps    = torch.cat(chunks, dim=0)

        fps = fps.to(self.device)
        for i in range(fps.shape[0]):
            self.buf[self.ptr] = fps[i]
            self.ptr           = (self.ptr + 1) % self.capacity
            self.filled        = min(self.filled + 1, self.capacity)

    @torch.no_grad()
    def diversity_reward(
        self,
        fps : torch.Tensor,   # [B, n_bins]
        w   : float = 1.0,
    ) -> torch.Tensor:        # [B]
        """Mean cosine dissimilarity to buffer contents, capped at 1."""
        if self.filled == 0:
            return torch.zeros(fps.shape[0], device=self.device)

        valid = self.buf[:self.filled]
        fps_n = F.normalize(fps.float(),   dim=-1)
        buf_n = F.normalize(valid.float(), dim=-1)
        cos   = fps_n @ buf_n.T               # [B, filled]
        return w * (1.0 - cos).mean(dim=-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Energy-based geometric reward
# ---------------------------------------------------------------------------

class EnergyRewarder:
    """
    Smooth force-field reward.  All ops are on-device; no CPU crossing.

    Pass batch_size so the repulsion buffer scales its capacity correctly.
    """

    def __init__(
        self,
        device      : str = "cuda",
        batch_size  : int = 60,
    ):
        self.device = device

        self.r_cov  = _R_COV.to(device)
        self.d_e    = _D_E.to(device)
        self.lj_s   = _LJ_S.to(device)
        self.lj_e   = _LJ_E.to(device)
        self.val    = _VAL.to(device)

        self.buffer = RepulsionBuffer(batch_size=batch_size, device=device)

        self._eye_N = -1
        self._eye   = None

    def _eye_mask(self, N: int) -> torch.Tensor:
        if N != self._eye_N:
            self._eye_N = N
            self._eye   = torch.eye(N, dtype=torch.bool, device=self.device)
        return self._eye

    @torch.no_grad()
    def compute(
        self,
        pos    : torch.Tensor,   # [M, B, N, 3]  Angstroms, pre-centred
        atom_z : torch.Tensor,   # [M, B, N, 1]  atomic numbers (0=pad)
    ) -> torch.Tensor:           # [M, B]
        """
        Geometric energy reward.

        Buffer updates are intentionally removed from here and handled in
        get_reward() where all three positions can be added symmetrically.
        """
        M, B, N, _ = pos.shape
        eye        = self._eye_mask(N)

        z      = atom_z.squeeze(-1).long().clamp(0, _MAX_Z)
        real   = z > 0
        real_f = real.float()
        n_real = real_f.sum(-1).clamp(min=1.0)             # [M, B]

        dist_mat  = torch.cdist(pos, pos)                   # [M, B, N, N]
        real_pair = real.unsqueeze(-1) & real.unsqueeze(-2)
        off_diag  = real_pair & ~eye

        # Per-pair physical parameters
        r0_i = self.r_cov[z]
        r0   = r0_i.unsqueeze(-1) + r0_i.unsqueeze(-2)     # eq. bond length

        de_i = self.d_e[z]
        de   = (de_i.unsqueeze(-1) * de_i.unsqueeze(-2)).sqrt()

        s_i  = self.lj_s[z]
        sig  = (s_i.unsqueeze(-1) + s_i.unsqueeze(-2)) * 0.5

        e_i  = self.lj_e[z]
        eps  = (e_i.unsqueeze(-1) * e_i.unsqueeze(-2)).sqrt()

        # ── Soft bond gate ─────────────────────────────────────────────────
        r_thr  = r0 * _BOND_FAC
        p_bond = torch.sigmoid((r_thr - dist_mat) / _BOND_T) * off_diag.float()

        # ── 1. Morse potential (bonded) ────────────────────────────────────
        dr      = dist_mat - r0
        morse_v = de * (1.0 - torch.exp(-_MORSE_A * dr)).pow(2)
        r_morse = -(morse_v * p_bond).sum((-2, -1)) / n_real

        # ── 2. Lennard-Jones (non-bonded) ──────────────────────────────────
        r_safe  = dist_mat.clamp(min=0.5)
        sr6     = (sig / r_safe).pow(6)
        lj_v    = 4.0 * eps * (sr6.pow(2) - sr6)
        nb_mask = (1.0 - p_bond) * off_diag.float()
        r_lj    = -(lj_v * nb_mask).sum((-2, -1)) / n_real

        # ── 3. Soft angle bending ──────────────────────────────────────────
        K_nbr      = min(4, N - 1)
        _, nbr_idx = p_bond.topk(K_nbr, dim=-1)            # [M, B, N, K_nbr]

        nbr_pos = torch.gather(
            pos.unsqueeze(2).expand(M, B, N, N, 3),
            3,
            nbr_idx.unsqueeze(-1).expand(M, B, N, K_nbr, 3)
        )

        nbr_pb  = torch.gather(p_bond, -1, nbr_idx)        # [M, B, N, K_nbr]
        v       = nbr_pos - pos.unsqueeze(-2)
        v_norm  = F.normalize(v, dim=-1, eps=1e-8)

        cos_ang = torch.einsum('...id,...jd->...ij', v_norm, v_norm)
        cos_ang = cos_ang.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta   = torch.acos(cos_ang)                       # [M, B, N, K_nbr, K_nbr]

        gate    = nbr_pb.unsqueeze(-1) * nbr_pb.unsqueeze(-2)
        ideal   = torch.full_like(theta, _IDEAL_ANGLE_SP3)
        r_angle = -(_K_ANGLE * (theta - ideal).pow(2) * gate).sum((-3, -2, -1)) / n_real

        # ── 4. Algebraic connectivity proxy (Fiedler) ──────────────────────
        degree     = p_bond.sum(-1)
        d_real     = degree * real_f
        d_min      = d_real.min(-1).values.clamp(min=0.0)
        d_max      = d_real.max(-1).values.clamp(min=1e-6)
        r_connect  = _K_CONNECT * (d_min / d_max)

        # ── 5. Valence satisfaction ────────────────────────────────────────
        target_val = self.val[z]
        val_excess = (p_bond.sum(-1) - target_val).clamp(min=0.0)
        r_val      = -(val_excess.pow(2) * real_f).sum(-1) / n_real

        # ── 6. Atom count ──────────────────────────────────────────────────
        deficit = (_N_TARGET - n_real).clamp(min=0.0)
        surplus = (n_real - _N_THRESH).clamp(min=0.0)
        r_atom  = -_K_DEFICIT * deficit - _K_SURPLUS * surplus

        return (1.5 * r_morse
              + 1.0 * r_lj
              + 0.5 * r_angle
              + 1.0 * r_connect
              + 2.0 * r_val
              + 1.0 * r_atom)   # [M, B]


# ---------------------------------------------------------------------------
# RDKit validity + quality bonus (CPU, thread-pool)
# ---------------------------------------------------------------------------

def _parse_mol_simple(args):
    coords_np, atomic_nums_np, w_valid = args
    real        = atomic_nums_np != 0
    real_coords = coords_np[real]
    real_nums   = atomic_nums_np[real]

    if len(real_nums) < 2:
        return -w_valid * 2.0, None

    try:
        edit = Chem.RWMol()
        conf = Chem.Conformer(len(real_nums))
        for i, z in enumerate(real_nums):
            sym = _ELEM_SYM.get(int(z), 'C')
            idx = edit.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(idx, real_coords[i].tolist())
        mol = edit.GetMol()
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(mol)
        Chem.AssignRadicals(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return -w_valid * 0.5, None

    n_frags   = len(Chem.GetMolFrags(mol))
    validity  = w_valid * (2.0 if n_frags == 1 else 0.2)
    valid_mol = mol if n_frags == 1 else None
    quality   = 0.0

    if valid_mol is not None:
        try:
            qed  = Descriptors.qed(mol)
            mw   = Descriptors.MolWt(mol)
            mw_s = float(np.exp(-0.5 * ((mw - 130.0) / 80.0) ** 2))
            quality = 0.5 * (0.7 * qed + 0.3 * mw_s)
        except Exception:
            pass

    return validity + quality, valid_mol


def _rdkit_validity_batch(
    coords     : torch.Tensor,
    atom_types : torch.Tensor,
    vocab      : torch.Tensor,
    scale      : float,
    w_valid    : float,
    device,
) -> torch.Tensor:
    B          = coords.shape[0]
    atomic_nums = vocab[atom_types.argmax(-1)]
    real_f      = (atomic_nums != 0).float().unsqueeze(-1)
    n_real      = real_f.sum(1, keepdim=True).clamp(min=1)
    centroid    = (coords * real_f).sum(1, keepdim=True) / n_real
    coords_c    = (coords - centroid) * scale

    coords_np = coords_c.detach().cpu().float().numpy()
    atoms_np  = atomic_nums.detach().cpu().numpy()
    args      = [(coords_np[b], atoms_np[b], w_valid) for b in range(B)]
    results   = list(_THREAD_POOL.map(_parse_mol_simple, args))

    return torch.tensor([r[0] for r in results], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_reward(
    mols,
    rewarder   : EnergyRewarder,
    vocab      : torch.Tensor,
    vdW        = None,           # kept for API compatibility; unused
    w_geom     : float = 1.0,
    w_valid    : float = 2.0,
    w_quality  : float = 1.5,   # folded into _parse_mol_simple
    w_diversity: float = 2.0,
    scale      : float = 2.2,
    debug      : bool  = False,
) -> torch.Tensor:               # [3, B]  row 0=start, 1=anchor, 2=last
    start, anchor, last = mols
    device = start[0].device

    # ── Geometric energy reward ────────────────────────────────────────────
    coords_all = torch.stack((start[0], anchor[0], last[0]))   # [3, B, N, 3]
    atoms_all  = torch.stack((start[1], anchor[1], last[1]))   # [3, B, N, A]
    atoms_num  = vocab[atoms_all.argmax(-1, keepdim=True)]     # [3, B, N, 1]

    real_f3d  = (atoms_num > 0).float()
    n_real_3  = real_f3d.sum(2, keepdim=True).clamp(min=1)
    centroid  = (coords_all * real_f3d).sum(2, keepdim=True) / n_real_3
    coords_c  = (coords_all - centroid) * scale                # [3, B, N, 3]

    geom    = rewarder.compute(coords_c, atoms_num)            # [3, B]
    rewards = w_geom * geom.clone()

    # ── Repulsion buffer: update from ALL THREE positions ─────────────────
    # Previously only `last` was added to the buffer.  Over time this filled
    # the buffer with end-of-trajectory molecules, creating disproportionate
    # diversity pressure on `last` and inverting the reward ordering
    # (start > anchor > last).  Adding all three symmetrically prevents this.
    real_mask_3 = (atoms_num.squeeze(-1) > 0)                  # [3, B, N]
    for m in range(3):
        fps_m = rewarder.buffer.fingerprint(coords_c[m], real_mask_3[m])
        rewarder.buffer.update(fps_m)

    # ── Diversity reward: FLAT weight for all three ────────────────────────
    # Diversity is a novelty bonus; it must not create a reward ordering
    # between start/anchor/last.  Progressive weights were the mechanism
    # that caused the inversion — flat weights decouple diversity from SDPO
    # advantage ordering entirely.
    for m in range(3):
        fps_m = rewarder.buffer.fingerprint(coords_c[m], real_mask_3[m])
        div   = rewarder.buffer.diversity_reward(fps_m, w=w_diversity * _DIV_W_FLAT)
        rewards[m] += div

    # ── RDKit validity + quality: progressive weights (intentional) ────────
    # This IS asymmetric by design: last gets 2× the validity/quality signal
    # as start.  This is the mechanism that makes R(last) > R(start) in
    # expectation.  Only diversity is flattened.
    for m, (mol_pair, w_rdkit) in enumerate(zip([start, anchor, last], _RDKIT_W)):
        val = _rdkit_validity_batch(
            mol_pair[0], mol_pair[1], vocab,
            scale=scale, w_valid=w_valid, device=device,
        )
        rewards[m] += w_rdkit * val

    if debug:
        for i, lbl in enumerate(("start", "anchor", "last")):
            print(f"[energy/{lbl}]  geom={w_geom*geom[i].mean():.3f}  "
                  f"total={rewards[i].mean():.3f}")

    return rewards   # [3, B]