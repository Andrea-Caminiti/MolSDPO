import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds, Descriptors

RDLogger.DisableLog('rdApp.*')

# Pre-cache element symbols so we never call GetPeriodicTable inside a loop
_PT = Chem.GetPeriodicTable()
_ELEMENT_SYMBOLS = {z: _PT.GetElementSymbol(z) for z in [1, 6, 7, 8, 9]}


# ---------------------------------------------------------------------------
# Tier 1: Dense geometric reward
# ---------------------------------------------------------------------------

class GeometricReward:
    """
    Fully batched GPU geometric reward.

    Key speed improvements over v1:
      - Radius / valence lookup uses pre-built GPU tensors indexed by atomic
        number (one gather call) instead of a Python list comprehension over
        M*B*N atoms on every forward pass.
      - real_mask.float() computed once and reused.
      - is_bonded / is_non_bonded derived from a single boolean mask via ~.
      - eye and lookup tables allocated once in __init__ and reused.
    """

    # Maximum atomic number we ever index; covers H(1) C(6) N(7) O(8) F(9)
    _MAX_Z = 10

    def __init__(self, device="cuda"):
        self.device = device

        vdw_defaults = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47}
        cov_defaults = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
        val_defaults = {1: 1,    6: 4,    7: 3,    8: 2,    9: 1}

        pt = Chem.GetPeriodicTable()

        # Build fixed-size lookup tensors of shape [_MAX_Z + 1] on CPU first,
        # then move to device.  Index with atom_types directly — no Python loop
        # at inference time.
        vdw_table = torch.full((self._MAX_Z + 1,), 1.5)
        cov_table = torch.full((self._MAX_Z + 1,), 0.77)
        val_table = torch.full((self._MAX_Z + 1,), 4.0)

        for z in range(1, self._MAX_Z + 1):
            try:
                vdw_table[z] = pt.GetRvdw(z)
                cov_table[z] = pt.GetRcovalent(z)
                val_table[z] = pt.GetDefaultValence(z)
            except Exception:
                vdw_table[z] = vdw_defaults.get(z, 1.5)
                cov_table[z] = cov_defaults.get(z, 0.77)
                val_table[z] = val_defaults.get(z, 4.0)

        # Padding atom (z=0) gets zero radii / valence so it never participates
        vdw_table[0] = 0.0
        cov_table[0] = 0.0
        val_table[0] = 0.0

        self.register_buffer('_vdw_table', vdw_table)
        self.register_buffer('_cov_table', cov_table)
        self.register_buffer('_val_table', val_table)

        # eye is fixed for a given N; we cache the last one seen
        self._cached_eye_N = -1
        self._cached_eye   = None

    def register_buffer(self, name, tensor):
        """Mimic nn.Module.register_buffer without subclassing it."""
        setattr(self, name, tensor.to(self.device))

    def _get_eye(self, N):
        if N != self._cached_eye_N:
            self._cached_eye_N = N
            self._cached_eye   = torch.eye(N, dtype=torch.bool, device=self.device)
        return self._cached_eye

    def compute_reward(self, pos, atom_types):
        """
        pos        : [M, B, N, 3]  coordinates in Angstroms
        atom_types : [M, B, N, 1]  atomic numbers (0 = padding)
        Returns    : [M, B]
        """
        M, B, N, _ = pos.shape

        # ── Lookup radii/valence from pre-built tables (single gather, no loop)
        z_idx = atom_types.squeeze(-1).long().clamp(0, self._MAX_Z)  # [M, B, N]
        radii_vdw = self._vdw_table[z_idx]   # [M, B, N]
        radii_cov = self._cov_table[z_idx]
        valences  = self._val_table[z_idx]

        real_mask = z_idx > 0                # [M, B, N]  True = real atom
        real_f    = real_mask.float()        # reused below — compute once

        # ── Pairwise distances
        dist_matrix = torch.cdist(pos, pos)  # [M, B, N, N]

        eye       = self._get_eye(N)         # [N, N]
        real_pair = real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)  # [M,B,N,N]

        # ── Bond identification  (combined into one mask derivation)
        bond_threshold = (radii_cov.unsqueeze(-1) + radii_cov.unsqueeze(-2)) * 1.15
        target_dists   = radii_cov.unsqueeze(-1) + radii_cov.unsqueeze(-2)

        is_bonded = (
            (dist_matrix > 0.1) &
            (dist_matrix < bond_threshold) &
            ~eye &
            real_pair
        )
        # non-bonded = everything that isn't bonded, isn't self, is real
        is_non_bonded = ~is_bonded & ~eye & real_pair

        # ── Bond length reward
        bond_diffs  = (dist_matrix - target_dists).pow(2)
        n_bonds     = is_bonded.float().sum(dim=(-2, -1)).clamp(min=1.0)
        bond_reward = -(bond_diffs * is_bonded).sum(dim=(-2, -1)) / n_bonds

        # ── Steric clash reward
        vdw_sum      = (radii_vdw.unsqueeze(-1) + radii_vdw.unsqueeze(-2)) * 0.9
        clash_val    = (vdw_sum - dist_matrix).clamp(min=0.0)
        n_real_per_mol = real_f.sum(dim=-1).clamp(min=1.0)   # [M, B]
        clash_reward = -(clash_val.pow(2) * is_non_bonded).sum(dim=(-2, -1)) \
                       / n_real_per_mol

        # ── Valence reward
        current_valence = is_bonded.float().sum(dim=-1)      # [M, B, N]
        valence_excess  = (current_valence - valences).clamp(min=0.0)
        valence_reward  = -valence_excess.pow(2).sum(dim=-1) / n_real_per_mol

        # ── Fragment penalty (isolated atoms proxy)
        has_bond        = is_bonded.any(dim=-1) & real_mask  # [M, B, N]
        isolated        = real_mask & ~has_bond
        fragment_reward = -(isolated.float().sum(dim=-1) / n_real_per_mol)

        # ── Coverage reward (anti-collapse)
        coverage_reward = real_f.mean(dim=-1) * 0.5

        return (bond_reward
                + 5.0 * clash_reward
                + 2.0 * valence_reward
                + 3.0 * fragment_reward
                + coverage_reward)           # [M, B]


# ---------------------------------------------------------------------------
# RDKit parsing helper  (called once per molecule, results shared by T2 + T3)
# ---------------------------------------------------------------------------

def _parse_molecule(args):
    """
    Worker function for ThreadPoolExecutor.
    Parses one molecule and returns (validity_reward, quality_reward).

    Optimisations vs v1:
      - Called ONCE per molecule; validity + quality computed together so
        _coords_atoms_to_rdkit is never called twice for the same input.
      - Pre-cached _ELEMENT_SYMBOLS avoids GetPeriodicTable() inside the loop.
      - DetermineBondOrders is skipped (it is the slowest RDKit call by far
        and fails on ~30% of inputs anyway). DetermineConnectivity + sanitize
        is sufficient to identify validity and compute QED/MW.
      - Padding atoms (z==0) are stripped before building the RDKit object,
        so the conformer and atom list are always compact.
    """
    coords_np, atomic_nums_np, w_valid, w_quality = args

    # Strip padding
    real = atomic_nums_np != 0
    real_coords = coords_np[real]
    real_nums   = atomic_nums_np[real]

    if len(real_nums) < 2:
        return 0.0, 0.0

    try:
        edit = Chem.RWMol()
        conf = Chem.Conformer(len(real_nums))
        for i, z in enumerate(real_nums):
            sym = _ELEMENT_SYMBOLS.get(int(z), 'C')
            idx = edit.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(idx, real_coords[i].tolist())

        mol = edit.GetMol()
        mol.AddConformer(conf, assignId=True)

        # DetermineConnectivity only — much faster than DetermineBondOrders
        rdDetermineBonds.DetermineConnectivity(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return 0.0, 0.0

    # Validity tier
    n_frags = len(Chem.GetMolFrags(mol))
    validity = w_valid * (2.0 if n_frags == 1 else 0.5)

    # Quality tier (only for single-fragment valid molecules)
    quality = 0.0
    if n_frags == 1 and w_quality > 0.0:
        try:
            qed = Descriptors.qed(mol)
            mw  = Descriptors.MolWt(mol)
            mw_score = float(np.clip(1.0 - abs(mw - 80.0) / 120.0, 0.0, 1.0))
            quality  = w_quality * (0.6 * qed + 0.4 * mw_score)
        except Exception:
            pass

    return validity, quality


# Module-level thread pool — created once, reused across training steps.
# Avoids repeated pool creation/teardown overhead.
# Size matches typical CPU core count; tune to your machine.
_THREAD_POOL = ThreadPoolExecutor(max_workers=16)


def rdkit_rewards_batch(coords, atom_types, vocab, scale=2.2,
                        w_valid=1.0, w_quality=0.5):
    """
    Compute validity + quality rewards for a whole batch in parallel.

    Optimisations vs v1:
      - Single function replaces two serial loops (rdkit_validity_reward +
        quality_reward), eliminating the double-parse of each molecule.
      - ThreadPoolExecutor with a persistent pool distributes RDKit work
        across CPU cores. RDKit's C++ internals release the GIL for the
        heavy parts (distance geometry, sanitization), so threading gives
        genuine parallelism here.
      - All tensor→numpy conversion happens once before dispatch.

    Returns:
        bonus : [B] float tensor   (validity + quality combined)
    """
    B = coords.shape[0]

    # Convert to numpy once for the whole batch before spawning threads
    coords_np   = (coords * scale).detach().cpu().float().numpy()     # [B, N, 3]
    atomic_nums = vocab[atom_types.argmax(-1)].detach().cpu().numpy() # [B, N]

    args = [
        (coords_np[b], atomic_nums[b], w_valid, w_quality)
        for b in range(B)
    ]

    results = list(_THREAD_POOL.map(_parse_molecule, args))

    validity_arr = torch.tensor([r[0] for r in results],
                                dtype=torch.float32, device=coords.device)
    quality_arr  = torch.tensor([r[1] for r in results],
                                dtype=torch.float32, device=coords.device)

    return validity_arr + quality_arr   # [B]


# ---------------------------------------------------------------------------
# Unified get_reward  (drop-in replacement)
# ---------------------------------------------------------------------------

def get_reward(
    mols,
    rewarder,        # GeometricReward instance
    vocab,
    vdW,             # kept for API compatibility, no longer used directly
    w_geom    = 1.0,
    w_valid   = 1.0,
    w_quality = 0.5,
    scale     = 2.2,
):
    """
    Returns [3, B]  matching the old API  (rows: start, anchor, last).
    """
    start, anchor, last = mols

    # ── Tier 1: geometric reward for all three time-points (GPU, batched) ──
    coords_all = torch.stack((start[0], anchor[0], last[0]))       # [3, B, N, 3]
    atoms_all  = torch.stack((start[1], anchor[1], last[1]))       # [3, B, N, A]
    atoms_num  = vocab[atoms_all.argmax(-1, keepdim=True)]         # [3, B, N, 1]

    geom_rewards = rewarder.compute_reward(coords_all * scale, atoms_num)  # [3, B]

    # ── Tier 2 + 3: RDKit bonus on the LAST molecule, parallelised ─────────
    bonus = rdkit_rewards_batch(
        last[0], last[1], vocab, scale=scale,
        w_valid=w_valid, w_quality=w_quality
    )   # [B]

    rewards        = w_geom * geom_rewards   # [3, B]
    rewards[2]    += bonus                   # add sparse bonus at last step only
    return rewards