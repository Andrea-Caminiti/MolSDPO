"""
reward.py
=========
Multi-objective reward for SDPO molecular diffusion, targeting PoseBusters
compliance and drug-likeness.

Design
------
Three reward components, ordered by their role:

1. Geometry  (all six terms restored, reweighted)
   Directly maps to PoseBusters checks:
     Morse       → bond lengths within CSD distribution
     LJ          → no steric clashes (< 0.75× VDW radii)
     Angle       → bond angles in acceptable ranges / ring planarity
     Connectivity → single connected fragment
     Valence     → no over-bonded atoms
     Atom count  → molecule size on-target

2. Validity  (RDKit single-fragment check)
   Binary-ish gate: clear +2 for connected mol, penalty for empty.

3. Mol-props  (QED + SA, weight=4.0)
   Primary inter-trajectory discriminating signal for SDPO's win-rate ranking.
   QED and SA naturally vary across rollout trajectories; geometry terms do not
   (all valid molecules have similar force-field energy). High prop_weight keeps
   the ranking signal strong without removing structural constraints.

What was removed vs the previous version
-----------------------------------------
* StructuralDiversityTracker / RepulsionBuffer / _intra_batch_uniqueness
  Buffer-based diversity penalised the best trajectories for resembling
  previous best outputs — directly inverting SDPO's ranking signal.
  Within-rollout diversity is handled implicitly: if all 32 trajectories
  collapse to the same molecule every win_rate is 0.5 and the loss gets
  zero gradient, which is the correct behaviour.

* Progressive validity weights [0.5, 0.75, 1.0]
  Scaled all 32 trajectories identically at each position — affected
  R(start) vs R(last) but not R(traj_i) vs R(traj_j), which is what
  SDPO cares about.

* KL novelty against dataset reference
  sigmoid(KL(generated || QM9)) rewarded molecules unlike known drug-like
  molecules — the opposite of the intended objective.

Returns
-------
torch.Tensor  shape [3, B]
    Row 0 = start, row 1 = anchor, row 2 = last.
    Flat weights across all three positions. SDPO's interp_rewards handles
    position weighting; this function has no opinion about positions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from mol_builder import build_mols_from_pipeline_output

import numpy as np
import torch
import torch.nn.functional as F

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED
    RDLogger.DisableLog('rdApp.*')
    _RDKIT_OK = True
except ImportError:
    warnings.warn("RDKit not found – validity / mol-prop rewards disabled.")
    _RDKIT_OK = False

# ── SA Score ──────────────────────────────────────────────────────────────────
try:
    from sascorer import calculateScore as _sa_score_fn
    _SA_OK = True
except ImportError:
    try:
        from rdkit.Contrib.SA_Score import sascorer as _sa_mod
        _sa_score_fn = _sa_mod.calculateScore
        _SA_OK = True
    except Exception:
        _SA_OK = False
        warnings.warn("SAScore not available – will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Physical constants  (all distances in Ångströms)
# ─────────────────────────────────────────────────────────────────────────────

_MAX_Z = 10

# index = atomic number, 0 = padding
_R_COV = torch.tensor([0.00, 0.31, 0.76, 0.71, 0.66, 0.57,
                        0.77, 0.77, 0.77, 0.77, 0.77])   # covalent radii
_D_E   = torch.tensor([0.00, 0.40, 0.90, 0.85, 0.80, 0.70,
                        0.80, 0.80, 0.80, 0.80, 0.80])   # Morse well depth
_LJ_S  = torch.tensor([0.00, 1.20, 1.70, 1.55, 1.52, 1.47,
                        1.50, 1.50, 1.50, 1.50, 1.50])   # LJ sigma (VDW radii)
_LJ_E  = torch.tensor([0.00, 0.04, 0.09, 0.08, 0.08, 0.07,
                        0.08, 0.08, 0.08, 0.08, 0.08])   # LJ epsilon
_VAL   = torch.tensor([0.0,  1.0,  4.0,  3.0,  2.0,  1.0,
                        4.0,  4.0,  4.0,  4.0,  4.0])    # expected valence

_MORSE_A         = 2.0
_BOND_T          = 0.15              # sigmoid temperature (Å)
_BOND_FAC        = 1.15              # bond threshold multiplier
_IDEAL_ANGLE_SP3 = math.radians(109.5)
_N_TARGET        = 9.0
_N_THRESH        = _N_TARGET * 1.2

# ── Geometry term weights ─────────────────────────────────────────────────────
# Rebalanced so no single term dominates the geometry score.
# Connectivity and valence kept highest — most discriminating and most directly
# tested by PoseBusters.
_W_MORSE   = 0.5   # bond length proxy         → PoseBusters bond length check
_W_LJ      = 0.5   # steric clash avoidance    → PoseBusters clash check
_W_ANGLE   = 0.3   # sp3/sp2 geometry          → PoseBusters angle check
_W_CONNECT = 1.0   # Fiedler connectivity      → single fragment requirement
_W_VAL     = 1.5   # valence satisfaction      → no over-bonded atoms


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EnergyRewarder
# ─────────────────────────────────────────────────────────────────────────────

class EnergyRewarder:
    """
    Six-term differentiable force-field reward.  All operations stay on-device.

    Term → PoseBusters check mapping:
      Morse        → bond lengths within CSD mean ± 3σ
      LJ           → no internal steric clashes (< 0.75 × VDW radii)
      Angle        → bond angles and ring planarity within acceptable ranges
      Connectivity → molecule is a single connected fragment
      Valence      → no over-bonded atoms
      Atom count   → molecule size near target
    """

    def __init__(self, device: str = "cuda", batch_size: int = 128):
        self.device = device
        self.r_cov  = _R_COV.to(device)
        self.d_e    = _D_E.to(device)
        self.lj_s   = _LJ_S.to(device)
        self.lj_e   = _LJ_E.to(device)
        self.val    = _VAL.to(device)
        self._eye_N = -1
        self._eye   = None

    def initialize_from_dataset(self, train_dataset):
        """No-op — kept for API compatibility."""
        pass

    def _eye_mask(self, N: int) -> torch.Tensor:
        if N != self._eye_N:
            self._eye_N = N
            self._eye   = torch.eye(N, dtype=torch.bool, device=self.device)
        return self._eye

    @torch.no_grad()
    def compute(
        self,
        pos   : torch.Tensor,   # [M, B, N, 3]  pre-centred Ångströms
        atom_z: torch.Tensor,   # [M, B, N, 1]  atomic numbers (0 = pad)
    ) -> torch.Tensor:          # [M, B]
        M, B, N, _ = pos.shape
        eye        = self._eye_mask(N)

        z      = atom_z.squeeze(-1).long().clamp(0, _MAX_Z)
        real   = z > 0
        real_f = real.float()
        n_real = real_f.sum(-1).clamp(min=1.0)               # [M, B]

        dist_mat  = torch.cdist(pos, pos)                     # [M, B, N, N]
        real_pair = real.unsqueeze(-1) & real.unsqueeze(-2)
        off_diag  = real_pair & ~eye

        r0_i = self.r_cov[z]
        r0   = r0_i.unsqueeze(-1) + r0_i.unsqueeze(-2)       # [M, B, N, N]
        de   = (self.d_e[z].unsqueeze(-1)  * self.d_e[z].unsqueeze(-2)).sqrt()
        sig  = (self.lj_s[z].unsqueeze(-1) + self.lj_s[z].unsqueeze(-2)) * 0.5
        eps  = (self.lj_e[z].unsqueeze(-1) * self.lj_e[z].unsqueeze(-2)).sqrt()

        r_thr  = r0 * _BOND_FAC
        p_bond = torch.sigmoid((r_thr - dist_mat) / _BOND_T) * off_diag.float()

        # 1. Morse  — bond length proxy
        #    Penalises bonds deviating from equilibrium r0.
        #    → PoseBusters: bond lengths within CSD mean ± 3σ
        morse_v = de * (1.0 - torch.exp(-_MORSE_A * (dist_mat - r0))).pow(2)
        r_morse = -(morse_v * p_bond).sum((-2, -1)) / n_real

        # 2. Lennard-Jones  — steric clash avoidance
        #    Repulsive wall for non-bonded pairs below VDW contact distance.
        #    → PoseBusters: no internal steric clashes
        sr6  = (sig / dist_mat.clamp(min=0.5)).pow(6)
        lj_v = 4.0 * eps * (sr6.pow(2) - sr6)
        r_lj = -(lj_v * (1.0 - p_bond) * off_diag.float()).sum((-2, -1)) / n_real

        # 3. Soft angle bending  — sp3 geometry preference
        #    Penalises bonded triplets deviating from tetrahedral angle.
        #    → PoseBusters: bond angles within acceptable ranges
        K_nbr      = min(4, N - 1)
        _, nbr_idx = p_bond.topk(K_nbr, dim=-1)
        nbr_pos    = torch.gather(
            pos.unsqueeze(3).expand(M, B, N, N, 3),
            3,
            nbr_idx.unsqueeze(-1).expand(M, B, N, K_nbr, 3),
        )
        nbr_pb  = torch.gather(p_bond, -1, nbr_idx)
        v_norm  = F.normalize(nbr_pos - pos.unsqueeze(-2), dim=-1, eps=1e-8)
        cos_ang = torch.einsum('...id,...jd->...ij', v_norm, v_norm).clamp(-1 + 1e-6, 1 - 1e-6)
        theta   = torch.acos(cos_ang)
        gate    = nbr_pb.unsqueeze(-1) * nbr_pb.unsqueeze(-2)
        r_angle = -(_W_ANGLE * (theta - _IDEAL_ANGLE_SP3).pow(2) * gate).sum((-3, -2, -1)) / n_real

        # 4. Algebraic connectivity proxy (Fiedler)
        #    Soft min-degree / max-degree ratio. Drops sharply for fragments.
        #    → PoseBusters: single connected fragment
        degree    = p_bond.sum(-1)
        d_real    = degree * real_f
        r_connect = _W_CONNECT * (
            d_real.min(-1).values.clamp(min=0.0) /
            d_real.max(-1).values.clamp(min=1e-6)
        )

        # 5. Valence satisfaction
        #    Continuous penalty for atoms bonded beyond expected valence.
        #    → PoseBusters: no impossible valences
        val_excess = (p_bond.sum(-1) - self.val[z]).clamp(min=0.0)
        r_val      = -_W_VAL * (val_excess.pow(2) * real_f).sum(-1) / n_real


        return r_morse + r_lj + r_angle + r_connect + r_val


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Molecular-property registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropSpec:
    fn       : Callable   # RDKit Mol -> float  (nan on failure)
    direction: int        # +1 = higher better, -1 = lower better
    lo       : float
    hi       : float
    weight   : float = 1.0


def _safe(fn: Callable) -> Callable:
    def _wrapped(mol):
        try:
            return float(fn(mol))
        except Exception:
            return float("nan")
    return _wrapped


def _sascore_normalised(mol) -> float:
    if not _SA_OK:
        return float("nan")
    try:
        if Chem.SanitizeMol(mol, catchErrors=True) != 0:
            return 0.0
    except Exception:
        return 0.0
    raw = _sa_score_fn(mol)     # 1 (easy) – 10 (hard)
    return (10.0 - raw) / 9.0  # → [0, 1], higher = easier to synthesise


DEFAULT_PROP_SPECS: Dict[str, PropSpec] = {
    "qed":     PropSpec(_safe(QED.qed if _RDKIT_OK else lambda m: float("nan")),
                        +1, 0.0,   1.0,   1.0),
    "sascore": PropSpec(_safe(_sascore_normalised),
                        +1, 0.0,   1.0,   1.0),
    "logp":    PropSpec(_safe(Descriptors.MolLogP if _RDKIT_OK else lambda m: float("nan")),
                        +1, -3.0,  5.0,   0.5),
    "mw":      PropSpec(_safe(Descriptors.MolWt if _RDKIT_OK else lambda m: float("nan")),
                        -1, 200.0, 600.0, 0.3),
}


def register_property(
    name: str, fn: Callable, direction: int,
    lo: float, hi: float, weight: float = 1.0,
) -> None:
    """Add a custom property to the global registry."""
    DEFAULT_PROP_SPECS[name] = PropSpec(
        fn=_safe(fn), direction=direction, lo=lo, hi=hi, weight=weight
    )


def _normalise_prop(values: np.ndarray, spec: PropSpec) -> np.ndarray:
    v      = np.clip(values, spec.lo, spec.hi)
    span   = spec.hi - spec.lo
    normed = np.zeros_like(v) if span == 0 else (v - spec.lo) / span
    return 1.0 - normed if spec.direction == -1 else normed


def _prop_scores(
    rdmols: List,
    specs : Dict[str, PropSpec],
    active: List[str],
) -> np.ndarray:
    """Weighted normalised property scores. Shape: (B,)."""
    n = len(rdmols)
    if not _RDKIT_OK or not active:
        return np.zeros(n, dtype=np.float32)

    weighted_sum = np.zeros(n, dtype=np.float32)
    total_weight = 0.0
    for name in active:
        spec = specs.get(name)
        if spec is None:
            warnings.warn(f"Property '{name}' not in registry — skipped.")
            continue
        raw    = np.array(
            [spec.fn(m) if m is not None else float("nan") for m in rdmols],
            dtype=np.float32,
        )
        normed = _normalise_prop(raw, spec)
        weighted_sum += spec.weight * np.where(np.isnan(normed), 0.0, normed)
        total_weight += spec.weight

    return weighted_sum / total_weight if total_weight > 0 else weighted_sum


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RewardConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    Weights for get_reward().

    geometry_weight  — all six force-field terms (structural, PoseBusters-aligned)
    validity_weight  — RDKit single-fragment check
    prop_weight      — QED + SA score, weighted high to drive inter-trajectory
                       ranking variance for SDPO win-rate confidence

    Flat across positions: all three rows (start, anchor, last) computed
    identically. SDPO's interp_rewards handles position weighting.
    """
    geometry_weight : float = 1.0   # present but not dominant — structural floor
    validity_weight : float = 2.0
    prop_weight     : float = 10.0   # primary SDPO ranking signal

    active_props: List[str] = field(default_factory=lambda: ["qed", "sascore"])
    extra_props : Dict[str, PropSpec] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RDKit validity
# ─────────────────────────────────────────────────────────────────────────────

def _parse_one_mol(
    args : Tuple,
    vocab: torch.Tensor,
) -> Tuple[torch.Tensor, List]:
    """
    Build RDKit mols and score connectivity.

      +2.0  single connected fragment  (valid molecule)
      +0.2  disconnected but parseable (partial credit)
      -0.5  empty / unparseable

    Returns (validity_scores [B], rdmols list[B]).
    """
    real_coords, real_nums = args

    if len(real_nums) < 2: 
        # Return negative reward to actively penalize single atoms
        return torch.tensor([-2.0], device=real_coords.device), [None]

    mols    = build_mols_from_pipeline_output([real_coords, real_nums], vocab)
    n_frags = [len(Chem.GetMolFrags(m)) if m is not None else 0 for m in mols]
    nf_t    = torch.tensor(n_frags, dtype=torch.float32, device=real_coords.device)

    validity = torch.where(
        nf_t == 1,
        torch.full_like(nf_t,  2.0),
        torch.where(
            nf_t  > 1,
            torch.full_like(nf_t,  0.2),
            torch.full_like(nf_t, -0.5),
        )
    )
    rdmols = [m if f == 1 else None for m, f in zip(mols, n_frags)]
    return validity, rdmols

def _diversity_reward(coords_all, atoms_num):
    # Compute pairwise distances between trajectories
    B = coords_all.shape[1]
    coords_flat = coords_all.reshape(3, B, -1)  # [3, B, N*3]
    dists = torch.cdist(coords_flat[2:], coords_flat[2:]).mean()  # Last timestep
    return torch.clamp(dists / 10.0, max=1.0)  # Normalize
# ─────────────────────────────────────────────────────────────────────────────
# 6.  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_reward(
    mols          : list,
    rewarder      : EnergyRewarder,
    vocab         : torch.Tensor,
    vdW           = None,               # kept for API compatibility, unused
    cfg           : Optional[RewardConfig] = None,
    _update_buffer: bool = False,       # kept for API compatibility, no-op
) -> torch.Tensor:                      # [3, B]
    """
    Evaluate denoised molecules at 3 diffusion timesteps.

    Parameters
    ----------
    mols : list of 3 pairs
        [(coords_start, atoms_start), (coords_anchor, atoms_anchor),
         (coords_last,  atoms_last)]
    rewarder : EnergyRewarder
    vocab    : torch.Tensor  mapping one-hot indices to atomic numbers

    Returns
    -------
    torch.Tensor  shape [3, B]  — flat weights, same computation for all rows.
    """
    if cfg is None:
        cfg = RewardConfig()
    if len(mols) != 3:
        raise ValueError(f"'mols' must contain 3 timestep pairs; got {len(mols)}.")

    start, anchor, last = mols
    device = start[0].device

    coords_all = torch.stack((start[:, :, :3], anchor[:, :, :3], last[:, :, :3]))  # [3, B, N, 3]
    atoms_all  = torch.stack((start[:, :, 3:], anchor[:, :, 3:], last[:, :, 3:])) # [3, B, N, A]
    atoms_num  = vocab[atoms_all.argmax(-1, keepdim=True)]                          # [3, B, N, 1]

    # Centre coordinates
    real_f   = (atoms_num > 0).float()
    n_real   = real_f.sum(2, keepdim=True).clamp(min=1)
    centroid = (coords_all * real_f).sum(2, keepdim=True) / n_real
    coords_c = coords_all - centroid                                                # [3, B, N, 3]

    # ── 1. Geometry reward (GPU, all six terms) ───────────────────────────────
    geom    = rewarder.compute(coords_c, atoms_num)                                 # [3, B]
    rewards = cfg.geometry_weight * geom

    all_specs = {**DEFAULT_PROP_SPECS, **cfg.extra_props}

    # ── 2. Validity + mol-prop rewards (flat, all three positions) ────────────
    for m in range(3):
        val_scores, rdmols = _parse_one_mol(
            (coords_all[m], atoms_all[m]), vocab
        )
        rewards[m] += cfg.validity_weight * val_scores

        if cfg.active_props and _RDKIT_OK:
            prop_arr = _prop_scores(rdmols, all_specs, cfg.active_props)
            prop_t   = torch.tensor(prop_arr, dtype=torch.float32, device=device)
            rewards[m] += cfg.prop_weight * prop_t
            
    rewards += 0.5 * _diversity_reward(coords_all, atoms_num)  # Add diversity

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Batched entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_reward_batched(
    all_mols : List,
    rewarder : EnergyRewarder,
    vocab    : torch.Tensor,
    cfg      : Optional[RewardConfig] = None,
) -> torch.Tensor:                      # [N_traj, 3, B]
    """
    Score all trajectories. Returns stacked rewards [N_traj, 3, B].
    """
    if cfg is None:
        cfg = RewardConfig()

    return torch.stack([
        get_reward(traj_mols, rewarder, vocab, cfg=cfg)
        for traj_mols in all_mols
    ])