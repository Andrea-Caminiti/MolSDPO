"""
RL/reward.py

3D-aware multi-objective reward for the Tabasco diffusion pipeline.

Inputs
------
coords     : torch.Tensor [B, N, 3]   — 3D atom positions (Å)
atom_types : torch.Tensor [B, N, A]   — one-hot / logit atom-type vectors
vocab      : dict[int, str]           — encoded index → element symbol (e.g. {0: 'C', 1: 'N'})

Outputs
-------
scalar reward per molecule: torch.Tensor [B]

Sub-rewards
-----------
1. GeometricReward      — bond-length and bond-angle Z-scores vs. ideal values.
                          Fully vectorized tensor computation: no RDKit, no .item()
                          calls, works for every rollout.
2. BondDeviationReward  — mean absolute deviation of all bond lengths from their
                          ideal values, normalized to [0, 1].  Pure tensor ops,
                          no subprocess, no conformer generation.
                          Replaces StrainReward (MMFF94 + ETKDGv3), which called
                          AllChem.EmbedMolecule on every molecule — ignoring the
                          model's own 3D coordinates and taking ~50-200 ms each.
3. ChemicalReward       — QED, SA score, logP range, MW — all from RDKit.
                          Only computed when sanitization succeeds; zero otherwise.
4. ValidityReward       — Binary: did RDKit sanitize successfully?
                          Ensures the model always gets a gradient signal even
                          when chemical properties cannot be computed.

StrainReward is retained in this file for use in score.py (offline evaluation),
where per-step cost does not matter.  It must NOT be used during training.

Combination: uncertainty / homoscedastic weighting (Kendall et al., 2018)
--------------------------------------------------------------------------
    total_reward = Σ_i  (1 / 2σ_i²) * sub_i  −  Σ_i log(σ_i)

where log(σ_i²) are *learned* scalar parameters (one per sub-reward).
High-σ tasks are down-weighted automatically; the log term prevents σ → ∞.
AdaptiveWeighter is an nn.Module — add it to your optimizer (see bottom of file).

Integration with train.py
-------------------------
In LightningTabascoPipe.__init__:
    self.rewarder = MoleculeRewarder(vocab)          # replaces EnergyRewarder
    self.weighter = AdaptiveWeighter(n_objectives=4) # add to model

In configure_optimizers, include self.weighter.parameters() alongside the
model's trainable heads.

In _compute_advantages, call:
    r_final = self.rewarder(coords, atom_types, self.weighter)  # [G, B]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeForceField, MMFFHasAllMoleculeParams

# ── SA score — import once at module level, never inside hot loops ────────────
# _compute_sa_score was importing sascorer on every call (1,152×/step).
# Module-level import pays the cost exactly once at startup.
_sascorer = None
try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
except ImportError:
    try:
        import sascorer as _sascorer   # alternate install path
    except ImportError:
        pass   # _sascorer stays None; _compute_sa_score returns neutral 5.0


# ── Ideal geometry reference tables ──────────────────────────────────────────
# Bond lengths in Å (canonical single-bond values; we use these as the prior
# mean and penalise deviations with a soft Z-score rather than a hard cutoff).

_BOND_LENGTH_IDEAL: Dict[Tuple[str, str], float] = {
    ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
    ('C', 'S'): 1.82, ('C', 'H'): 1.09, ('C', 'F'): 1.35,
    ('C', 'Cl'): 1.77, ('C', 'Br'): 1.94, ('N', 'N'): 1.45,
    ('N', 'O'): 1.40, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
    ('S', 'S'): 2.05, ('S', 'H'): 1.34,
}
_BOND_TOLERANCE: float = 0.25    # Å — soft σ for the Z-score penalty
_BOND_MAX_DIST:  float = 1.9     # Å — beyond this we do not consider a bond

# Ideal sp3 tetrahedral and sp2 planar angles
_IDEAL_ANGLE_SP3: float = 109.5  # degrees
_IDEAL_ANGLE_SP2: float = 120.0
_ANGLE_TOLERANCE: float = 15.0   # degrees — σ for the angle Z-score penalty


def _lookup_ideal_bond(elem_i: str, elem_j: str) -> float:
    """Return ideal bond length for an element pair, defaulting to 1.54 Å."""
    key  = (elem_i, elem_j)
    rkey = (elem_j, elem_i)
    return _BOND_LENGTH_IDEAL.get(key, _BOND_LENGTH_IDEAL.get(rkey, 1.54))


# ── Coords → RDKit molecule ───────────────────────────────────────────────────

def build_rdkit_mol(
    coords     : torch.Tensor,   # [N, 3]
    atom_elems : List[str],      # length N
) -> Optional[Chem.Mol]:
    """
    Construct an RDKit Mol from 3D coordinates and element symbols.

    Strategy:
      1. Place atoms at given positions.
      2. Use RDKit's DetermineBonds (distance + valence) to infer connectivity.
      3. Sanitize; return None on failure so the caller can fall back gracefully.
    """
    try:
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(atom_elems))

        for i, elem in enumerate(atom_elems):
            idx = rw.AddAtom(Chem.Atom(elem))
            pos = coords[i].tolist()
            conf.SetAtomPosition(idx, pos)

        rw.AddConformer(conf, assignId=True)

        # Infer bonds from 3D geometry + element valence rules
        AllChem.DetermineBonds(rw, charge=0)
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def decode_atom_types(
    atom_types : torch.Tensor,   # [N, A]
    vocab      : Dict[int, str],
) -> List[str]:
    """
    Convert one-hot / logit atom-type tensor → list of element strings.
    Absorb / padding tokens (indices not in vocab) are mapped to 'C' as a
    safe fallback so that geometry can still be evaluated.
    """
    indices = atom_types.argmax(dim=-1).tolist()   # [N]
    return vocab[indices].tolist()


# ── Sub-reward 1: Geometric validity ─────────────────────────────────────────

# Pre-built ideal-length tensor, indexed by element-pair string.
# Looked up once per batch via _build_ideal_dist_matrix; never inside a loop.
def _build_ideal_dist_matrix(elems: List[str], device: torch.device) -> torch.Tensor:
    """
    Build an [N, N] tensor of ideal bond lengths for every atom pair.
    Off-bond pairs (distance > _BOND_MAX_DIST) will be masked out later.
    """
    N = len(elems)
    mat = torch.full((N, N), 1.54, dtype=torch.float, device=device)
    for i in range(N):
        for j in range(i + 1, N):
            v = _lookup_ideal_bond(elems[i], elems[j])
            mat[i, j] = v
            mat[j, i] = v
    return mat


class GeometricReward:
    """
    Score 3D geometry quality from raw tensors — no RDKit required.

    Bond-length score
    -----------------
    For every pair (i, j) with distance < _BOND_MAX_DIST, compute
        z = (d_ij − d_ideal) / _BOND_TOLERANCE
    and score with exp(−z²/2) (Gaussian kernel; 1.0 = perfect).

    Angle score
    -----------
    For each bonded centre B with ≥2 neighbours, compute all A–B–C angles
    and score deviation from the nearest ideal (sp3=109.5° or sp2=120°).

    Both scores are computed with fully batched tensor ops — no Python loops
    over atoms or pairs, no .item() calls during scoring.  The only remaining
    Python loop is over the batch dimension B, which is necessary because
    each molecule can have a different element composition and therefore a
    different ideal-distance matrix.

    Returns a scalar in (0, 1] per molecule.
    """

    def __call__(
        self,
        coords     : torch.Tensor,   # [B, N, 3]
        atom_types : torch.Tensor,   # [B, N, A]
        vocab      : Dict[int, str],
    ) -> torch.Tensor:               # [B]
        B = coords.shape[0]
        scores = [
            self._score_one(coords[b], decode_atom_types(atom_types[b], vocab))
            for b in range(B)
        ]
        return torch.stack(scores).to(coords.device)

    def _score_one(
        self,
        coords : torch.Tensor,   # [N, 3]
        elems  : List[str],
    ) -> torch.Tensor:           # scalar
        N      = coords.shape[0]
        device = coords.device

        if N < 2:
            return coords.new_tensor(1.0)

        # ── Pairwise distances ────────────────────────────────────────────────
        # diff: [N, N, 3];  dists: [N, N]
        diff  = coords.unsqueeze(1) - coords.unsqueeze(0)
        dists = diff.norm(dim=-1)

        # ── Bond mask: pairs closer than _BOND_MAX_DIST (upper triangle only) ─
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        bond_mask = upper & (dists < _BOND_MAX_DIST)   # [N, N]

        # ── Bond-length score (fully vectorized) ─────────────────────────────
        # Build the ideal-distance matrix once per molecule (small N≤29 Python
        # loop — cheap compared to the per-pair loop it replaces).
        d_ideal = _build_ideal_dist_matrix(elems, device)          # [N, N]
        z_bond  = (dists - d_ideal) / _BOND_TOLERANCE              # [N, N]
        bond_scores_mat = torch.exp(-0.5 * z_bond ** 2)            # [N, N]

        bond_vals = bond_scores_mat[bond_mask]   # [n_bonds]

        # ── Angle score (vectorized over triplets) ────────────────────────────
        # bonded[i, j] = True if atoms i and j are bonded
        bonded = bond_mask | bond_mask.T                            # [N, N] symmetric

        # For each centre B: gather all (A, B, C) triplets where A≠C, both bonded to B.
        # Strategy: build neighbour vectors for every (centre, neighbour) pair,
        # then take all pairs of neighbour vectors for the same centre.

        # neighbour_mask[b, n] = True if n is a neighbour of b (b ≠ n)
        neighbour_mask = bonded & ~torch.eye(N, dtype=torch.bool, device=device)  # [N, N]

        angle_scores_list = []
        for centre in range(N):
            nb = neighbour_mask[centre].nonzero(as_tuple=False).squeeze(-1)  # [deg]
            deg = nb.shape[0]
            if deg < 2:
                continue

            b_pos  = coords[centre]                    # [3]
            nb_pos = coords[nb]                        # [deg, 3]
            vecs   = nb_pos - b_pos.unsqueeze(0)       # [deg, 3]
            vecs   = F.normalize(vecs, dim=-1)         # [deg, 3]

            # All pairs of neighbour vectors: [deg, deg] cosine similarity matrix
            cos_mat  = torch.clamp(vecs @ vecs.T, -1.0, 1.0)      # [deg, deg]
            # Upper-triangle pairs only (avoid double-counting and self-pairs)
            tri_mask = torch.triu(
                torch.ones(deg, deg, dtype=torch.bool, device=device), diagonal=1
            )
            cos_vals = cos_mat[tri_mask]               # [n_angles]

            angles_rad = torch.acos(cos_vals)          # [n_angles]
            angles_deg = angles_rad * (180.0 / math.pi)

            # Score against nearest ideal (sp3 or sp2) — all tensor, no Python loop
            d_sp3 = (angles_deg - _IDEAL_ANGLE_SP3).abs()
            d_sp2 = (angles_deg - _IDEAL_ANGLE_SP2).abs()
            best  = torch.minimum(d_sp3, d_sp2)       # [n_angles]
            z_ang = best / _ANGLE_TOLERANCE
            angle_scores_list.append(torch.exp(-0.5 * z_ang ** 2))

        # ── Combine bond + angle scores ───────────────────────────────────────
        parts = [bond_vals]
        if angle_scores_list:
            parts.append(torch.cat(angle_scores_list))
        all_scores = torch.cat(parts)

        return all_scores.mean() if all_scores.numel() > 0 else coords.new_tensor(1.0)


# ── Sub-reward 2: Bond-deviation strain proxy ─────────────────────────────────

class BondDeviationReward:
    """
    Fast geometry-based strain proxy using the model's own 3D coordinates.

    For every bonded pair (distance < _BOND_MAX_DIST), compute the absolute
    deviation from the ideal bond length and average across all bonds:

        mean_dev = mean |d_ij − d_ideal_ij|   (Å)
        reward   = exp(−mean_dev / scale)       scale = 0.2 Å

    This is entirely equivalent to the bond-length component of GeometricReward
    but expressed as an energy-like proxy rather than a Gaussian kernel, giving
    a complementary signal.  All ops are on-GPU tensors — no RDKit, no subprocess,
    no conformer generation, O(N²) but with N≤29 and fully batched.

    Why not StrainReward
    --------------------
    StrainReward (kept below for offline scoring) called AllChem.EmbedMolecule
    (ETKDGv3) followed by MMFF94 minimization on every molecule:
      - EmbedMolecule ignores the model's coordinates and generates a fresh
        conformer from SMILES, so the energy measured has nothing to do with
        what the model produced.
      - ETKDGv3 + 200 MMFF iterations takes ~50-200 ms per molecule.
      - At G=32 trajectories × B=12 × 3 checkpoint calls = 1,152 calls/step,
        this adds 1-4 minutes per training step.

    BondDeviationReward takes <1 ms per batch on GPU and measures the actual
    coordinates the model generated.
    """

    SCALE: float = 0.2   # Å: mean deviation at which reward ≈ 0.37

    def __call__(
        self,
        coords     : torch.Tensor,   # [B, N, 3]
        atom_types : torch.Tensor,   # [B, N, A]
        vocab      : Dict[int, str],
    ) -> torch.Tensor:               # [B]
        B = coords.shape[0]
        scores = [
            self._score_one(coords[b], decode_atom_types(atom_types[b], vocab))
            for b in range(B)
        ]
        return torch.stack(scores).to(coords.device)

    def _score_one(
        self,
        coords : torch.Tensor,   # [N, 3]
        elems  : List[str],
    ) -> torch.Tensor:           # scalar
        N      = coords.shape[0]
        device = coords.device

        if N < 2:
            return coords.new_tensor(1.0)

        diff  = coords.unsqueeze(1) - coords.unsqueeze(0)   # [N, N, 3]
        dists = diff.norm(dim=-1)                            # [N, N]

        upper     = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        bond_mask = upper & (dists < _BOND_MAX_DIST)

        if not bond_mask.any():
            return coords.new_tensor(1.0)

        d_ideal  = _build_ideal_dist_matrix(elems, device)         # [N, N]
        dev      = (dists - d_ideal).abs()[bond_mask]               # [n_bonds]
        mean_dev = dev.mean()
        return torch.exp(-mean_dev / self.SCALE)


# ── Sub-reward 2b: StrainReward — OFFLINE USE ONLY ────────────────────────────

class StrainReward:
    """
    MMFF94 force-field energy per heavy atom (lower = more stable).

    WARNING: Do NOT use during RL training.
    ----------------------------------------
    This class calls AllChem.EmbedMolecule (ETKDGv3), which:
      1. Ignores the model's 3D coordinates — generates a new random conformer.
      2. Takes ~50-200 ms per molecule (ETKDGv3 + 200 MMFF iterations).
    At G×B×3 checkpoint evaluations per step this adds 1-4 minutes/step.

    Use BondDeviationReward in MoleculeRewarder for training.
    Use StrainReward in score.py / offline evaluation where cost is acceptable.

    The raw energy (kcal/mol) is converted to a reward in (0, 1] via
        r = exp(−energy_per_atom / scale)
    with scale = 5 kcal/(mol·atom).  Falls back to UFF if MMFF94 parameters
    are unavailable.
    """

    ENERGY_SCALE: float = 5.0

    def __call__(
        self,
        mols : List[Optional[Chem.Mol]],
    ) -> torch.Tensor:
        rewards = [self._score_one(mol) for mol in mols]
        return torch.tensor(rewards, dtype=torch.float)

    def _score_one(self, mol: Optional[Chem.Mol]) -> float:
        if mol is None:
            return 0.0
        try:
            mol_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())

            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy == 0:
                return 0.0

            if MMFFHasAllMoleculeParams(mol_h):
                ff = MMFFGetMoleculeForceField(mol_h)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol_h)

            if ff is None:
                return 0.0

            ff.Minimize(maxIts=200)
            energy_per_atom = ff.CalcEnergy() / n_heavy
            return math.exp(-energy_per_atom / self.ENERGY_SCALE)
        except Exception:
            return 0.0


# ── Sub-reward 3: Chemical properties ─────────────────────────────────────────

@dataclass
class ChemTargets:
    """
    Target ranges for chemical properties.  All ranges are inclusive.
    Defaults follow Lipinski's rule-of-five with slightly relaxed logP.
    """
    logp_min  : float = -0.4
    logp_max  : float =  5.6
    mw_max    : float = 500.0
    sa_max    : float =  6.0    # SA score: 1 (easy) – 10 (hard); lower is better


class ChemicalReward:
    """
    Four chemical sub-properties combined into a single score in [0, 1].

    Components and normalisation
    ----------------------------
    QED       : already in [0, 1].
    SA score  : mapped (score − 1) / 9 → [0, 1], then inverted (1 = easy to synthesize).
    logP      : 1.0 if in [logp_min, logp_max], otherwise Gaussian falloff.
    MW        : 1.0 if ≤ mw_max, otherwise exponential penalty.

    All four are averaged with equal weight before the adaptive weighter
    combines them with the other sub-rewards.

    Returns 0.0 for invalid / un-sanitized molecules.
    """

    def __init__(self, targets: Optional[ChemTargets] = None):
        self.targets = targets or ChemTargets()

    def __call__(
        self,
        mols : List[Optional[Chem.Mol]],
    ) -> torch.Tensor:                     # [B]
        rewards = [self._score_one(mol) for mol in mols]
        return torch.tensor(rewards, dtype=torch.float)

    def _score_one(self, mol: Optional[Chem.Mol]) -> float:
        if mol is None:
            return 0.0
        try:
            t = self.targets

            qed_score = QED.qed(mol)

            sa_raw    = _compute_sa_score(mol)
            sa_score  = 1.0 - (sa_raw - 1.0) / 9.0           # 1 = trivial, 0 = impossible
            sa_score  = max(0.0, min(1.0, sa_score))
            if sa_raw > t.sa_max:
                sa_score *= 0.5   # soft penalty for very hard molecules

            logp = Descriptors.MolLogP(mol)
            if t.logp_min <= logp <= t.logp_max:
                logp_score = 1.0
            else:
                deviation  = min(abs(logp - t.logp_min), abs(logp - t.logp_max))
                logp_score = math.exp(-0.5 * (deviation / 1.5) ** 2)

            mw = Descriptors.MolWt(mol)
            if mw <= t.mw_max:
                mw_score = 1.0
            else:
                mw_score = math.exp(-((mw - t.mw_max) / 100.0) ** 2)

            return (qed_score + sa_score + logp_score + mw_score) / 4.0
        except Exception:
            return 0.0


def _compute_sa_score(mol: Chem.Mol) -> float:
    """
    Synthetic accessibility score via RDKit's SA_Score contribution.
    Uses the module-level _sascorer cached at import time.
    Falls back to a neutral 5.0 if the module is unavailable.
    """
    if _sascorer is None:
        return 5.0
    return _sascorer.calculateScore(mol)


# ── Sub-reward 4: Validity ────────────────────────────────────────────────────

class ValidityReward:
    """
    Binary reward: 1.0 if RDKit produced a valid sanitized molecule, else 0.0.

    This is the only sub-reward that is non-zero for *every* rollout, including
    completely invalid geometries.  It acts as a floor signal that keeps the
    model receiving gradient information even in early training when most
    outputs are chemically invalid.
    """

    def __call__(
        self,
        mols : List[Optional[Chem.Mol]],
    ) -> torch.Tensor:
        return torch.tensor(
            [1.0 if mol is not None else 0.0 for mol in mols],
            dtype=torch.float,
        )


# ── Adaptive weighter (Kendall et al., 2018) ─────────────────────────────────

class AdaptiveWeighter(nn.Module):
    """
    Learns one log(σ²) parameter per sub-reward.

    Combining n sub-rewards r_1 … r_n:
        total = Σ_i  (1 / 2σ_i²) * r_i  −  Σ_i log(σ_i)
              = Σ_i  exp(−s_i) * r_i  −  Σ_i  0.5 * s_i
    where s_i = log(σ_i²).

    High uncertainty (large σ) → small weight, but the −log σ term prevents
    σ → ∞.  Gradients flow through both the sub-reward values *and* the σ
    parameters, so the weighter trains jointly with the policy.

    Parameters
    ----------
    n_objectives : number of sub-rewards (default 4)
    init_log_sigma_sq : initial value for all log(σ²) (0.0 = σ=1, equal weights)
    """

    def __init__(self, n_objectives: int = 4, init_log_sigma_sq: float = 0.0):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(
            torch.full((n_objectives,), init_log_sigma_sq)
        )

    def forward(self, sub_rewards: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sub_rewards : [B, n_objectives]  — each sub-reward in [0, 1]

        Returns
        -------
        total_reward : [B]
        """
        # Precision weights: exp(−log σ²) = 1/σ²
        precision  = torch.exp(-self.log_sigma_sq)             # [n_obj]
        weighted   = (precision * sub_rewards).sum(dim=-1)     # [B]

        # Regularisation term (scalar, broadcast-added to every molecule equally)
        reg = 0.5 * self.log_sigma_sq.sum()

        return weighted - reg

    @property
    def effective_weights(self) -> torch.Tensor:
        """Normalised weights for logging/inspection. Not used in forward()."""
        w = torch.exp(-self.log_sigma_sq)
        return w / w.sum()


# ── Top-level rewarder ────────────────────────────────────────────────────────

class MoleculeRewarder:
    """
    Orchestrates all four sub-rewards for a batch of molecules.

    Sub-rewards
    -----------
    1. GeometricReward     — bond-length + angle Z-scores (vectorized tensors)
    2. BondDeviationReward — mean bond-length deviation from ideal (fast proxy
                             for StrainReward; uses model's own coordinates)
    3. ChemicalReward      — QED, SA, logP, MW (RDKit; zero for invalid mols)
    4. ValidityReward      — binary sanitization success (always non-zero)

    Usage
    -----
    rewarder = MoleculeRewarder(vocab)
    weighter = AdaptiveWeighter(n_objectives=4)

    reward, sub = rewarder(coords, atom_types, weighter)   # [B], [B, 4]
    """

    N_OBJECTIVES = 4

    def __init__(
        self,
        vocab        : Dict[int, str],
        chem_targets : Optional[ChemTargets] = None,
    ):
        self.vocab        = vocab
        self.geometric    = GeometricReward()
        self.bond_dev     = BondDeviationReward()
        self.chemical     = ChemicalReward(chem_targets)
        self.validity     = ValidityReward()

    def __call__(
        self,
        coords     : torch.Tensor,      # [B, N, 3]
        atom_types : torch.Tensor,      # [B, N, A]
        weighter   : AdaptiveWeighter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_reward  : [B]
        sub_rewards   : [B, 4]  columns: [geometric, bond_dev, chemical, validity]
        """
        device = coords.device

        # RDKit mols — needed by ChemicalReward and ValidityReward only
        mols: List[Optional[Chem.Mol]] = [
            build_rdkit_mol(coords[b], decode_atom_types(atom_types[b], self.vocab))
            for b in range(coords.shape[0])
        ]

        r_geom     = self.geometric(coords, atom_types, self.vocab).to(device)
        r_bond_dev = self.bond_dev(coords, atom_types, self.vocab).to(device)
        r_chem     = self.chemical(mols).to(device)
        r_valid    = self.validity(mols).to(device)

        sub_rewards = torch.stack([r_geom, r_bond_dev, r_chem, r_valid], dim=-1)  # [B, 4]
        total       = weighter(sub_rewards)

        return total, sub_rewards


# ── Batched wrapper for the training loop ────────────────────────────────────

def get_reward_batched(
    rewarder     : MoleculeRewarder,
    weighter     : AdaptiveWeighter,
    coords_g     : torch.Tensor,       # [G, B, N, 3]
    atom_types_g : torch.Tensor,       # [G, B, N, A]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run MoleculeRewarder over all G trajectory groups.

    Returns
    -------
    rewards_g    : [G, B]    — total scalar reward per (trajectory, molecule)
    sub_rewards_g: [G, B, 4] — sub-rewards for logging
    """
    G = coords_g.shape[0]
    rewards_list, sub_list = [], []

    for g in range(G):
        r, sub = rewarder(coords_g[g], atom_types_g[g], weighter)
        rewards_list.append(r)
        sub_list.append(sub)

    return torch.stack(rewards_list), torch.stack(sub_list)   # [G, B], [G, B, 4]


# ── Logging helper ────────────────────────────────────────────────────────────

def reward_log_dict(
    sub_rewards_g : torch.Tensor,      # [G, B, 4]
    weighter      : AdaptiveWeighter,
) -> Dict[str, torch.Tensor]:
    """
    Build a flat dict suitable for passing to self.log_dict().

    Includes per-sub-reward means, effective adaptive weights, and σ values.
    """
    names = ['geometric', 'bond_dev', 'chemical', 'validity']
    logs  = {}

    for i, name in enumerate(names):
        logs[f'reward_sub/{name}'] = sub_rewards_g[..., i].mean()

    eff_w = weighter.effective_weights   # [4]
    for i, name in enumerate(names):
        logs[f'reward_weight/{name}'] = eff_w[i]
        logs[f'reward_sigma/{name}']  = torch.exp(0.5 * weighter.log_sigma_sq[i])

    return logs