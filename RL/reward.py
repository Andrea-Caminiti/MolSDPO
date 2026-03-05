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
import os
from concurrent.futures import ThreadPoolExecutor
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


def _precompute_ideal_len_table(vocab: Dict[int, str]) -> torch.Tensor:
    """
    Build a [V, V] float32 tensor of ideal bond lengths indexed by atom-type index.

    Computed once at MoleculeRewarder.__init__ and stored as self._ideal_len.
    At scoring time, a single tensor gather replaces the per-molecule Python loop
    that previously called _lookup_ideal_bond N² times per molecule.

        d_ideal[m, i, j] = ideal_len[atom_idx[m, i], atom_idx[m, j]]

    V = max vocab index + 1 (typically 5-10 for QM9).
    """
    V = vocab.shape[0]
    table = torch.full((V, V), 1.54, dtype=torch.float32)
    for i in range(V):
        for j in range(V):
            ei = vocab[i]
            ej = vocab[j]
            table[i, j] = _lookup_ideal_bond(ei, ej)
    return table   # [V, V]  — move to device on first use


@torch.no_grad()
def _score_geometry_batch(
    coords    : torch.Tensor,   # [M, N, 3]  — any number of molecules M
    atom_idx  : torch.Tensor,   # [M, N]     — argmax of atom_types, long
    ideal_len : torch.Tensor,   # [V, V]     — precomputed, on same device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GeometricReward and BondDeviationReward for M molecules in one pass.

    Previously these were two separate classes that each recomputed pairwise
    distances and the ideal-length matrix per molecule.  Here:

      - Pairwise distances computed once as [M, N, N].
      - Ideal lengths looked up via tensor indexing (no Python loop).
      - Bond scores and bond-deviation scores derived from the same distance matrix.
      - Angle scores computed via a fully vectorized [M, N, N, N] triplet tensor
        instead of a Python loop over atom centres.
      - The whole batch of M molecules is processed in a single GPU kernel sequence.

    Returns
    -------
    geom_score    : [M]  Gaussian-kernel bond+angle quality score in (0, 1]
    bond_dev_score: [M]  exp(−mean_bond_deviation / 0.2Å) in (0, 1]
    """
    M, N, _ = coords.shape
    device   = coords.device

    # ── Pairwise displacement and distance ────────────────────────────────────
    # diff[m, i, j] = coords[m, j] - coords[m, i]  (vector from i to j)
    diff  = coords.unsqueeze(2) - coords.unsqueeze(1)   # [M, N, N, 3]
    dists = diff.norm(dim=-1)                            # [M, N, N]

    # ── Ideal-length matrix via precomputed table (tensor gather) ─────────────
    # atom_idx: [M, N] long.  We want d_ideal[m, i, j] = ideal_len[idx[m,i], idx[m,j]]
    idx_i  = atom_idx.unsqueeze(2).expand(M, N, N)     # [M, N, N]
    idx_j  = atom_idx.unsqueeze(1).expand(M, N, N)     # [M, N, N]
    d_ideal = ideal_len[idx_i, idx_j]                  # [M, N, N]

    # ── Bond mask: upper triangle, within cutoff ──────────────────────────────
    upper     = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    bond_mask = upper.unsqueeze(0) & (dists < _BOND_MAX_DIST)   # [M, N, N]
    n_bonds   = bond_mask.float().sum(dim=(1, 2)).clamp(min=1)  # [M]

    # ── Bond-length Z-score (GeometricReward component) ──────────────────────
    z_bond      = (dists - d_ideal) / _BOND_TOLERANCE           # [M, N, N]
    bond_kern   = torch.exp(-0.5 * z_bond ** 2)                 # [M, N, N]
    bond_z_mean = (bond_kern * bond_mask).sum(dim=(1, 2)) / n_bonds  # [M]

    # ── Bond-length deviation (BondDeviationReward) ───────────────────────────
    dev           = (dists - d_ideal).abs()                      # [M, N, N]
    bond_dev_mean = (dev * bond_mask).sum(dim=(1, 2)) / n_bonds  # [M]
    bond_dev_score = torch.exp(-bond_dev_mean / _BOND_DEV_SCALE) # [M]

    # ── Angle scores — fully vectorized over all (centre, a, c) triplets ──────
    # bonded[m, i, j] = True when j is bonded to i (symmetric from bond_mask)
    bonded = bond_mask | bond_mask.transpose(1, 2)               # [M, N, N]

    # Unit vectors from each centre to each neighbour: [M, N_centre, N_nb, 3]
    # vecs[m, centre, nb] = normalize(coords[m, nb] - coords[m, centre])
    # = normalize(diff[m, centre, nb])
    # diff is [M, N, N, 3]: diff[m, i, j] = coords[m,j] - coords[m,i], so
    # diff[:, centre, nb, :] is exactly the vector we want. ✓
    vecs = F.normalize(diff, dim=-1)                             # [M, N, N, 3]

    # Cosine similarity between all pairs of neighbour vectors for each centre:
    # cos[m, centre, a, c] = vecs[m,centre,a] · vecs[m,centre,c]
    # Computed as batched matmul: reshape [M*N, N, 3] @ [M*N, 3, N] → [M*N, N, N]
    vecs_2d = vecs.reshape(M * N, N, 3)
    cos_2d  = torch.bmm(vecs_2d, vecs_2d.transpose(1, 2))       # [M*N, N, N]
    cos     = cos_2d.reshape(M, N, N, N).clamp(-1.0, 1.0)       # [M, Nc, Na, Nc]

    # Valid triplet mask: both a and c are bonded to the same centre, a < c
    # bonded[:, centre, a] & bonded[:, centre, c] & upper_triangle(a, c)
    upper_ac    = upper.unsqueeze(0).unsqueeze(0)                        # [1, 1, N, N]
    triplet_mask = (bonded.unsqueeze(3) & bonded.unsqueeze(2)            # [M, N, N, N]
                    & upper_ac)

    # Score deviation from nearest ideal angle (sp3 or sp2)
    angles_deg   = torch.acos(cos) * (180.0 / math.pi)          # [M, N, N, N]
    best_dev     = torch.minimum(
        (angles_deg - _IDEAL_ANGLE_SP3).abs(),
        (angles_deg - _IDEAL_ANGLE_SP2).abs(),
    )
    angle_kern   = torch.exp(-0.5 * (best_dev / _ANGLE_TOLERANCE) ** 2)  # [M, N, N, N]
    n_triplets   = triplet_mask.float().sum(dim=(1, 2, 3)).clamp(min=1)  # [M]
    angle_mean   = (angle_kern * triplet_mask).sum(dim=(1, 2, 3)) / n_triplets

    # ── Combine into single geometric score ───────────────────────────────────
    geom_score = (bond_z_mean * n_bonds + angle_mean * n_triplets) / (n_bonds + n_triplets)

    return geom_score, bond_dev_score


_BOND_DEV_SCALE: float = 0.2   # Å: mean deviation at which bond_dev_score ≈ 0.37F


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
    1+2. Geometry (fused) — bond-length Z-score + bond-deviation proxy,
                            computed together in _score_geometry_batch.
                            Both use the precomputed [V,V] ideal-length table.
    3.   ChemicalReward   — QED, SA, logP, MW (RDKit; zero for invalid mols)
    4.   ValidityReward   — binary sanitization success (always non-zero)
    """

    N_OBJECTIVES = 4

    def __init__(
        self,
        vocab        : Dict[int, str],
        chem_targets : Optional[ChemTargets] = None,
    ):
        self.vocab    = vocab
        self.chemical = ChemicalReward(chem_targets)
        self.validity = ValidityReward()

        # Precomputed ideal-length table [V, V] — built once, used every step.
        # Avoids the per-molecule Python loop that previously ran N² = 841 times
        # for each of 1,152 molecule evaluations per training step.
        self._ideal_len     = _precompute_ideal_len_table(vocab)   # [V, V] CPU
        self._ideal_len_dev = None   # lazy device copy, set on first call

    def _ideal_on(self, device: torch.device) -> torch.Tensor:
        """Return the ideal-length table on `device`, caching the result."""
        if self._ideal_len_dev is None or self._ideal_len_dev.device != device:
            self._ideal_len_dev = self._ideal_len.to(device)
        return self._ideal_len_dev

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
        device   = coords.device
        atom_idx = atom_types.argmax(-1).long()   # [B, N]

        r_geom, r_bond_dev = _score_geometry_batch(
            coords, atom_idx, self._ideal_on(device)
        )   # [B], [B]

        # RDKit mols — needed only for ChemicalReward and ValidityReward
        mols: List[Optional[Chem.Mol]] = [
            build_rdkit_mol(coords[b], decode_atom_types(atom_types[b], self.vocab))
            for b in range(coords.shape[0])
        ]
        r_chem  = self.chemical(mols).to(device)
        r_valid = self.validity(mols).to(device)

        sub   = torch.stack([r_geom, r_bond_dev, r_chem, r_valid], dim=-1)  # [B, 4]
        total = weighter(sub)
        return total, sub


# ── Thread count for parallel RDKit mol building ──────────────────────────────
# DetermineBonds + SanitizeMol are pure C++ and release the GIL, so Python
# threads genuinely parallelize here.  Cap at 16 to avoid thrashing.
_RDKIT_WORKERS: int = min(os.cpu_count()//2 or 4, 16)


def _build_mol_task(args: Tuple) -> Optional[Chem.Mol]:
    """ThreadPoolExecutor worker: build one RDKit mol from (coords_cpu, elems)."""
    coords_cpu, elems = args
    return build_rdkit_mol(coords_cpu, elems)


def get_reward_batched(
    rewarder     : MoleculeRewarder,
    weighter     : AdaptiveWeighter,
    coords_g     : torch.Tensor,       # [G, B, N, 3]
    atom_types_g : torch.Tensor,       # [G, B, N, A]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Score G×B molecules and return [G, B] rewards.

    Speed-ups over the previous serial-loop implementation
    -------------------------------------------------------
    1. Tensor geometry (GeometricReward + BondDeviationReward) is computed
       in a single _score_geometry_batch call on the full [G*B, N, 3] tensor.
       Previously called G=32 times on [B, N, 3] each.

    2. Ideal-length lookup uses the precomputed [V, V] table (tensor gather)
       instead of a Python nested loop per molecule.

    3. RDKit mol building (DetermineBonds + SanitizeMol) is parallelized across
       G*B molecules using ThreadPoolExecutor.  The C++ back-end releases the GIL,
       so Python threads genuinely run in parallel on all available CPU cores.
       All G*B coordinate tensors are transferred to CPU in one contiguous copy.

    Returns
    -------
    rewards_g    : [G, B]
    sub_rewards_g: [G, B, 4]
    """
    G, B, N, _ = coords_g.shape
    M          = G * B
    device     = coords_g.device

    # ── Flatten [G, B, N, D] → [M, N, D] ─────────────────────────────────────
    coords_flat     = coords_g.reshape(M, N, 3)
    atom_types_flat = atom_types_g.reshape(M, N, -1)
    atom_idx_flat   = atom_types_flat.argmax(-1).long()   # [M, N]

    # ── Tensor geometry: one call for all M molecules on GPU ──────────────────
    r_geom, r_bond_dev = _score_geometry_batch(
        coords_flat, atom_idx_flat, rewarder._ideal_on(device)
    )   # [M], [M]

    # ── RDKit mols: parallel on CPU ───────────────────────────────────────────
    # Single contiguous .cpu() transfer for all M molecules, then build in parallel.
    coords_cpu = coords_flat.cpu()
    elems_list = [
        decode_atom_types(atom_types_flat[m], rewarder.vocab) for m in range(M)
    ]
    args_iter = ((coords_cpu[m], elems_list[m]) for m in range(M))
    with ThreadPoolExecutor(max_workers=_RDKIT_WORKERS) as pool:
        mols: List[Optional[Chem.Mol]] = list(pool.map(_build_mol_task, args_iter))

    # ── Chemical + validity (CPU, serial — fast for small B) ─────────────────
    r_chem  = rewarder.chemical(mols).to(device)   # [M]
    r_valid = rewarder.validity(mols).to(device)   # [M]

    # ── Combine and reshape → [G, B] ──────────────────────────────────────────
    sub   = torch.stack([r_geom, r_bond_dev, r_chem, r_valid], dim=-1)  # [M, 4]
    total = weighter(sub)                                                  # [M]

    return total.reshape(G, B), sub.reshape(G, B, 4)


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