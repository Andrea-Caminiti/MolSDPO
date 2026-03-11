from __future__ import annotations

import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeForceField, MMFFHasAllMoleculeParams

_sascorer = None
try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
except ImportError:
    try:
        import sascorer as _sascorer
    except ImportError:
        pass


_BOND_LENGTH_IDEAL: Dict[Tuple[str, str], float] = {
    ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
    ('C', 'S'): 1.82, ('C', 'H'): 1.09, ('C', 'F'): 1.35,
    ('C', 'Cl'): 1.77, ('C', 'Br'): 1.94, ('N', 'N'): 1.45,
    ('N', 'O'): 1.40, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
    ('S', 'S'): 2.05, ('S', 'H'): 1.34,
}
_BOND_TOLERANCE: float = 0.25
_BOND_MAX_DIST: float = 1.9

_IDEAL_ANGLE_SP3: float = 109.5
_IDEAL_ANGLE_SP2: float = 120.0
_ANGLE_TOLERANCE: float = 15.0


def _lookup_ideal_bond(elem_i: str, elem_j: str) -> float:
    """Return ideal bond length for an element pair, defaulting to 1.54 Å."""
    key = (elem_i, elem_j)
    rkey = (elem_j, elem_i)
    return _BOND_LENGTH_IDEAL.get(key, _BOND_LENGTH_IDEAL.get(rkey, 1.54))


def _precompute_ideal_len_table(vocab: Dict[int, str]) -> torch.Tensor:
    """
    Build a [V, V] float32 tensor of ideal bond lengths indexed by atom-type index.
    """
    V = vocab.shape[0]
    table = torch.full((V, V), 1.54, dtype=torch.float32)
    for i in range(V):
        for j in range(V):
            ei = vocab[i]
            ej = vocab[j]
            table[i, j] = _lookup_ideal_bond(ei, ej)
    return table


@torch.no_grad()
def _score_geometry_batch(
    coords: torch.Tensor,  #[M, N, 3]
    atom_idx: torch.Tensor,  #[M, N]
    ideal_len: torch.Tensor,  #[V, V]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GeometricReward and BondDeviationReward for M molecules.

    Returns
    -------
    geom_score    : [M]  Gaussian-kernel bond+angle quality score in (0, 1]
    bond_dev_score: [M]  exp(−mean_bond_deviation / 0.2Å) in (0, 1]
    """
    M, N, _ = coords.shape
    device = coords.device

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  #[M, N, N, 3]
    dists = diff.norm(dim=-1)  #[M, N, N]

    idx_i = atom_idx.unsqueeze(2).expand(M, N, N)  #[M, N, N]
    idx_j = atom_idx.unsqueeze(1).expand(M, N, N)  #[M, N, N]
    d_ideal = ideal_len[idx_i, idx_j]  #[M, N, N]

    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    bond_mask = upper.unsqueeze(0) & (dists < _BOND_MAX_DIST)  #[M, N, N]
    n_bonds = bond_mask.float().sum(dim=(1, 2)).clamp(min=1)  #[M]

    z_bond = (dists - d_ideal) / _BOND_TOLERANCE  #[M, N, N]
    bond_kern = torch.exp(-0.5 * z_bond ** 2)  #[M, N, N]
    bond_z_mean = (bond_kern * bond_mask).sum(dim=(1, 2)) / n_bonds  #[M]

    dev = (dists - d_ideal).abs()  #[M, N, N]
    bond_dev_mean = (dev * bond_mask).sum(dim=(1, 2)) / n_bonds  #[M]
    bond_dev_score = torch.exp(-bond_dev_mean / _BOND_DEV_SCALE)  #[M]

    bonded = bond_mask | bond_mask.transpose(1, 2)  #[M, N, N]

    vecs = F.normalize(diff, dim=-1)  #[M, N, N, 3]

    vecs_2d = vecs.reshape(M * N, N, 3)
    cos_2d = torch.bmm(vecs_2d, vecs_2d.transpose(1, 2))  #[M*N, N, N]
    cos = cos_2d.reshape(M, N, N, N).clamp(-1.0, 1.0)  #[M, Nc, Na, Nc]

    upper_ac = upper.unsqueeze(0).unsqueeze(0)  #[1, 1, N, N]
    triplet_mask = (bonded.unsqueeze(3) & bonded.unsqueeze(2)  #[M, N, N, N]
                    & upper_ac)

    angles_deg = torch.acos(cos) * (180.0 / math.pi)  #[M, N, N, N]
    best_dev = torch.minimum(
        (angles_deg - _IDEAL_ANGLE_SP3).abs(),
        (angles_deg - _IDEAL_ANGLE_SP2).abs(),
    )
    angle_kern = torch.exp(-0.5 * (best_dev / _ANGLE_TOLERANCE) ** 2)  #[M, N, N, N]
    n_triplets = triplet_mask.float().sum(dim=(1, 2, 3)).clamp(min=1)  #[M]
    angle_mean = (angle_kern * triplet_mask).sum(dim=(1, 2, 3)) / n_triplets

    geom_score = (bond_z_mean * n_bonds + angle_mean * n_triplets) / (n_bonds + n_triplets)

    return geom_score, bond_dev_score


_BOND_DEV_SCALE: float = 0.2  #Å: mean deviation at which bond_dev_score ≈ 0.37


def build_rdkit_mol(
    coords: torch.Tensor,  #[N, 3]
    atom_elems: List[str],  #N
) -> Optional[Chem.Mol]:
    """
    Construct an RDKit Mol from 3D coordinates and element symbols.

    """
    try:
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(atom_elems))

        for i, elem in enumerate(atom_elems):
            idx = rw.AddAtom(Chem.Atom(elem))
            pos = coords[i].tolist()
            conf.SetAtomPosition(idx, pos)

        rw.AddConformer(conf, assignId=True)

        AllChem.DetermineBonds(rw, charge=0)
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def decode_atom_types(
    atom_types: torch.Tensor,  #[N, A]
    vocab: Dict[int, str],
) -> List[str]:
    """
    Convert one-hot / logit atom-type tensor to list of element strings.
    """
    indices = atom_types.argmax(dim=-1).tolist()  #[N]
    return vocab[indices].tolist()

@dataclass
class ChemTargets:
    """
    Target ranges for chemical properties.  All ranges are inclusive.
    Defaults follow Lipinski's rule-of-five with slightly relaxed logP.
    """
    logp_min: float = -0.4
    logp_max: float = 5.6
    mw_max: float = 250.0
    sa_max: float = 6.0


class ChemicalReward:
    """
    Four chemical sub-properties combined into a single score in [0, 1].

    Components and normalisation
    ----------------------------
    QED       : already in [0, 1].
    SA score  : mapped to [0, 1], then inverted (1 = easy to synthesize).
    logP      : 1.0 if in [logp_min, logp_max], otherwise Gaussian falloff.
    MW        : 1.0 if <= mw_max, otherwise exponential penalty.

    All four are averaged with equal weight before the adaptive weighter
    combines them with the other sub-rewards.

    Returns 0.0 for invalid molecules.
    """

    def __init__(self, targets: Optional[ChemTargets] = None):
        self.targets = targets or ChemTargets()

    def __call__(
        self,
        mols: List[Optional[Chem.Mol]],
    ) -> torch.Tensor:  #[B]
        rewards = [self._score_one(mol) for mol in mols]
        return torch.tensor(rewards, dtype=torch.float)

    def _score_one(self, mol: Optional[Chem.Mol]) -> float:
        if mol is None:
            return 0.0
        try:
            t = self.targets

            qed_score = QED.qed(mol)

            sa_raw = _compute_sa_score(mol)
            sa_score = 1.0 - (sa_raw - 1.0) / 9.0
            sa_score = max(0.0, min(1.0, sa_score))
            if sa_raw > t.sa_max:
                sa_score *= 0.5

            logp = Descriptors.MolLogP(mol)
            if t.logp_min <= logp <= t.logp_max:
                logp_score = 1.0
            else:
                deviation = min(abs(logp - t.logp_min), abs(logp - t.logp_max))
                logp_score = math.exp(-0.5 * (deviation / 1.5) ** 2)

            mw = Descriptors.MolWt(mol)
            if mw <= t.mw_max:
                mw_score = 1.0
            else:
                mw_score = math.exp(-((mw - t.mw_max) / 40.0) ** 2)

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


class ValidityReward:
    """
    Binary reward: 1.0 if RDKit produced a valid sanitized molecule, else 0.0.

    """

    def __call__(
        self,
        mols: List[Optional[Chem.Mol]],
    ) -> torch.Tensor:
        return torch.tensor(
            [1.0 if mol is not None else 0.0 for mol in mols],
            dtype=torch.float,
        )


class AdaptiveWeighter(nn.Module):
    """
    Learns one weight parameter per sub-reward.

    Parameters
    ----------
    n_objectives : number of sub-rewards (default 4)
    init_log_sigma_sq : initial value for all log(σ²)
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
        precision = torch.exp(-self.log_sigma_sq)  #[n_obj]
        weighted = (precision * sub_rewards).sum(dim=-1)  #[B]

        reg = 0.5 * self.log_sigma_sq.sum()

        return weighted - reg

    @property
    def effective_weights(self) -> torch.Tensor:
        """Normalised weights for logging/inspection. Not used in forward()."""
        w = torch.exp(-self.log_sigma_sq)
        return w / w.sum()


class MoleculeRewarder:
    """
    Orchestrates all four sub-rewards for a batch of molecules.

    Sub-rewards
    -----------
    1+2. Geometry (fused) — bond-length Z-score + bond-deviation proxy
    3.   ChemicalReward   — QED, SA, logP, MW (RDKit; zero for invalid mols)
    4.   ValidityReward   — binary sanitization success (always non-zero)
    """

    N_OBJECTIVES = 4

    def __init__(
        self,
        vocab: Dict[int, str],
        chem_targets: Optional[ChemTargets] = None,
    ):
        self.vocab = vocab
        self.chemical = ChemicalReward(chem_targets)
        self.validity = ValidityReward()

        self._ideal_len = _precompute_ideal_len_table(vocab)  #[V, V]
        self._ideal_len_dev = None

    def _ideal_on(self, device: torch.device) -> torch.Tensor:
        """Return the ideal-length table on device, caching the result."""
        if self._ideal_len_dev is None or self._ideal_len_dev.device != device:
            self._ideal_len_dev = self._ideal_len.to(device)
        return self._ideal_len_dev

    def __call__(
        self,
        coords: torch.Tensor,  #[B, N, 3]
        atom_types: torch.Tensor,  #[B, N, A]
        weighter: AdaptiveWeighter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_reward  : [B]
        sub_rewards   : [B, 4]  columns: [geometric, bond_dev, chemical, validity]
        """
        device = coords.device
        atom_idx = atom_types.argmax(-1).long()  #[B, N]

        r_geom, r_bond_dev = _score_geometry_batch(
            coords, atom_idx, self._ideal_on(device)
        )  #[B], [B]

        mols: List[Optional[Chem.Mol]] = [
            build_rdkit_mol(coords[b], decode_atom_types(atom_types[b], self.vocab))
            for b in range(coords.shape[0])
        ]
        r_chem = self.chemical(mols).to(device)
        r_valid = self.validity(mols).to(device)

        sub = torch.stack([r_geom, r_bond_dev, r_chem, r_valid], dim=-1)  #[B, 4]
        total = weighter(sub)
        return total, sub


_RDKIT_WORKERS: int = min(os.cpu_count() // 2 or 4, 16)


def _build_mol_task(args: Tuple) -> Optional[Chem.Mol]:
    """ThreadPoolExecutor worker: build one RDKit mol from (coords_cpu, elems)."""
    coords_cpu, elems = args
    return build_rdkit_mol(coords_cpu, elems)


def get_reward_batched(
    rewarder: MoleculeRewarder,
    weighter: AdaptiveWeighter,
    coords_g: torch.Tensor,  #[G, B, N, 3]
    atom_types_g: torch.Tensor,  #[G, B, N, A]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Score molecules and return [G, B] rewards.

    Returns
    -------
    rewards_g    : [G, B]
    sub_rewards_g: [G, B, 4]
    """
    G, B, N, _ = coords_g.shape
    M = G * B
    device = coords_g.device

    coords_flat = coords_g.reshape(M, N, 3)
    atom_types_flat = atom_types_g.reshape(M, N, -1)
    atom_idx_flat = atom_types_flat.argmax(-1).long()  #[M, N]

    r_geom, r_bond_dev = _score_geometry_batch(
        coords_flat, atom_idx_flat, rewarder._ideal_on(device)
    )  #[M], [M]

    coords_cpu = coords_flat.cpu()
    elems_list = [
        decode_atom_types(atom_types_flat[m], rewarder.vocab) for m in range(M)
    ]
    args_iter = ((coords_cpu[m], elems_list[m]) for m in range(M))
    with ThreadPoolExecutor(max_workers=_RDKIT_WORKERS) as pool:
        mols: List[Optional[Chem.Mol]] = list(pool.map(_build_mol_task, args_iter))

    r_chem = rewarder.chemical(mols).to(device)  #[M]
    r_valid = rewarder.validity(mols).to(device)  #[M]

    sub = torch.stack([r_geom, r_bond_dev, r_chem, r_valid], dim=-1)  #[M, 4]
    total = weighter(sub)  #[M]

    return total.reshape(G, B), sub.reshape(G, B, 4)


def compute_diversity(x0_pred: torch.Tensor) -> torch.Tensor:
    """
    For each batch position b, score how structurally distinct trajectory g's
    molecule is from the other G-1 trajectories at the same position.

    Uses pairwise cosine similarity in flattened coordinate space. A score of 1.0 means this
    trajectory's predicted molecule is completely orthogonal to all others;
    0.0 means it is identical to every other trajectory.

    Parameters
    ----------
    x0_pred : [G, B, N, D]  — x̂₀_pred at one of the three checkpoints

    Returns
    -------
    diversity : [G, B]  in [0, 1]
    """
    G, B, N, D = x0_pred.shape
    x_flat = x0_pred.reshape(G, B, N * D)  #[G, B, ND]
    x_norm = F.normalize(x_flat.float(), dim=-1)  #[G, B, ND]

    sim = torch.einsum('gbd,hbd->ghb', x_norm, x_norm)  #[G, G, B]

    mean_sim = (sim.sum(dim=1) - 1.0) / max(G - 1, 1)  #[G, B]
    return (1.0 - mean_sim).clamp(0.0, 1.0)  #[G, B]


class NoveltyBuffer:
    """
    Rolling FIFO buffer of recently generated SMILES strings.

    Score of 1.0 = molecule has not been generated in the last capacity unique
    molecules.  Score of 0.0 = molecule was generated recently (or is invalid).

    Parameters
    ----------
    capacity : int
        Number of unique SMILES to retain.  Default 5000
    """

    def __init__(self, capacity: int = 5_000):
        self._capacity = capacity
        self._buf: deque = deque()
        self._set: set = set()

    def __len__(self) -> int:
        return len(self._buf)

    def score_and_update(
        self,
        mols: List[Optional[Chem.Mol]],
    ) -> List[float]:
        """
        Score each molecule (1.0 = new, 0.0 = seen/invalid) then add new
        ones to the buffer.  Evicts oldest entries when capacity is exceeded.

        """
        scores: List[float] = []
        to_add: List[str] = []

        for mol in mols:
            smi = _safe_smiles(mol)
            if smi is None or smi in self._set:
                scores.append(0.0)
            else:
                scores.append(1.0)
                to_add.append(smi)
                self._set.add(smi)

        for smi in to_add:
            if len(self._buf) >= self._capacity:
                evicted = self._buf.popleft()
                self._set.discard(evicted)
            self._buf.append(smi)

        return scores


def _safe_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Return canonical SMILES or None on any failure."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi if smi else None
    except Exception:
        return None


def compute_novelty_batched(
    coords_g: torch.Tensor,  #[G, B, N, 3]
    atom_types_g: torch.Tensor,  #[G, B, N, A]
    vocab: Dict[int, str],
    buffer: NoveltyBuffer,
) -> torch.Tensor:  #[G, B]  in {0, 1}
    """
    Build RDKit mols for all molecules, score each against the
    novelty buffer, update the buffer with new molecules, and return a [G, B]
    float tensor of novelty scores (1.0 = new, 0.0 = seen/invalid).

    """
    G, B, N, _ = coords_g.shape
    M = G * B

    coords_flat = coords_g.reshape(M, N, 3).cpu()
    types_flat = atom_types_g.reshape(M, N, -1)
    elems_list = [decode_atom_types(types_flat[m], vocab) for m in range(M)]

    with ThreadPoolExecutor(max_workers=_RDKIT_WORKERS) as pool:
        mols: List[Optional[Chem.Mol]] = list(
            pool.map(_build_mol_task, ((coords_flat[m], elems_list[m]) for m in range(M)))
        )

    scores = buffer.score_and_update(mols)
    return torch.tensor(scores, dtype=torch.float,
                        device=coords_g.device).reshape(G, B)

def reward_log_dict(
    sub_rewards_g: torch.Tensor,  #[G, B, 4]
    weighter: AdaptiveWeighter,
) -> Dict[str, torch.Tensor]:
    """
    Build a flat dict suitable for passing to self.log_dict().
    """
    names = ['geometric', 'bond_dev', 'chemical', 'validity']
    logs = {}

    for i, name in enumerate(names):
        logs[f'reward_sub/{name}'] = sub_rewards_g[..., i].mean()

    eff_w = weighter.effective_weights  #[4]
    for i, name in enumerate(names):
        logs[f'reward_weight/{name}'] = eff_w[i]
        logs[f'reward_sigma/{name}'] = torch.exp(0.5 * weighter.log_sigma_sq[i])

    return logs