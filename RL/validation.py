"""
validation.py

Validation loop for SDPO molecule generation.

Metrics computed
----------------
validity        fraction of generated molecules that RDKit can sanitize
uniqueness      fraction of valid molecules that are structurally unique
novelty         fraction of unique molecules not in the training set
diversity       1 - mean pairwise Tanimoto (Morgan r=2) over unique mols
qed             mean QED over unique valid molecules
sa_score        mean SA score (1=easy, 10=hard) over unique valid molecules
mol_weight      mean molecular weight over unique valid molecules

Early-stopping metric
---------------------
stopping_score = validity × uniqueness × (0.50·qed + 0.30·sa_norm + 0.20·novelty)

  - validity × uniqueness acts as a hard gate: if either collapses to 0
    (mode collapse or structural invalidity), the score goes to 0 immediately.
  - The weighted term rewards chemical quality among molecules that are both
    valid and unique. Weights reflect your priority ranking:
        QED      0.50  (drug-likeness)
        SA_norm  0.30  (synthesizability, inverted+normalised to [0,1])
        Novelty  0.20  (not in training set)

Lightning early stopping config
--------------------------------
    EarlyStopping(
        monitor   = 'val/stopping_score',
        mode      = 'max',
        patience  = 20,        # validation epochs, not steps
        min_delta = 0.001,
    )

Bond inference strategy
-----------------------
`build_mol_from_coords` tries RDKit DetermineBonds first (no subprocess, fast).
If RDKit fails or produces an unsanitizable molecule, it falls back to an
OpenBabel subprocess.  The first approach that yields a sanitizable mol wins.
`build_rdkit_mol` from RL.reward is reused directly — no duplication.
"""

from __future__ import annotations

import os
import math
import time
import tempfile
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from RL.SDPO import pipeline_with_logprob
from RL.reward import build_rdkit_mol, decode_atom_types   # reuse, no duplication

try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
    _HAS_SA = True
except ImportError:
    try:
        import sascorer as _sascorer
        _HAS_SA = True
    except ImportError:
        _HAS_SA = False

try:
    from openbabel import openbabel as _ob
    _HAS_OB = True
except ImportError:
    _HAS_OB = False

# Use 'spawn' to avoid CUDA + fork deadlocks in subprocess workers.
_MP_CTX = mp.get_context('spawn')


# ── Stopping score weights ────────────────────────────────────────────────────
# Matches priority ranking: Uniqueness > Validity > QED > SA > Novelty.
# Uniqueness and validity are the gate (multiplicative); the rest are weighted.
_W_QED     = 0.50
_W_SA      = 0.30
_W_NOVELTY = 0.20


# ── Bond inference ────────────────────────────────────────────────────────────

def _try_rdkit(coords: torch.Tensor, elems: List[str]) -> Optional[Chem.Mol]:
    """
    Attempt bond inference via RDKit DetermineBonds.
    Returns a sanitized Mol or None.
    Reuses build_rdkit_mol from reward.py — single implementation.
    """
    return build_rdkit_mol(coords, elems)


def _ob_worker(args) -> Optional[str]:
    """
    Subprocess worker: XYZ coords → OpenBabel bond inference → molblock string.
    Runs in isolation to contain any OpenBabel crashes or hangs.
    """
    elems, xyz = args
    if len(elems) < 2:
        return None

    tmp_xyz = tmp_sdf = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False, mode='w') as f:
            tmp_xyz = f.name
            f.write(f'{len(elems)}\n\n')
            for elem, (x, y, z) in zip(elems, xyz):
                f.write(f'{elem:<3} {x:12.6f} {y:12.6f} {z:12.6f}\n')

        tmp_sdf = tmp_xyz.replace('.xyz', '.sdf')

        conv = _ob.OBConversion()
        conv.SetInAndOutFormats('xyz', 'sdf')
        mol  = _ob.OBMol()
        conv.ReadFile(mol, tmp_xyz)
        conv.WriteFile(mol, tmp_sdf)

        supplier = Chem.SDMolSupplier(tmp_sdf, sanitize=False, removeHs=False)
        rdmol    = supplier[0] if supplier else None
        if rdmol is None:
            return None

        # Reconstruct a clean RWMol so we don't carry OB-specific properties
        rw = Chem.RWMol()
        for atom in rdmol.GetAtoms():
            rw.AddAtom(Chem.Atom(atom.GetSymbol()))
        rw.AddConformer(rdmol.GetConformer(0), assignId=True)
        for bond in rdmol.GetBonds():
            rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        Chem.SanitizeMol(rw)
        return Chem.MolToMolBlock(rw)

    except Exception:
        return None
    finally:
        for p in (tmp_xyz, tmp_sdf):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _try_openbabel(
    elems : List[str],
    xyz   : List[Tuple[float, float, float]],
    timeout: float,
) -> Optional[Chem.Mol]:
    """Run OpenBabel bond inference in a subprocess with a hard timeout."""
    if not _HAS_OB:
        return None
    with _MP_CTX.Pool(processes=1) as pool:
        fut = pool.apply_async(_ob_worker, ((elems, xyz),))
        try:
            molblock = fut.get(timeout=timeout)
            if molblock is None:
                return None
            return Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        except (mp.TimeoutError, Exception):
            pool.terminate()
            pool.join()
            return None


def build_mol_from_coords(
    coords     : torch.Tensor,   # [N, 3]
    atom_types : torch.Tensor,   # [N, A]  one-hot / logit
    vocab      : Dict[int, str],
    ob_timeout : float = 3.0,
) -> Optional[Chem.Mol]:
    """
    Build an RDKit Mol from 3D coordinates and atom-type tensor.

    Strategy: RDKit DetermineBonds first (fast, no subprocess).
    If RDKit fails or produces an unsanitizable molecule, fall back to
    OpenBabel (more permissive valence rules, but heavier).
    Returns the first approach that yields a sanitizable mol, or None.
    """
    elems = decode_atom_types(atom_types, vocab)

    mol = _try_rdkit(coords, elems)
    if mol is not None:
        return mol

    # RDKit failed — try OpenBabel
    xyz = coords.cpu().float().tolist()
    return _try_openbabel(elems, xyz, ob_timeout)


def _build_mol_worker(args) -> Optional[str]:
    """
    Top-level pool worker for batch mol building.
    Returns MolBlock string (serialisable across processes) or None.
    Tries RDKit first, OpenBabel second, exactly as build_mol_from_coords does.
    """
    elems, xyz_list = args
    if not elems or len(elems) < 2:
        return None

    # ── RDKit attempt ─────────────────────────────────────────────────────────
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        xyz = torch.tensor(xyz_list, dtype=torch.float)
        rw   = Chem.RWMol()
        conf = Chem.Conformer(len(elems))
        for i, elem in enumerate(elems):
            idx = rw.AddAtom(Chem.Atom(elem))
            conf.SetAtomPosition(idx, xyz[i].tolist())
        rw.AddConformer(conf, assignId=True)
        AllChem.DetermineBonds(rw, charge=0)
        Chem.SanitizeMol(rw)
        return Chem.MolToMolBlock(rw)
    except Exception:
        pass

    # ── OpenBabel fallback ────────────────────────────────────────────────────
    return _ob_worker((elems, xyz_list))


def build_mols_batch(
    coords_list    : List[torch.Tensor],   # list of [N, 3]
    atom_types_list: List[torch.Tensor],   # list of [N, A]
    vocab          : Dict[int, str],
    n_workers      : int   = 0,
    wall_timeout   : float = 30.0,
) -> Tuple[List[Optional[Chem.Mol]], int]:
    """
    Build RDKit mols for a whole batch in parallel.

    Uses a single shared pool to avoid per-molecule spawn overhead.
    A single wall-clock deadline covers the entire batch — avoids the
    original per-future timeout that could accumulate to n_mols × timeout
    in the worst case.

    Returns (mols, n_timeouts).
    """
    if not coords_list:
        return [], 0

    n_workers = n_workers or min(mp.cpu_count()//2, len(coords_list))

    args_list = [
        (decode_atom_types(at, vocab), coords.cpu().float().tolist())
        for coords, at in zip(coords_list, atom_types_list)
    ]

    results   : List[Optional[Chem.Mol]] = [None] * len(args_list)
    n_timeouts = 0
    deadline   = time.monotonic() + wall_timeout

    with _MP_CTX.Pool(processes=n_workers) as pool:
        futures = [pool.apply_async(_build_mol_worker, (a,)) for a in args_list]

        for i, fut in enumerate(futures):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                n_timeouts += len(futures) - i
                break
            try:
                molblock = fut.get(timeout=remaining)
                if molblock is not None:
                    results[i] = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
                else:
                    results[i] = None
            except mp.TimeoutError:
                n_timeouts += 1
            except Exception:
                pass

        pool.terminate()   # cancel any still-running workers
        pool.join()

    return results, n_timeouts


# ── SMILES utilities ──────────────────────────────────────────────────────────

def _to_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi or None
    except Exception:
        return None


# ── Per-molecule property computation ────────────────────────────────────────

def _safe_qed(mol: Chem.Mol) -> float:
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0


def _safe_sa(mol: Chem.Mol) -> Optional[float]:
    if not _HAS_SA:
        return None
    try:
        return _sascorer.calculateScore(mol)
    except Exception:
        return None


def _sa_to_norm(sa: float) -> float:
    """Map SA score [1, 10] → [0, 1], inverted (1 = trivial to synthesize)."""
    return max(0.0, min(1.0, 1.0 - (sa - 1.0) / 9.0))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    mols         : List[Optional[Chem.Mol]],
    train_smiles : set,
) -> Tuple[Dict[str, float], List[str], List[Chem.Mol]]:
    """
    Compute all validation metrics from a list of RDKit mols.

    Returns (metrics_dict, unique_smiles, unique_mols).
    unique_* are deduplicated valid molecules, useful for SDF export.
    """
    n_total = len(mols)

    # ── Validity ──────────────────────────────────────────────────────────────
    smiles_map = [(m, _to_smiles(m)) for m in mols]
    valid_pairs = [(m, s) for m, s in smiles_map if s is not None]
    n_valid     = len(valid_pairs)
    validity    = n_valid / n_total if n_total > 0 else 0.0

    # ── Uniqueness ────────────────────────────────────────────────────────────
    seen: Dict[str, Chem.Mol] = {}
    for m, s in valid_pairs:
        if s not in seen:
            seen[s] = m
    unique_smiles = list(seen.keys())
    unique_mols   = list(seen.values())
    n_unique      = len(unique_smiles)
    uniqueness    = n_unique / n_valid if n_valid > 0 else 0.0

    # ── Novelty ───────────────────────────────────────────────────────────────
    n_novel = sum(1 for s in unique_smiles if s not in train_smiles)
    novelty = n_novel / n_unique if n_unique > 0 else 0.0

    # ── Diversity ─────────────────────────────────────────────────────────────
    diversity = _compute_diversity(unique_mols)

    # ── Chemical properties ───────────────────────────────────────────────────
    qed_scores = [_safe_qed(m)   for m in unique_mols]
    sa_scores  = [_safe_sa(m)    for m in unique_mols]
    mw_scores  = [Descriptors.MolWt(m) for m in unique_mols]

    mean_qed = float(np.mean(qed_scores)) if qed_scores else 0.0
    mean_sa  = float(np.mean([s for s in sa_scores if s is not None])) if any(
        s is not None for s in sa_scores
    ) else 5.0   # neutral fallback
    mean_mw  = float(np.mean(mw_scores))  if mw_scores  else 0.0

    sa_norm = _sa_to_norm(mean_sa)

    # ── Stopping score ────────────────────────────────────────────────────────
    # Gate:    validity × uniqueness  → 0 if either collapses
    # Weights: QED(0.50) + SA_norm(0.30) + novelty(0.20)
    gate    = validity * uniqueness
    quality = _W_QED * mean_qed + _W_SA * sa_norm + _W_NOVELTY * novelty
    stopping_score = gate * quality

    metrics = {
        # Production-level
        'validity'       : validity,
        'uniqueness'     : uniqueness,
        'novelty'        : novelty,
        'n_valid'        : n_valid,
        'n_unique'       : n_unique,
        # Chemical quality
        'qed'            : mean_qed,
        'sa_score'       : mean_sa,
        'sa_norm'        : sa_norm,
        'mol_weight'     : mean_mw,
        'diversity'      : diversity,
        # Composite
        'stopping_score' : stopping_score,
        # Diagnostic sub-scores
        'gate'           : gate,
        'quality'        : quality,
    }
    return metrics, unique_smiles, unique_mols


def _compute_diversity(mols: List[Chem.Mol], max_mols: int = 1000) -> float:
    """1 − mean pairwise Tanimoto (Morgan r=2) over unique valid mols."""
    if len(mols) < 2:
        return 0.0
    if len(mols) > max_mols:
        idx  = np.random.choice(len(mols), max_mols, replace=False)
        mols = [mols[i] for i in idx]

    fps     = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    sim_sum = 0.0
    count   = 0
    for i, fp in enumerate(fps):
        sims     = BulkTanimotoSimilarity(fp, fps[i + 1:])
        sim_sum += sum(sims)
        count   += len(sims)

    return 1.0 - (sim_sum / count) if count > 0 else 0.0


# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_molecules(
    model,
    scheduler,
    vocab        : Dict[int, str],
    device,
    n_samples    : int   = 256,
    n_atoms      : int   = 29,
    sample_steps : int   = 25,
    eta          : float = 1.0,
    batch_size   : int   = 32,
    coord_dim    : int   = 3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate n_samples molecules and return their coords and atom types.

    Uses x̂₀_pred_last — the model's final predicted clean molecule — rather
    than the raw last denoising state.  x̂₀_pred_last is already fully decoded
    and clean, which is what we want to evaluate during validation.

    atom_dim is inferred from vocab rather than hard-coded so it stays
    consistent with whatever vocabulary the model was built with.

    Returns
    -------
    coords_list    : list of n_samples tensors [N, 3]
    atom_types_list: list of n_samples tensors [N, A]
    """
    atom_dim = len(vocab)
    model.eval()

    all_coords     = []
    all_atom_types = []

    for start in range(0, n_samples, batch_size):
        B     = min(batch_size, n_samples - start)
        x     = torch.randn(B, n_atoms, 3,        device=device)
        types = torch.randn(B, n_atoms, atom_dim,  device=device)

        # x0_pred_last: [B, N, D]  — the new pipeline return
        _, _, _, _, _, _, x0_pred_last, _, _ = pipeline_with_logprob(
            model, x, types,
            scheduler           = scheduler,
            B                   = B,
            device              = device,
            num_inference_steps = sample_steps,
            eta                 = eta,
        )

        coords_b    = x0_pred_last[:, :, :coord_dim]         # [B, N, 3]
        atom_types_b = x0_pred_last[:, :, coord_dim:]        # [B, N, A]

        for b in range(B):
            all_coords.append(coords_b[b])
            all_atom_types.append(atom_types_b[b])

    return all_coords, all_atom_types


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model,
    scheduler,
    vocab        : Dict[int, str],
    train_smiles : set,
    device,
    n_samples    : int   = 512,
    sample_steps : int   = 25,
    eta          : float = 1.0,
    batch_size   : int   = 32,
    n_workers    : int   = 0,
    wall_timeout : float = 60.0,
) -> Tuple[Dict[str, float], List[Chem.Mol], List[str]]:
    """
    Full evaluation pass: sample → build mols → compute metrics.

    Returns (metrics, unique_mols, unique_smiles).
    unique_* are available for SDF export if needed.
    """
    coords_list, atom_types_list = sample_molecules(
        model, scheduler, vocab, device,
        n_samples    = n_samples,
        sample_steps = sample_steps,
        eta          = eta,
        batch_size   = batch_size,
    )

    mols, n_timeouts = build_mols_batch(
        coords_list, atom_types_list, vocab,
        n_workers    = n_workers,
        wall_timeout = wall_timeout,
    )

    if n_timeouts > 0:
        print(f'[evaluate] {n_timeouts}/{n_samples} molecules timed out in bond inference')

    metrics, unique_smiles, unique_mols = compute_metrics(mols, train_smiles)
    metrics['n_timeouts'] = n_timeouts

    return metrics, unique_mols, unique_smiles


# ── SDF export ────────────────────────────────────────────────────────────────

def write_sdf(
    mols         : List[Chem.Mol],
    smiles       : List[str],
    path         : str,
    train_smiles : Optional[set] = None,
    kekulize     : bool          = False,
) -> int:
    """
    Write annotated unique valid molecules to an SDF file.

    Each molecule is annotated with SMILES, QED, MolWeight, SA_Score (if
    available), and Novel (0/1 against training set if provided).

    Returns the number of molecules written.
    """
    n_written = 0
    with open(path, 'w') as fh:
        for mol, smi in zip(mols, smiles):
            if mol is None:
                continue
            try:
                rw = Chem.RWMol(mol)
                rw.SetProp('SMILES', smi)
                rw.SetDoubleProp('QED', _safe_qed(rw))
                rw.SetDoubleProp('MolWeight', Descriptors.MolWt(rw))

                sa = _safe_sa(rw)
                if sa is not None:
                    rw.SetDoubleProp('SA_Score', sa)

                if train_smiles is not None:
                    rw.SetIntProp('Novel', int(smi not in train_smiles))

                block = Chem.MolToMolBlock(rw, kekulize=kekulize)
                fh.write(block + '$$$$\n')
                n_written += 1
            except Exception:
                continue

    return n_written


# ── Lightning mixin ───────────────────────────────────────────────────────────

class ValidationMixin:
    """
    Mix into LightningTabascoPipe to add structured validation and early stopping.

        class LightningTabascoPipe(ValidationMixin, pl.LightningModule):
            ...

    Optional attributes on the host class
    --------------------------------------
    val_n_samples   int   number of molecules to generate per validation epoch (default 256)
    val_write_sdf   bool  write an SDF file every validation epoch (default False)
    val_sdf_dir     str   directory for SDF files (default 'logs/val_mols')
    train_smiles    set   training SMILES for novelty computation
    """

    def validation_step(self, batch, batch_idx):
        pass   # data is generated inside on_validation_epoch_end

    def on_validation_epoch_end(self):
        n_samples    = getattr(self, 'val_n_samples', 256)
        train_smiles = getattr(self, 'train_smiles',  set())
        write_sdf_   = getattr(self, 'val_write_sdf', False)
        sdf_dir      = getattr(self, 'val_sdf_dir',   'logs/val_mols')

        metrics, unique_mols, unique_smiles = evaluate(
            model        = self.model,
            scheduler    = self.scheduler,
            vocab        = self.vocab,
            train_smiles = train_smiles,
            device       = self.device,
            n_samples    = n_samples,
            sample_steps = self.args.sample_steps,
            eta          = self.eta,
            batch_size   = 32,
        )

        # Log all float metrics; highlight stopping_score in the progress bar
        for k, v in metrics.items():
            if isinstance(v, float):
                self.log(
                    f'val/{k}', v,
                    prog_bar  = (k == 'stopping_score'),
                    sync_dist = True,
                )

        # Optional SDF export — useful for qualitative inspection
        if write_sdf_ and unique_mols:
            os.makedirs(sdf_dir, exist_ok=True)
            path = os.path.join(
                sdf_dir, f'epoch{self.current_epoch:04d}_step{self.global_step}.sdf'
            )
            n = write_sdf(
                unique_mols, unique_smiles, path,
                train_smiles=train_smiles,
            )
            if self.args.debug:
                print(f'[val] wrote {n} molecules to {path}')

        if self.args.debug:
            g = metrics
            print(
                f'\n[val epoch {self.current_epoch}]'
                f'  validity={g["validity"]:.3f}'
                f'  unique={g["uniqueness"]:.3f}'
                f'  novelty={g["novelty"]:.3f}'
                f'  qed={g["qed"]:.3f}'
                f'  sa={g["sa_score"]:.2f}'
                f'  stopping={g["stopping_score"]:.4f}'
            )