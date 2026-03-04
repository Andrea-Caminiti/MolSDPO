"""
validation.py

Drop-in validation loop for SDPO molecule generation.
Computes validity, uniqueness, novelty, diversity, QED and SA score.
Integrates with Lightning via validation_step / on_validation_epoch_end,
and drives early stopping through the 'val/stopping_score' metric.

Early stopping config in trainer:
    from lightning.pytorch.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val/stopping_score',
        mode='max',
        patience=20,        # in validation epochs, not steps
        min_delta=0.001,
    )
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdDetermineBonds, Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

try:
    from rdkit.Contrib.SA_Score import sascorer
    _HAS_SA = True
except ImportError:
    try:
        import sascorer
        _HAS_SA = True
    except ImportError:
        _HAS_SA = False

from RL.SDPO import pipeline_with_logprob
import tempfile
import os
from openbabel import openbabel
from rdkit import Chem
import multiprocessing as mp

# Always use 'spawn' to avoid CUDA + fork deadlocks.
_MP_CTX = mp.get_context('spawn')


def _atomic_num_to_symbol(z: int) -> str:
    pt = Chem.GetPeriodicTable()
    return pt.GetElementSymbol(z)


def _write_xyz(path: str, zs: list, xyz) -> None:
    """Write a minimal XYZ file from atomic numbers and coordinates."""
    lines = [str(len(zs)), '']
    for z, (x, y, coord_z) in zip(zs, xyz):   # coord_z: third spatial coordinate
        lines.append(f'{_atomic_num_to_symbol(z):<3} {x:12.6f} {y:12.6f} {coord_z:12.6f}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _coords_to_mol_worker(args) -> Optional[str]:
    """
    Subprocess worker: XYZ -> OpenBabel bond inference -> molblock string.
    Returns MolBlock string or None.
    """
    zs, xyz = args

    mask = [z != 0 for z in zs]
    zs   = [z for z, m in zip(zs, mask) if m]
    xyz  = [c for c, m in zip(xyz, mask) if m]

    if len(zs) < 2:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            tmp_path = tmp.name

        _write_xyz(tmp_path, zs, xyz)

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats('xyz', 'sdf')
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_path)
        obConversion.WriteFile(ob_mol, tmp_path)

        tmp_mol = Chem.SDMolSupplier(tmp_path, sanitize=False, removeHs=False)[0]
        os.unlink(tmp_path)

        if tmp_mol is None:
            return None

        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        mol.AddConformer(tmp_mol.GetConformer(0))
        for bond in tmp_mol.GetBonds():
            mol.AddBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )

        Chem.SanitizeMol(mol)
        return Chem.MolToMolBlock(mol)

    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def coords_to_mol(
    atomic_nums: torch.Tensor,
    coords     : torch.Tensor,
    timeout    : float = 5.0,
) -> Optional[Chem.Mol]:
    """
    Build an RDKit Mol from atomic number and coordinate tensors using
    OpenBabel for bond inference.  Runs in a subprocess with a hard timeout.
    """
    zs  = atomic_nums.cpu().tolist()
    xyz = coords.cpu().float().numpy().tolist()

    with _MP_CTX.Pool(processes=1) as pool:
        fut = pool.apply_async(_coords_to_mol_worker, ((zs, xyz),))
        try:
            molblock = fut.get(timeout=timeout)
            if molblock is None:
                return None
            return Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        except mp.TimeoutError:
            pool.terminate()
            pool.join()
            return None
        except Exception:
            return None


def _coords_to_mol_batch(
    args_list: list,
    timeout  : float = 2.0,
    n_workers: int   = 0,
) -> Tuple[List[Optional[Chem.Mol]], int]:
    """
    Convert a whole batch of molecules using a single shared pool.
    Avoids the overhead of spawning a new pool per molecule.
    """
    if not args_list:
        return [], 0

    if n_workers <= 0:
        n_workers = min(mp.cpu_count(), len(args_list))

    mols: List[Optional[Chem.Mol]] = []
    n_timeouts = 0

    with _MP_CTX.Pool(processes=n_workers) as pool:
        futures = [pool.apply_async(_coords_to_mol_worker, (args,)) for args in args_list]

        for args, fut in zip(args_list, futures):
            try:
                molblock = fut.get(timeout=timeout)
                mols.append(
                    None if molblock is None
                    else Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
                )
            except mp.TimeoutError:
                mols.append(None)
                if args[0]:
                    n_timeouts += 1
            except Exception:
                mols.append(None)

    return mols, n_timeouts


def _mol_to_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Convert a mol to canonical SMILES, returning None on failure."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi if smi else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_validity_uniqueness_novelty(
    mols        : List[Optional[Chem.Mol]],
    train_smiles: set,
) -> Tuple[float, float, float, List[str], List[Chem.Mol]]:
    """
    Returns (validity, uniqueness, novelty, valid_smiles_list, unique_mols).

      validity   = n_valid  / n_total
      uniqueness = n_unique / n_valid
      novelty    = n_novel  / n_unique
    """
    n_total = len(mols)
    smiles_results = [_mol_to_smiles(mol) for mol in mols]

    valid_pairs: List[Tuple[str, Chem.Mol]] = [
        (smi, mol)
        for smi, mol in zip(smiles_results, mols)
        if smi is not None
    ]
    n_valid = len(valid_pairs)

    seen: Dict[str, Chem.Mol] = {}
    for smi, mol in valid_pairs:
        if smi not in seen:
            seen[smi] = mol

    unique_smiles = list(seen.keys())
    unique_mols   = list(seen.values())
    n_unique      = len(unique_smiles)

    validity   = n_valid  / n_total if n_total  > 0 else 0.0
    uniqueness = n_unique / n_valid if n_valid  > 0 else 0.0
    n_novel    = sum(1 for s in unique_smiles if s not in train_smiles)
    novelty    = n_novel / n_unique if n_unique > 0 else 0.0

    return validity, uniqueness, novelty, unique_smiles, unique_mols


def compute_diversity_from_mols(mols: List[Chem.Mol], max_mols: int = 1000) -> float:
    """1 - mean pairwise Tanimoto (Morgan r=2) over unique valid mols."""
    if len(mols) < 2:
        return 0.0
    if len(mols) > max_mols:
        indices = np.random.choice(len(mols), max_mols, replace=False)
        mols    = [mols[i] for i in indices]

    fps     = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    sim_sum = 0.0
    count   = 0
    for i in range(len(fps)):
        sims     = BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_sum += sum(sims)
        count   += len(sims)

    return 1.0 - (sim_sum / count) if count > 0 else 0.0


def compute_drug_properties(mols: List[Chem.Mol]) -> Dict[str, float]:
    """Returns mean QED, SA score (if available), and mean MW."""
    mols = [m for m in mols if m is not None]
    if not mols:
        return {'qed': 0.0, 'sa': 0.0, 'mw': 0.0}

    result = {
        'qed': float(np.mean([QED.qed(m)           for m in mols])),
        'mw':  float(np.mean([Descriptors.MolWt(m) for m in mols])),
        'sa':  0.0,
    }
    if _HAS_SA:
        try:
            result['sa'] = float(np.mean([sascorer.calculateScore(m) for m in mols]))
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_molecules(
    model,
    scheduler,
    vocab,            # enc2atom tensor  [A] → atomic numbers
    device,
    n_samples   : int   = 256,
    n_atoms     : int   = 29,
    sample_steps: int   = 25,
    eta         : float = 1.0,
    batch_size  : int   = 32,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate n_samples molecules in batches.

    atom_dim is inferred from vocab.shape[0] so it stays consistent with
    whatever vocabulary the model was built with.

    Returns list of (coords [N,3], atom_types_onehot [N,A]) tuples.
    """
    atom_dim = vocab.shape[0]   # inferred from vocab, not hard-coded
    model.eval()
    all_last = []

    for start in range(0, n_samples, batch_size):
        B     = min(batch_size, n_samples - start)
        x     = torch.randn(B, n_atoms, 3,        device=device)
        types = torch.randn(B, n_atoms, atom_dim,  device=device)

        mols, *_ = pipeline_with_logprob(
            model, x, types,
            scheduler=scheduler, B=B,
            device=device,
            num_inference_steps=sample_steps,
            eta=eta,
        )
        last_coords = mols[2][:, :, :3]   # [B, N, 3]
        last_types  = mols[2][:, :, 3:]   # [B, N, A]
        for b in range(B):
            all_last.append((last_coords[b], last_types[b]))

    return all_last


# ---------------------------------------------------------------------------
# SDF writer
# ---------------------------------------------------------------------------

def _mol_to_sdf_block(args: Tuple) -> Optional[str]:
    """Annotate one molecule and render it to an SDF block string."""
    mol, smi, train_smiles_set, kekulize = args
    try:
        rw = Chem.RWMol(mol)
        rw.SetProp('SMILES',          smi)
        rw.SetDoubleProp('QED',       QED.qed(rw))
        rw.SetDoubleProp('MolWeight', Descriptors.MolWt(rw))

        if _HAS_SA:
            try:
                rw.SetDoubleProp('SA_Score', sascorer.calculateScore(rw))
            except Exception:
                pass

        if train_smiles_set is not None:
            rw.SetIntProp('Novel', int(smi not in train_smiles_set))

        block = Chem.MolToMolBlock(rw, kekulize=kekulize)
        return block + '$$$$\n'
    except Exception:
        return None


def write_unique_mols_sdf(
    mols         : List[Optional[Chem.Mol]],
    path         : str,
    train_smiles : Optional[set]       = None,
    kekulize     : bool                = False,
    unique_smiles: Optional[List[str]] = None,
    unique_mols  : Optional[List[Chem.Mol]] = None,
) -> int:
    """
    Deduplicate a list of RDKit Mols (by canonical SMILES) and write to SDF.

    If unique_smiles + unique_mols are provided (e.g. from evaluate()),
    deduplication is skipped entirely.

    Returns the number of unique valid molecules written.
    """
    if unique_smiles is not None and unique_mols is not None:
        u_smiles, u_mols = unique_smiles, unique_mols
    else:
        smi_results = [_mol_to_smiles(m) for m in mols]
        seen: Dict[str, Chem.Mol] = {}
        for smi, mol in zip(smi_results, mols):
            if smi is not None and mol is not None and smi not in seen:
                seen[smi] = mol
        u_smiles = list(seen.keys())
        u_mols   = list(seen.values())

    if not u_mols:
        return 0

    blocks = [
        _mol_to_sdf_block((mol, smi, train_smiles, kekulize))
        for mol, smi in zip(u_mols, u_smiles)
    ]
    valid_blocks = [b for b in blocks if b is not None]
    with open(path, 'w') as fh:
        fh.write(''.join(valid_blocks))

    return len(valid_blocks)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    scheduler,
    vocab,
    train_smiles : set,
    device,
    n_samples    : int   = 512,
    sample_steps : int   = 25,
    eta          : float = 1.0,
    batch_size   : int   = 32,
    mol_timeout  : float = 2.0,
) -> Dict[str, float]:
    """
    Full evaluation pass.

    stopping_score = validity × uniqueness
      - Drops on model collapse (validity ↓) or mode collapse (uniqueness ↓)
      - Independent of training reward (avoids Goodhart's law)
    """
    samples = sample_molecules(
        model, scheduler, vocab, device,
        n_samples    = n_samples,
        sample_steps = sample_steps,
        eta          = eta,
        batch_size   = batch_size,
    )

    worker_args = [
        (
            vocab[atom_types_oh.argmax(-1)].cpu().tolist(),
            coords.cpu().float().numpy().tolist(),
        )
        for coords, atom_types_oh in samples
    ]

    mols, n_timeouts = _coords_to_mol_batch(worker_args, timeout=mol_timeout)

    if n_timeouts > 0:
        print(f'[evaluate] Warning: {n_timeouts}/{n_samples} mols timed out in bond inference')

    validity, uniqueness, novelty, unique_smiles, unique_mols = \
        compute_validity_uniqueness_novelty(mols, train_smiles)

    diversity = compute_diversity_from_mols(unique_mols)
    drug      = compute_drug_properties(unique_mols)

    return {
        'validity':       validity,
        'uniqueness':     uniqueness,
        'novelty':        novelty,
        'diversity':      diversity,
        'stopping_score': validity * uniqueness,
        'qed':            drug['qed'],
        'sa_score':       drug['sa'],
        'mol_weight':     drug['mw'],
        'n_valid':        int(validity   * n_samples),
        'n_unique':       int(uniqueness * validity * n_samples),
        'n_timeouts':     n_timeouts,
    }, unique_mols, unique_smiles


# ---------------------------------------------------------------------------
# Lightning integration
# ---------------------------------------------------------------------------

class ValidationMixin:
    """
    Mix into LightningTabascoPipe to add evaluation and early stopping.

    Usage:
        class LightningTabascoPipe(ValidationMixin, pl.LightningModule):
            ...
    """

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        n            = getattr(self, 'val_n_samples', 256)
        train_smiles = getattr(self, 'train_smiles', set())

        metrics, *_ = evaluate(
            model        = self.model,
            scheduler    = self.scheduler,
            vocab        = self.vocab,
            train_smiles = train_smiles,
            device       = self.device,
            n_samples    = n,
            sample_steps = self.args.sample_steps,
            eta          = self.eta,
            batch_size   = 32,
        )

        for k, v in metrics.items():
            if isinstance(v, float):
                self.log(f'val/{k}', v, prog_bar=(k == 'stopping_score'), sync_dist=True)