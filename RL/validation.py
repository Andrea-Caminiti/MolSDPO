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
# SA score — needs RDKit contrib. Gracefully degrade if unavailable.
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

# FIX 1: Always use 'spawn' to avoid CUDA + fork deadlocks.
# After PyTorch initialises CUDA, forking a process leaves CUDA's internal
# mutexes in a locked state in the child, causing an immediate hang.
# 'spawn' starts a fresh interpreter with no inherited CUDA state.
_MP_CTX = mp.get_context('spawn')


def _atomic_num_to_symbol(z: int) -> str:
    pt = Chem.GetPeriodicTable()
    return pt.GetElementSymbol(z)

def _write_xyz(path: str, zs: list, xyz) -> None:
    """Write a minimal XYZ file from atomic numbers and coordinates."""
    lines = [str(len(zs)), '']  # atom count + blank comment line
    for z, (x, y, w) in zip(zs, xyz):
        lines.append(f'{_atomic_num_to_symbol(z):<3} {x:12.6f} {y:12.6f} {w:12.6f}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _coords_to_mol_worker(args) -> Optional[str]:
    """
    Subprocess worker: XYZ -> OpenBabel bond inference -> molblock string.
    Returns MolBlock string or None. Molblock is used so coordinates survive
    the process boundary (SMILES would discard the conformer).
    """
    zs, xyz = args

    mask = [z != 0 for z in zs]
    zs  = [z for z, m in zip(zs, mask) if m]
    xyz = [c for c, m in zip(xyz, mask) if m]

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

        # Rebuild mol to strip radicals (same workaround as original)
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        mol.AddConformer(tmp_mol.GetConformer(0))
        for bond in tmp_mol.GetBonds():
            mol.AddBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType()
            )

        Chem.SanitizeMol(mol)
        return Chem.MolToMolBlock(mol)  # preserves 3D coords across process boundary

    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def coords_to_mol(atomic_nums: torch.Tensor, coords: torch.Tensor, timeout: float = 5.0) -> Optional[Chem.Mol]:
    """
    Build an RDKit Mol from atomic number and coordinate tensors using
    OpenBabel for bond inference. Runs in a subprocess with a hard timeout
    so hung OpenBabel calls are forcibly killed rather than blocking forever.

    Args:
        atomic_nums : [N] integer tensor of atomic numbers (0 = padding)
        coords      : [N, 3] float tensor in Angstroms
        timeout     : seconds before the subprocess is hard-killed

    Returns:
        RDKit Mol with 3D conformer, or None on failure/timeout.
    """
    zs  = atomic_nums.cpu().tolist()
    xyz = coords.cpu().float().numpy().tolist()

    # FIX 1 (applied): Use _MP_CTX (spawn) instead of mp.Pool (fork).
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
    timeout: float = 2.0,
    n_workers: int = 0,
) -> Tuple[List[Optional[Chem.Mol]], int]:
    """
    FIX 2: Convert a whole batch of molecules using a single shared pool.

    Previously, evaluate() called coords_to_mol() (which spawns its own pool)
    in a serial loop — creating and destroying 512+ processes one at a time,
    each paying the full spawn overhead.  This function submits all tasks to
    one pool up front, collects results with per-task timeouts, and terminates
    the pool only once at the end.

    Args:
        args_list : list of (zs, xyz) tuples (plain Python lists, picklable)
        timeout   : per-molecule wall-clock timeout in seconds
        n_workers : pool size; 0 → min(cpu_count, len(args_list))

    Returns:
        (mols, n_timeouts) where mols[i] is the RDKit Mol or None
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
                if molblock is None:
                    mols.append(None)
                else:
                    mols.append(Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False))
            except mp.TimeoutError:
                mols.append(None)
                if args[0]:  # non-empty atom list → genuine timeout, not padding
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
# Individual metrics
# ---------------------------------------------------------------------------

def compute_validity_uniqueness_novelty(
    mols: List[Optional[Chem.Mol]],
    train_smiles: set,
) -> Tuple[float, float, float, List[str], List[Chem.Mol]]:
    """
    Returns (validity, uniqueness, novelty, valid_smiles_list, unique_mols).

    Definitions:
      validity   = n_valid  / n_total
      uniqueness = n_unique / n_valid
      novelty    = n_novel  / n_unique  (unique valid not in train set)

    unique_mols is returned so downstream callers can skip re-parsing SMILES.
    """
    n_total = len(mols)

    # FIX 3: Run SMILES conversion in-process instead of via ProcessPoolExecutor.
    #
    # The original code passed RDKit Mol objects to a ProcessPoolExecutor.
    # Mol objects are pickled across the process boundary on every call, which
    # is expensive and can silently hang when the pickled C++ state is large or
    # contains non-picklable internals (platform/build dependent).
    # MolToSmiles is fast enough that the serialisation overhead dominates anyway.
    smiles_results = [_mol_to_smiles(mol) for mol in mols]

    # Keep valid (smiles, mol) pairs together to avoid re-parsing later
    valid_pairs: List[Tuple[str, Chem.Mol]] = [
        (smi, mol)
        for smi, mol in zip(smiles_results, mols)
        if smi is not None
    ]

    n_valid = len(valid_pairs)

    # Deduplicate while preserving order; keep the mol for each unique SMILES
    seen: Dict[str, Chem.Mol] = {}
    for smi, mol in valid_pairs:
        if smi not in seen:
            seen[smi] = mol

    unique_smiles = list(seen.keys())
    unique_mols   = list(seen.values())
    n_unique      = len(unique_smiles)

    validity   = n_valid  / n_total  if n_total  > 0 else 0.0
    uniqueness = n_unique / n_valid  if n_valid  > 0 else 0.0

    # train_smiles is already a set, so `not in` is O(1)
    n_novel = sum(1 for s in unique_smiles if s not in train_smiles)
    novelty = n_novel / n_unique if n_unique > 0 else 0.0

    return validity, uniqueness, novelty, unique_smiles, unique_mols


def compute_diversity(smiles_list: List[str], max_mols: int = 1000) -> float:
    """
    Internal diversity = 1 - mean pairwise Tanimoto (Morgan r=2 bit fingerprints).

    Uses bit vectors (faster Tanimoto) and caps at max_mols to keep O(n²) tractable.
    """
    n = len(smiles_list)
    if n < 2:
        return 0.0

    if n > max_mols:
        smiles_list = list(np.random.choice(smiles_list, max_mols, replace=False))

    # FIX: walrus operator bug — filter and assign must be separated
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0

    # Bit vectors are faster for Tanimoto than sparse count fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

    sim_sum = 0.0
    count   = 0
    for i in range(len(fps)):
        sims     = BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_sum += sum(sims)
        count   += len(sims)

    return 1.0 - (sim_sum / count) if count > 0 else 0.0


def compute_diversity_from_mols(mols: List[Chem.Mol], max_mols: int = 1000) -> float:
    """
    Same as compute_diversity but accepts pre-built Mol objects, avoiding re-parsing.
    """
    if len(mols) < 2:
        return 0.0

    if len(mols) > max_mols:
        indices = np.random.choice(len(mols), max_mols, replace=False)
        mols = [mols[i] for i in indices]

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

    sim_sum = 0.0
    count   = 0
    for i in range(len(fps)):
        sims     = BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_sum += sum(sims)
        count   += len(sims)

    return 1.0 - (sim_sum / count) if count > 0 else 0.0


def compute_drug_properties(mols: List[Chem.Mol]) -> Dict[str, float]:
    """
    Returns mean QED, mean SA score (if available), and mean MW.
    Accepts pre-built Mol objects to avoid redundant SMILES re-parsing.
    """
    mols = [m for m in mols if m is not None]
    if not mols:
        return {'qed': 0.0, 'sa': 0.0, 'mw': 0.0}

    qeds = [QED.qed(m)          for m in mols]
    mws  = [Descriptors.MolWt(m) for m in mols]

    result = {
        'qed': float(np.mean(qeds)),
        'mw':  float(np.mean(mws)),
        'sa':  0.0,
    }

    if _HAS_SA:
        try:
            sas = [sascorer.calculateScore(m) for m in mols]
            result['sa'] = float(np.mean(sas))
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
    n_samples:    int = 256,
    n_atoms:      int = 29,
    atom_dim:     int = 6,
    sample_steps: int = 25,
    eta:          float = 1.0,
    batch_size:   int = 32,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate n_samples molecules in batches.
    Returns list of (coords [N,3], atom_types_onehot [N,A]) tuples.
    """

    model.eval()
    all_last = []

    for start in range(0, n_samples, batch_size):
        B     = min(batch_size, n_samples - start)
        x     = torch.randn(B, n_atoms, 3,       device=device)
        types = torch.randn(B, n_atoms, atom_dim, device=device)

        mols, _, _, _, _, _, _ = pipeline_with_logprob(
            model, x, types,
            scheduler=scheduler, B=B,
            device=device,
            num_inference_steps=sample_steps,
            eta=eta,
        )
        last_coords = mols[2][0]   # [B, N, 3]
        last_types  = mols[2][1]   # [B, N, A]

        for b in range(B):
            all_last.append((last_coords[b], last_types[b]))

    return all_last


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def _mol_to_sdf_block(args: Tuple) -> Optional[str]:
    """
    Worker: annotate one molecule and render it to an SDF block string.

    Runs in a Process pool — each mol is independent, so no locking needed.
    We work on a fresh RWMol copy so we never mutate the caller's objects.

    Returns the SDF block string, or None if anything fails.
    """
    mol, smi, train_smiles_set, kekulize = args
    try:
        rw = Chem.RWMol(mol)                          # cheap copy; don't mutate caller's mol
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

        # MolToMolBlock is the same C++ path SDWriter uses internally,
        # but returning a string lets us batch all I/O into one write().
        block = Chem.MolToMolBlock(rw, kekulize=kekulize)
        return block + '$$$$\n'           # SDF record terminator
    except Exception:
        return None

    
def write_unique_mols_sdf(
    mols: List[Optional[Chem.Mol]],
    path: str,
    train_smiles: Optional[set] = None,
    kekulize: bool = False,
    # ── Fast-path: pass these when calling from evaluate() so dedup is skipped ──
    unique_smiles: Optional[List[str]]      = None,
    unique_mols:   Optional[List[Chem.Mol]] = None,
) -> int:
    """
    Deduplicate a list of RDKit Mols (by canonical SMILES) and write them to
    an SDF file.  Annotates each record with SMILES, QED, MolWeight, SA_Score,
    and (optionally) a Novel flag.

    Performance notes
    -----------------
    * If you already have ``unique_smiles`` + ``unique_mols`` from a previous
      call to ``compute_validity_uniqueness_novelty`` or ``evaluate()``, pass
      them in — deduplication (the most expensive serial step) is skipped
      entirely.
    * All SDF blocks are accumulated in memory and flushed to disk in a single
      ``write()`` call, minimising syscall overhead.

    Args:
        mols          : raw generated mols (may contain None / duplicates).
                        Ignored when ``unique_smiles`` + ``unique_mols`` are
                        both provided.
        path          : output file path, e.g. ``'generated.sdf'``
        train_smiles  : if provided, tags each record with ``Novel=1/0``
        kekulize      : write Kekulé form (explicit single/double bonds) rather
                        than aromatic bonds — required by some downstream tools
        unique_smiles : pre-computed list of unique canonical SMILES
        unique_mols   : pre-computed list of unique Mol objects (same order)

    Returns:
        Number of unique valid molecules written.
    """
    # ------------------------------------------------------------------
    # 1. Deduplication — O(n) in-process SMILES conversion
    #    Skipped entirely when the caller provides pre-computed uniques.
    # ------------------------------------------------------------------
    if unique_smiles is not None and unique_mols is not None:
        u_smiles, u_mols = unique_smiles, unique_mols
    else:
        # FIX 3 (applied here too): convert in-process to avoid pickling Mol objects.
        smi_results: List[Optional[str]] = [_mol_to_smiles(m) for m in mols]

        seen: Dict[str, Chem.Mol] = {}
        for smi, mol in zip(smi_results, mols):
            if smi is not None and mol is not None and smi not in seen:
                seen[smi] = mol
        u_smiles = list(seen.keys())
        u_mols   = list(seen.values())

    if not u_mols:
        return 0

    # ------------------------------------------------------------------
    # 2. Property computation + SDF-block rendering — in-process
    #    (pickling Mol objects into a ProcessPoolExecutor is unreliable
    #     and often slower than in-process for typical batch sizes)
    # ------------------------------------------------------------------
    blocks: List[Optional[str]] = [
        _mol_to_sdf_block((mol, smi, train_smiles, kekulize))
        for mol, smi in zip(u_mols, u_smiles)
    ]

    # ------------------------------------------------------------------
    # 3. Single buffered write — one syscall for the entire file
    # ------------------------------------------------------------------
    valid_blocks = [b for b in blocks if b is not None]
    with open(path, 'w') as fh:
        fh.write(''.join(valid_blocks))

    return len(valid_blocks)

def evaluate(
    model,
    scheduler,
    vocab,
    train_smiles: set,
    device,
    n_samples:    int   = 512,
    sample_steps: int   = 25,
    eta:          float = 1.0,
    batch_size:   int   = 32,
    mol_timeout:  float = 2.0,
) -> Dict[str, float]:
    """
    Full evaluation pass. Returns a dict of all metrics plus stopping_score.

    stopping_score = validity × uniqueness
      - Drops when model collapses (validity falls) or mode-collapses (uniqueness falls)
      - Does NOT depend on the training reward function (avoids Goodhart's law)
      - Simple, interpretable, and fast to compute
    """
    samples = sample_molecules(
        model, scheduler, vocab, device,
        n_samples=n_samples,
        sample_steps=sample_steps,
        eta=eta,
        batch_size=batch_size,
    )

    # All CUDA work done on the main process to avoid process-safety issues.
    # coords scaled to Angstroms here (×2.2), then moved to CPU for subprocess pickling.
    worker_args = [
        (
            vocab[atom_types_oh.argmax(-1)].cpu().tolist(),
            (coords * 2.2).cpu().float().numpy().tolist(),
        )
        for coords, atom_types_oh in samples
    ]

    # FIX 2 (applied): use a single shared pool for all molecules instead of
    # spawning a new pool per molecule.  The old per-molecule loop created and
    # destroyed 512+ processes serially, paying the full spawn cost every time.
    mols, n_timeouts = _coords_to_mol_batch(worker_args, timeout=mol_timeout)

    if n_timeouts > 0:
        print(f'[evaluate] Warning: {n_timeouts}/{n_samples} mols timed out in bond inference')

    validity, uniqueness, novelty, unique_smiles, unique_mols = \
        compute_validity_uniqueness_novelty(mols, train_smiles)

    # Reuse unique_mols — no re-parsing of SMILES required
    diversity = compute_diversity_from_mols(unique_mols)
    drug      = compute_drug_properties(unique_mols)

    stopping_score = validity * uniqueness
    
    return {
        # ── Core generative quality ──────────────────────────────────────
        'validity':       validity,
        'uniqueness':     uniqueness,
        'novelty':        novelty,
        'diversity':      diversity,
        'stopping_score': stopping_score,
        # ── Molecular quality ────────────────────────────────────────────
        'qed':            drug['qed'],
        'sa_score':       drug['sa'],
        'mol_weight':     drug['mw'],
        # ── Counts ───────────────────────────────────────────────────────
        'n_valid':        int(validity   * n_samples),
        'n_unique':       int(uniqueness * validity * n_samples),
        # ── Diagnostics ─────────────────────────────────────────────────
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

    In train.py, add to Trainer:
        from lightning.pytorch.callbacks import EarlyStopping
        early_stop = EarlyStopping(
            monitor='val/stopping_score',
            mode='max',
            patience=20,
            min_delta=0.001,
        )
        trainer = pl.Trainer(..., callbacks=[..., early_stop])

    Set check_val_every_n_epoch or val_check_interval in Trainer to control
    how often validation runs (recommend every 200–500 training steps).
    """

    # Set these in __init__ of the main class:
    #   self.train_smiles : set of canonical SMILES from QM9 training split
    #   self.val_n_samples : int, e.g. 256 during training, 2048 for final eval

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
                self.log(f'val/{k}', v, prog_bar=(k == 'stopping_score'),
                         sync_dist=True)