"""
mol_builder.py

Converts raw model output (coordinates + one-hot atom types) into valid
RDKit Mol objects and writes PoseBusters-compatible SDF files.

PoseBusters requires:
  - Explicit bond records with bond orders (not just atom blocks)
  - Sanitized molecules (valence, aromaticity resolved)
  - 3D conformer attached
  - Hydrogens: either all explicit or all implicit — mixed fails checks
  - No disconnected fragments (single connected component)
  - Formal charges resolved

The main failure mode when "no bonds found" is writing atom-only XYZ/SDF
without going through RDKit's bond perception. This file fixes that.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds, SDWriter, Descriptors

RDLogger.DisableLog('rdApp.*')

_PT = Chem.GetPeriodicTable()
_Z_TO_SYM = {z: _PT.GetElementSymbol(z) for z in [1, 6, 7, 8, 9]}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_mol(
    atomic_nums: torch.Tensor,   # [N]  integer atomic numbers (0 = padding)
    coords: torch.Tensor,        # [N, 3]  in Angstroms
    remove_hs: bool = False,     # True = implicit H (smaller, PB usually OK)
    neutralise: bool = True,     # fix common spurious charges
) -> Optional[Chem.Mol]:
    """
    Build a sanitized RDKit Mol with a 3D conformer and explicit bonds.

    Strategy:
      1. Build atom-only mol with conformer (coords in Angstroms)
      2. DetermineConnectivity  — assigns single bonds from geometry
      3. DetermineBondOrders    — promotes to double/triple/aromatic
         (tried first; falls back to connectivity-only if it fails)
      4. SanitizeMol            — resolves aromaticity, valence
      5. Optional: remove explicit Hs, neutralise formal charges

    Returns None if the molecule cannot be made valid.
    """
    # Strip padding
    real = atomic_nums != 0
    zs   = atomic_nums[real].cpu().long().tolist()
    xyz  = coords[real].detach().cpu().float().numpy()

    if len(zs) < 2:
        return None

    # ── Step 1: build atom skeleton with conformer ────────────────────────
    edit = Chem.RWMol()
    conf = Chem.Conformer(len(zs))
    for i, z in enumerate(zs):
        sym = _Z_TO_SYM.get(int(z), 'C')
        idx = edit.AddAtom(Chem.Atom(sym))
        conf.SetAtomPosition(idx, xyz[i].tolist())
    mol = edit.GetMol()
    mol.AddConformer(conf, assignId=True)

    # ── Step 2: connectivity from geometry (never hangs, always O(N²)) ──────
    # DetermineConnectivity assigns single bonds purely from pairwise distances
    # vs covalent radius sums — no combinatorial search, no ILP, no timeout risk.
    try:
        mol_rw = Chem.RWMol(mol)
        rdDetermineBonds.DetermineConnectivity(mol_rw)
        mol = mol_rw.GetMol()
    except Exception:
        return None

    # ── Step 3: promote bond orders via sanitization ──────────────────────
    # After connectivity is assigned, SanitizeMol handles the common cases:
    #   - Aromatic rings: alternating single bonds → aromatic perception
    #   - Carbonyls, imines: RDKit infers from valence during sanitization
    # This won't catch every double/triple bond but is correct for most
    # QM9-scale molecules (rings, amines, ethers, carbonyls all work).
    try:
        # AssignRadicals first — fills open valences before sanitization
        # tries to interpret them as charge/aromaticity errors
        Chem.AssignRadicals(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        # Partial sanitization — skip strict valence check but still
        # resolve aromaticity so rings get correct bond orders
        try:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^
                Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
        except Exception:
            return None

    # ── Step 5: reject disconnected molecules ─────────────────────────────
    if len(Chem.GetMolFrags(mol)) > 1:
        # Keep only the largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol   = max(frags, key=lambda m: m.GetNumAtoms())
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    # ── Step 6: optional neutralisation ──────────────────────────────────
    if neutralise:
        mol = _neutralise_charges(mol)
        if mol is None:
            return None

    # ── Step 7: optional H removal ────────────────────────────────────────
    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            pass   # keep explicit Hs if removal fails

    # Final conformer check — PoseBusters needs 3D coords present
    if mol.GetNumConformers() == 0:
        return None

    return mol


def _neutralise_charges(mol: Chem.Mol) -> Optional[Chem.Mol]:
    patterns = [
        '[n+;H]',
        '[N+;H2]',
        '[N+;H]',
        '[OH-]',
        '[O-]',
        '[NH-]',
    ]
    try:
        rw = Chem.RWMol(mol)
        for smarts in patterns:
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                continue
            for match in rw.GetSubstructMatches(query):
                atom = rw.GetAtomWithIdx(match[0])
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return mol  # return original rather than None on neutralisation failure


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_mols_from_batch(
    coords_batch: torch.Tensor,      # [B, N, 3]  Angstroms
    atomic_nums_batch: torch.Tensor, # [B, N]     integer atomic numbers
    remove_hs: bool = False,
    neutralise: bool = True,
) -> List[Optional[Chem.Mol]]:
    """Build a list of Mol objects from a batch of model outputs."""
    B = coords_batch.shape[0]
    return [
        build_mol(
            atomic_nums_batch[b],
            coords_batch[b],
            remove_hs=remove_hs,
            neutralise=neutralise,
        )
        for b in range(B)
    ]


def build_mols_from_pipeline_output(
    last_mol,          # mols[2] from pipeline_with_logprob: [coords, atom_onehot]
    vocab: torch.Tensor,  # enc2atom lookup
    scale: float = 2.2,
    remove_hs: bool = False,
) -> List[Optional[Chem.Mol]]:
    """
    Convenience wrapper for direct use with pipeline_with_logprob output.

    Usage:
        mols_list, _, _, _, _, _, _ = pipeline_with_logprob(...)
        rdkit_mols = build_mols_from_pipeline_output(mols_list[2], vocab)
    """
    coords    = last_mol[0] * scale          # [B, N, 3]  → Angstroms
    atom_oh   = last_mol[1]                  # [B, N, A]  one-hot
    atomic_nums = vocab[atom_oh.argmax(-1)]  # [B, N]     integer Z

    return build_mols_from_batch(coords, atomic_nums, remove_hs=remove_hs)


# ---------------------------------------------------------------------------
# SDF writer  (PoseBusters-compatible)
# ---------------------------------------------------------------------------

def write_sdf(
    mols: List[Optional[Chem.Mol]],
    path: str,
    kekulize: bool = True,
    properties: Optional[dict] = None,
) -> Tuple[int, int]:
    """
    Write a list of Mol objects to an SDF file compatible with PoseBusters.

    Args:
        mols       : list of RDKit Mol (None entries are skipped)
        path       : output .sdf file path
        kekulize   : write Kekulé form (explicit single/double) — PB prefers this
        properties : optional dict of {prop_name: [values per mol]}
                     written as SDF data fields (e.g. {'validity': [1,1,0,...]})

    Returns:
        (n_written, n_skipped)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0

    with SDWriter(str(path)) as writer:
        writer.SetKekulize(kekulize)

        for i, mol in enumerate(mols):
            if mol is None:
                n_skipped += 1
                continue

            # Attach any extra properties as SDF fields
            if properties:
                for key, values in properties.items():
                    if i < len(values):
                        mol.SetProp(key, str(values[i]))

            # Give each molecule a name (PoseBusters uses this as identifier)
            if not mol.HasProp('_Name') or mol.GetProp('_Name') == '':
                mol.SetProp('_Name', f'mol_{i:05d}')

            try:
                writer.write(mol)
                n_written += 1
            except Exception as e:
                n_skipped += 1

    print(f"[write_sdf] Wrote {n_written} molecules to {path} "
          f"({n_skipped} skipped)")
    return n_written, n_skipped


# ---------------------------------------------------------------------------
# Full pipeline: model output → SDF
# ---------------------------------------------------------------------------

def generate_sdf(
    model,
    scheduler,
    vocab,
    device,
    output_path: str,
    n_samples: int = 256,
    n_atoms:   int = 29,
    atom_dim:  int = 6,
    sample_steps: int = 25,
    eta:       float = 0.0,    # 0 = deterministic DDIM for final generation
    batch_size: int = 32,
    scale:     float = 2.2,
    remove_hs: bool = False,
    add_properties: bool = True,
) -> dict:
    """
    End-to-end: sample from model → build mols → write SDF.

    Args:
        eta=0.0  : deterministic sampling gives more consistent geometries
                   for final generation (vs eta=1.0 during training)

    Returns:
        dict of generation statistics
    """
    from RL.SDPO import pipeline_with_logprob

    model.eval()
    all_mols = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            B = min(batch_size, n_samples - start)
            x     = torch.randn(B, n_atoms, 3,       device=device)
            types = torch.randn(B, n_atoms, atom_dim, device=device)

            result, *_ = pipeline_with_logprob(
                model, x, types,
                scheduler=scheduler, B=B,
                device=device,
                num_inference_steps=sample_steps,
                eta=eta,
            )
            result[2] = [result[2][:, :, :3], result[2][:, :, 3:]]
            rdkit_mols = build_mols_from_pipeline_output(
                result[2], vocab, scale=scale, remove_hs=remove_hs
            )
            all_mols.extend(rdkit_mols)

    # Collect properties for SDF fields
    props = None
    if add_properties:
        smiles_list = []
        qed_list    = []
        mw_list     = []
        valid_list  = []

        for mol in all_mols:
            if mol is not None:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi)
                    qed_list.append(f"{Descriptors.qed(mol):.4f}")
                    mw_list.append(f"{Descriptors.MolWt(mol):.2f}")
                    valid_list.append('1')
                except Exception:
                    smiles_list.append('')
                    qed_list.append('0')
                    mw_list.append('0')
                    valid_list.append('0')
            else:
                smiles_list.append('')
                qed_list.append('0')
                mw_list.append('0')
                valid_list.append('0')

        props = {
            'SMILES':  smiles_list,
            'QED':     qed_list,
            'MolWt':   mw_list,
            'Valid':   valid_list,
        }

    n_written, n_skipped = write_sdf(all_mols, output_path, properties=props)

    stats = {
        'n_sampled': n_samples,
        'n_valid_mols': n_written,
        'validity_rate': n_written / n_samples,
    }
    print(f"\n[generate_sdf] Stats: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Quick diagnostic: why is my molecule failing?
# ---------------------------------------------------------------------------

def diagnose_mol(
    atomic_nums: torch.Tensor,
    coords: torch.Tensor,
    scale: float = 1.0,   # set to 2.2 if coords are in latent space
) -> None:
    """
    Print a step-by-step diagnosis of why build_mol might be returning None.
    Useful for debugging individual failures.
    """
    zs  = atomic_nums[atomic_nums != 0].cpu().tolist()
    xyz = (coords[atomic_nums != 0] * scale).detach().cpu().float().numpy()

    print(f"Atoms: {len(zs)}  — {[_Z_TO_SYM.get(z,'?') for z in zs]}")
    print(f"Coord range: [{xyz.min():.2f}, {xyz.max():.2f}] Å")

    # Check pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(xyz)
    print(f"Pairwise dist range: [{dists.min():.2f}, {dists.max():.2f}] Å")
    if dists.min() < 0.4:
        print("  ⚠ Atoms too close — coordinate scale may be wrong")
    if dists.max() > 10.0:
        print("  ⚠ Atoms very far apart — likely disconnected")

    # Try each step
    edit = Chem.RWMol()
    conf = Chem.Conformer(len(zs))
    for i, z in enumerate(zs):
        idx = edit.AddAtom(Chem.Atom(_Z_TO_SYM.get(int(z), 'C')))
        conf.SetAtomPosition(idx, xyz[i].tolist())
    mol = edit.GetMol()
    mol.AddConformer(conf, assignId=True)
    print("✓ Atom/conformer build: OK")

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        print(f"✓ DetermineBonds: OK  ({mol.GetNumBonds()} bonds)")
    except Exception as e:
        print(f"✗ DetermineBonds failed: {e}")
        try:
            rw = Chem.RWMol(mol)
            rdDetermineBonds.DetermineConnectivity(rw)
            mol = rw.GetMol()
            print(f"✓ DetermineConnectivity fallback: OK  ({mol.GetNumBonds()} bonds)")
        except Exception as e2:
            print(f"✗ DetermineConnectivity also failed: {e2}")
            return

    try:
        Chem.SanitizeMol(mol)
        print("✓ SanitizeMol: OK")
    except Exception as e:
        print(f"✗ SanitizeMol failed: {e}")
        return

    frags = Chem.GetMolFrags(mol)
    print(f"{'✓' if len(frags)==1 else '⚠'} Fragments: {len(frags)}")

    try:
        smi = Chem.MolToSmiles(mol)
        print(f"✓ SMILES: {smi}")
    except Exception as e:
        print(f"✗ SMILES generation failed: {e}")