"""
score.py

Offline evaluation script for a trained Tabasco checkpoint.

Outputs (all written to --output-dir, named after the checkpoint stem)
-----------------------------------------------------------------------
  <stem>_report.txt   human-readable summary of all metrics + distributions
  <stem>_mols.csv     per-molecule property table (SMILES, QED, SA, logP, MW)
  <stem>_mols.sdf     annotated SDF of all unique valid molecules
  <stem>_hist.png     property distribution histograms (generated vs training)

Usage
-----
  python score.py --ckpt logs/TrainingSDPO/ckpts/best.ckpt \\
                  --output-dir results/ \\
                  --n-samples 1000

All other arguments have sensible defaults matching the training config.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from diffusers import DDIMScheduler
from config import DDIM_config
from data.dataloader import build_qm9_dataloader
from model.model import TabascoV2
from RL.validation import (
    evaluate,
    write_sdf,
    _safe_qed,
    _safe_sa,
    _HAS_SA,
)

try:
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend — safe in scripts
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Property extraction ───────────────────────────────────────────────────────

_PROP_NAMES = ['QED', 'SA Score', 'logP', 'MW']


def _mol_properties(mol: Chem.Mol) -> Optional[Dict[str, float]]:
    """Return {QED, SA Score, logP, MW} for a valid mol, or None."""
    if mol is None:
        return None
    try:
        sa = _safe_sa(mol)
        return {
            'QED'     : _safe_qed(mol),
            'SA Score': sa if sa is not None else float('nan'),
            'logP'    : Descriptors.MolLogP(mol),
            'MW'      : Descriptors.MolWt(mol),
        }
    except Exception:
        return None


def _smiles_properties(smi: str) -> Optional[Dict[str, float]]:
    """Compute properties from a SMILES string (used for training set stats)."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return _mol_properties(mol)
    except Exception:
        return None


def _prop_stats(values: List[float]) -> Dict[str, float]:
    """Return descriptive statistics for a list of floats."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {k: float('nan') for k in ('mean', 'std', 'min', 'p25', 'p50', 'p75', 'max')}
    return {
        'mean': float(np.mean(arr)),
        'std' : float(np.std(arr)),
        'min' : float(np.min(arr)),
        'p25' : float(np.percentile(arr, 25)),
        'p50' : float(np.percentile(arr, 50)),
        'p75' : float(np.percentile(arr, 75)),
        'max' : float(np.max(arr)),
    }


# ── Training set statistics ───────────────────────────────────────────────────

def compute_train_stats(
    train_smiles : set,
    max_sample   : int = 5000,
) -> Dict[str, Dict[str, float]]:
    """
    Compute property distributions over a sample of training SMILES.

    Capped at max_sample to stay fast for large training sets.
    Returns { prop_name: { stat_name: value } }.
    """
    smiles = list(train_smiles)
    if len(smiles) > max_sample:
        rng    = np.random.default_rng(seed=42)
        smiles = rng.choice(smiles, max_sample, replace=False).tolist()

    rows = [_smiles_properties(s) for s in smiles]
    rows = [r for r in rows if r is not None]

    return {
        prop: _prop_stats([r[prop] for r in rows])
        for prop in _PROP_NAMES
    }


# ── Report writing ────────────────────────────────────────────────────────────

def _fmt_stats_row(stats: Dict[str, float]) -> str:
    return (
        f"  mean={stats['mean']:.3f}  std={stats['std']:.3f}"
        f"  min={stats['min']:.3f}  p25={stats['p25']:.3f}"
        f"  p50={stats['p50']:.3f}  p75={stats['p75']:.3f}"
        f"  max={stats['max']:.3f}"
    )


def write_report(
    path          : str,
    ckpt_path     : str,
    args          : argparse.Namespace,
    metrics       : Dict[str, float],
    gen_stats     : Dict[str, Dict[str, float]],
    train_stats   : Dict[str, Dict[str, float]],
    n_written_sdf : int,
) -> None:
    """Write a human-readable .txt report summarising all evaluation results."""

    lines = []
    hr    = '─' * 64

    def h(title: str):
        lines.append('')
        lines.append(hr)
        lines.append(f'  {title}')
        lines.append(hr)

    lines.append('Tabasco Evaluation Report')
    lines.append(f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Checkpoint: {ckpt_path}')
    lines.append(f'Samples   : {args.n_samples}')

    h('Production Metrics')
    lines.append(f'  Validity    {metrics["validity"]:.3f}  ({metrics["n_valid"]}/{args.n_samples})')
    lines.append(f'  Uniqueness  {metrics["uniqueness"]:.3f}  ({metrics["n_unique"]} unique)')
    lines.append(f'  Novelty     {metrics["novelty"]:.3f}')
    lines.append(f'  Diversity   {metrics["diversity"]:.3f}')
    lines.append(f'  Timeouts    {int(metrics.get("n_timeouts", 0))}')

    h('Stopping Score  (validity × uniqueness × weighted_quality)')
    lines.append(f'  stopping_score  {metrics["stopping_score"]:.4f}')
    lines.append(f'  gate            {metrics["gate"]:.4f}  (validity × uniqueness)')
    lines.append(f'  quality         {metrics["quality"]:.4f}  (0.50·QED + 0.30·SA_norm + 0.20·novelty)')

    h('Chemical Properties — Generated (unique valid molecules)')
    for prop in _PROP_NAMES:
        if prop in gen_stats:
            lines.append(f'  {prop:<10}{_fmt_stats_row(gen_stats[prop])}')

    h('Chemical Properties — Training Set (reference sample)')
    for prop in _PROP_NAMES:
        if prop in train_stats:
            lines.append(f'  {prop:<10}{_fmt_stats_row(train_stats[prop])}')

    h('Side-by-side Mean Comparison  (generated vs training)')
    lines.append(f'  {"Property":<12}  {"Generated":>12}  {"Training":>12}  {"Δ":>10}')
    lines.append(f'  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*10}')
    for prop in _PROP_NAMES:
        g_mean = gen_stats[prop]['mean']   if prop in gen_stats   else float('nan')
        t_mean = train_stats[prop]['mean'] if prop in train_stats else float('nan')
        delta  = g_mean - t_mean
        lines.append(f'  {prop:<12}  {g_mean:>12.3f}  {t_mean:>12.3f}  {delta:>+10.3f}')

    h('Output Files')
    stem = Path(path).stem.replace('_report', '')
    dir_ = Path(path).parent
    lines.append(f'  Report   : {path}')
    lines.append(f'  Molecules: {dir_ / (stem + "_mols.csv")}  ({metrics["n_unique"]} rows)')
    lines.append(f'  SDF      : {dir_ / (stem + "_mols.sdf")}  ({n_written_sdf} mols)')
    if _HAS_MPL:
        lines.append(f'  Histograms: {dir_ / (stem + "_hist.png")}')
    else:
        lines.append('  Histograms: skipped (matplotlib not installed)')

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def write_mol_csv(
    path         : str,
    unique_mols  : List[Chem.Mol],
    unique_smiles: List[str],
    train_smiles : set,
) -> List[Dict[str, float]]:
    """
    Write per-molecule property table to CSV.
    Returns the list of property dicts (reused for distribution stats).
    """
    rows = []
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['SMILES', 'QED', 'SA_Score', 'logP', 'MW', 'Novel'])
        for mol, smi in zip(unique_mols, unique_smiles):
            props = _mol_properties(mol)
            if props is None:
                continue
            novel = int(smi not in train_smiles)
            writer.writerow([
                smi,
                f'{props["QED"]:.4f}',
                f'{props["SA Score"]:.4f}',
                f'{props["logP"]:.4f}',
                f'{props["MW"]:.3f}',
                novel,
            ])
            rows.append(props)
    return rows


# ── Histogram ─────────────────────────────────────────────────────────────────

def write_histograms(
    path        : str,
    gen_rows    : List[Dict[str, float]],
    train_stats : Dict[str, Dict[str, float]],
) -> None:
    """
    Save a 2×2 grid of property histograms (generated vs training distribution).

    Training set is shown as a shaded normal approximation (mean ± std)
    since we only have its summary statistics, not per-molecule values.
    Generated molecules are shown as a histogram.
    """
    if not _HAS_MPL:
        print('[score] matplotlib not available — skipping histograms')
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Property Distributions: Generated vs Training', fontsize=14, y=1.01)
    axes_flat = axes.flatten()

    prop_meta = {
        'QED'     : (0.0,  1.0,  30, 'Drug-likeness (higher is better)'),
        'SA Score': (1.0,  10.0, 30, 'Synthetic accessibility (lower is better)'),
        'logP'    : (-4.0, 10.0, 40, 'Lipophilicity'),
        'MW'      : (50.0, 700.0, 40, 'Molecular weight (Da)'),
    }

    for ax, prop in zip(axes_flat, _PROP_NAMES):
        x_min, x_max, bins, desc = prop_meta[prop]

        # Generated — histogram
        gen_vals = np.array([
            r[prop] for r in gen_rows
            if prop in r and not np.isnan(r[prop])
        ])
        if len(gen_vals) > 0:
            ax.hist(
                gen_vals,
                bins=bins, range=(x_min, x_max),
                density=True, alpha=0.65,
                color='steelblue', label='Generated',
            )

        # Training — Gaussian approximation from summary stats
        if prop in train_stats:
            ts   = train_stats[prop]
            mu, sigma = ts['mean'], ts['std']
            if sigma > 1e-6:
                xs  = np.linspace(x_min, x_max, 300)
                pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                ax.plot(xs, pdf, color='tomato', linewidth=2, label='Training (approx)')
                ax.axvline(mu, color='tomato', linewidth=1, linestyle='--', alpha=0.6)

        ax.set_title(f'{prop}', fontsize=11)
        ax.set_xlabel(desc, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, args: argparse.Namespace, absorb_idx: int) -> TabascoV2:
    """
    Load a TabascoV2 model from a Lightning checkpoint.
    Strips the 'model.' prefix added by LightningTabascoPipe.
    """
    raw       = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state_dict = {
        k[len('model._orig_mod.'):]: v
        for k, v in raw.items()
        if k.startswith('model._orig_mod.')
    }

    model = TabascoV2(
        atom_vocab_size  = absorb_idx,
        d_model          = args.d_model,
        n_heads          = args.n_heads,
        n_layers         = args.n_layers,
        pos_coord_dim    = 128,
        pair_rbf_centers = args.d_model // 2,
        dropout          = 0.1,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Output paths ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    stem = Path(args.ckpt).stem   # e.g. 'epoch=0-step=300-Reward0_mean=3.9790'

    path_report = os.path.join(args.output_dir, f'{stem}_report.txt')
    path_csv    = os.path.join(args.output_dir, f'{stem}_mols.csv')
    path_sdf    = os.path.join(args.output_dir, f'{stem}_mols.sdf')
    path_hist   = os.path.join(args.output_dir, f'{stem}_hist.png')

    print(f'[score] checkpoint : {args.ckpt}')
    print(f'[score] output dir : {args.output_dir}')
    print(f'[score] n_samples  : {args.n_samples}')

    # ── Data / vocab ──────────────────────────────────────────────────────────
    _, enc2atom, _, train_smiles = build_qm9_dataloader(
        root=args.data_root, batch_size=args.batch_size
    )
    ABSORB_IDX = len(enc2atom)
    device     = torch.device(args.device)
    enc2atom   = enc2atom.to(device)

    # Build vocab dict {int → element symbol} for validation/reward functions
    vocab = enc2atom

    # ── Model + scheduler ─────────────────────────────────────────────────────
    print('[score] loading checkpoint...')
    model     = load_checkpoint(args.ckpt, args, ABSORB_IDX).to(device)
    scheduler = DDIMScheduler.from_config(DDIM_config)

    # ── Generate and evaluate ─────────────────────────────────────────────────
    print(f'[score] generating {args.n_samples} molecules...')
    start = time.time()
    metrics, unique_mols, unique_smiles = evaluate(
        model        = model,
        scheduler    = scheduler,
        vocab        = vocab,
        train_smiles = set(train_smiles),
        device       = device,
        n_samples    = args.n_samples,
        sample_steps = args.sample_steps,
        eta          = args.eta,
        batch_size   = args.batch_size,
    )
    print(f'Generaterd 1000 molecules in {time.time() - start} s')
    print(
        f'[score] validity={metrics["validity"]:.3f}  '
        f'unique={metrics["uniqueness"]:.3f}  '
        f'novel={metrics["novelty"]:.3f}  '
        f'stopping={metrics["stopping_score"]:.4f}'
    )

    # ── Per-molecule CSV + distribution stats ─────────────────────────────────
    print('[score] computing property distributions...')
    gen_rows  = write_mol_csv(path_csv, unique_mols, unique_smiles, set(train_smiles))
    gen_stats = {prop: _prop_stats([r[prop] for r in gen_rows]) for prop in _PROP_NAMES}

    # ── Training set reference stats ──────────────────────────────────────────
    print('[score] computing training set reference statistics...')
    train_stats = compute_train_stats(set(train_smiles), max_sample=args.train_sample)

    # ── SDF ───────────────────────────────────────────────────────────────────
    n_sdf = write_sdf(
        unique_mols, unique_smiles, path_sdf,
        train_smiles=set(train_smiles),
    )
    print(f'[score] wrote {n_sdf} molecules to {path_sdf}')

    # ── Histograms ────────────────────────────────────────────────────────────
    print('[score] saving histograms...')
    write_histograms(path_hist, gen_rows, train_stats)

    # ── Text report ───────────────────────────────────────────────────────────
    write_report(
        path_report, args.ckpt, args,
        metrics, gen_stats, train_stats, n_sdf,
    )
    print(f'[score] report written to {path_report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a Tabasco checkpoint and generate a full property report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument('--ckpt',        required=True,
                        help='Path to Lightning checkpoint (.ckpt)')
    # Output
    parser.add_argument('--output-dir',  default='results/',
                        help='Directory for all output files')
    # Data
    parser.add_argument('--data-root',   default='data/QM9')
    parser.add_argument('--dataset',     default='qm9')
    parser.add_argument('--train-sample', type=int, default=5000,
                        help='Max training molecules to use for reference statistics')
    # Model architecture (must match checkpoint)
    parser.add_argument('--d-model',     type=int,   default=384)
    parser.add_argument('--n-heads',     type=int,   default=12)
    parser.add_argument('--n-layers',    type=int,   default=24)
    # Sampling
    parser.add_argument('--n-samples',   type=int,   default=1000)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--sample-steps', type=int,  default=25)
    parser.add_argument('--eta',         type=float, default=1.0)
    parser.add_argument('--device',      default='cuda')
    args = parser.parse_args()
    main(args)