from RL.validation import compute_metrics
from score import compute_train_stats
import os
import pickle
import rdkit.Chem as Chem
import argparse
from data.dataloader import build_qm9_dataloader

def main(models_dir, train_smiles):
    for model in os.listdir(models_dir):
        p = os.path.join(models_dir, model)
        print(model)
        if p.endswith('sdf') or p.endswith('pkl'):

            outfile = p[:-3] + 'txt'
            if 'pkl' in p:
                with open(p, 'rb') as f:
                    mols = pickle.load(f)
            else:
                mols = []
                with Chem.MultithreadedSDMolSupplier(p) as sdSupl:
                    for mol in sdSupl:
                        if mol is not None:
                            mols.append(mol)

            metrics, umols, usmiles = compute_metrics(mols, train_smiles)
            lines = []
            hr    = '─' * 64

            def h(title: str):
                lines.append('')
                lines.append(hr)
                lines.append(f'  {title}')
                lines.append(hr)

            lines.append('Tabasco Evaluation Report')
            lines.append(f'Samples   : 1000')

            h('Production Metrics')
            lines.append(f'  Validity    {metrics["validity"]:.3f}  ({metrics["n_valid"]}/1000)')
            lines.append(f'  Uniqueness  {metrics["uniqueness"]:.3f}  ({metrics["n_unique"]} unique)')
            lines.append(f'  Novelty     {metrics["novelty"]:.3f}')
            lines.append(f'  Diversity   {metrics["diversity"]:.3f}')
            lines.append(f'  Timeouts    {int(metrics.get("n_timeouts", 0))}')

            h('Stopping Score  (validity × uniqueness × weighted_quality)')
            lines.append(f'  stopping_score  {metrics["stopping_score"]:.4f}')
            lines.append(f'  gate            {metrics["gate"]:.4f}  (validity × uniqueness)')
            lines.append(f'  quality         {metrics["quality"]:.4f}  (0.50·QED + 0.30·SA_norm + 0.20·novelty)')

            with open(outfile, 'w') as fh:
                fh.write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    _, _, _, train_smiles = build_qm9_dataloader(
        root='data/QM9', batch_size=1,
    )

    main('../generated', train_smiles)
    
