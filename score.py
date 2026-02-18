import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdMolDescriptors import CalcMolFormula # Necessary for robust molecule creation
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import QED # For QED calculation
from rdkit.Chem import RDConfig
import os, sys
import torch
import argparse
import gc
from tqdm import tqdm
from data.dataloader import build_qm9_dataloader, load_qm9_smiles_from_dataset
from model.model import TabascoV2
from model.scheduler import precompute_schedule
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob, categorical_reverse_step
from RL.validation import evaluate
# Append the path to the SAScore implementation (standard practice for RDKit SAscore)
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # Requires the SA_Score files from RDKit contrib
from diffusers import DDIMScheduler
from config import DDIM_config


# --- IV. Example Usage ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=200_000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--sample-steps', type=int, default=25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    module, enc2atom, _ = build_qm9_dataloader()
    module.setup()
    dl = load_qm9_smiles_from_dataset(module.train_dataset)
    device = 'cuda'
    ABSORB_IDX = len(enc2atom)
    enc2atom = enc2atom.to(device)
    checkpoint = torch.load('logs/TrainingSDPO/ckpts/epoch=0-step=11600-Reward0_mean=-1.9925.ckpt')['state_dict']
    checkpoint = {k[7 + k[6:].index('.'):]: v for k, v in checkpoint.items() if 'model' in k}
    tabasco = TabascoV2(
        atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, pos_coord_dim=128,
        pair_rbf_centers=args.d_model // 2, dropout=0.1
    )
    tabasco.load_state_dict(checkpoint)
    tabasco = tabasco.to(device)
    scheduler = DDIMScheduler.from_config(DDIM_config)
    
    results = evaluate(tabasco, scheduler, enc2atom, dl, args.device, 1000)
    with open('preliminary_test.txt', 'w') as f:
        f.write('Generated 1000 mols\n')
        for k, v in results.items():
            f.write(f'{k}: {v}\n')
        
        
    # Expected results for mock data:
    # Valid = 80/100 = 80.0%
    # Unique = (70+10)/80 = 100.0% (assuming C2H6 and H2O are unique)
    # Novel = 70/80 = 87.5% (C2H6 is novel, H2O is not)