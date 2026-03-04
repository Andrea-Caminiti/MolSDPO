from rdkit.Chem import RDConfig
import os, sys
import torch
import argparse
from data.dataloader import build_qm9_dataloader, load_qm9_smiles_from_dataset
from model.model import TabascoV2
from mol_builder import generate_sdf
from diffusers import DDIMScheduler
from config import DDIM_config
from RL.SDPO import pipeline_with_logprob

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
    module, enc2atom, _, smiles = build_qm9_dataloader()
    device = 'cuda'
    ABSORB_IDX = len(enc2atom)
    enc2atom = enc2atom.to(device)
    checkpoint = torch.load('logs/TrainingSDPO/ckpts/epoch=0-step=600-Reward0_mean=0.989128589630127.ckpt')['state_dict']
    #checkpoint = {k[6:]: v for k, v in checkpoint.items() if 'model' in k}
    checkpoint = {k[7 + k[6:].index('.'):]: v for k, v in checkpoint.items() if 'model' in k}
    tabasco = TabascoV2(
        atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, pos_coord_dim=128,
        pair_rbf_centers=args.d_model // 2, dropout=0.1
    )
    tabasco.load_state_dict(checkpoint)
    tabasco = tabasco.to(device)
    scheduler = DDIMScheduler.from_config(DDIM_config)
       
    results = generate_sdf(tabasco, scheduler, enc2atom, args.device, 'test1.sdf', 1000, eta=1.0, scale=1.0)
    
    with open('test1.txt', 'w') as f:
        f.write('Generated 1000 mols\n')
        for k, v in results.items():
            f.write(f'{k}: {v}\n')
        
        
    # Expected results for mock data:
    # Valid = 80/100 = 80.0%
    # Unique = (70+10)/80 = 100.0% (assuming C2H6 and H2O are unique)
    # Novel = 70/80 = 87.5% (C2H6 is novel, H2O is not)