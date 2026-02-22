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
    module, enc2atom, _ = build_qm9_dataloader()
    module.setup()
    dl = load_qm9_smiles_from_dataset(module.train_dataset)
    device = 'cuda'
    ABSORB_IDX = len(enc2atom)
    enc2atom = enc2atom.to(device)
    checkpoint = torch.load('logs/TrainingSDPO/ckpts/epoch=0-step=15800-Reward0_mean=0.4852.ckpt')['state_dict']
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
    # Run this on one batch before build_mol to diagnose
    with torch.no_grad():
        x     = torch.randn(4, 29, 3, device=device)
        types = torch.randn(4, 29, 6, device=device)
        result, _, _, _, _, _, _ = pipeline_with_logprob(tabasco, x, types, scheduler, 'cuda')
        
        atom_oh = result[2][1]                      # [B, N, A] one-hot
        predicted_indices = atom_oh.argmax(-1)      # [B, N]
        atomic_nums = enc2atom[predicted_indices]       # [B, N]
        
        print("Predicted class indices:\n", predicted_indices[0])   # what class is predicted?
        print("Atomic numbers after vocab:\n", atomic_nums[0])      # how many non-zero?
        print("Non-padding atoms per molecule:", (atomic_nums != 0).sum(-1))
        last_coords = result[2][0]   # [B, N, 3], before scale

        print("Raw coord std:", last_coords.std().item())
        print("Scaled coord std:", (last_coords * 2.2).std().item())

        from torch.nn.functional import pdist
        dists = pdist(last_coords[0] * 2.2)
        print("Scaled pairwise dists — min:", dists.min().item(), 
                                    "mean:", dists.mean().item(),
                                    "max:", dists.max().item())
    results = generate_sdf(tabasco, scheduler, enc2atom, args.device, 'preliminary_test.sdf', 1000, eta=0.5, scale=1.0)
    
    with open('preliminary_test.txt', 'w') as f:
        f.write('Generated 1000 mols\n')
        for k, v in results.items():
            f.write(f'{k}: {v}\n')
        
        
    # Expected results for mock data:
    # Valid = 80/100 = 80.0%
    # Unique = (70+10)/80 = 100.0% (assuming C2H6 and H2O are unique)
    # Novel = 70/80 = 87.5% (C2H6 is novel, H2O is not)