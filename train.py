import os 
import torch
import argparse
from typing import Dict
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from model.model import TabascoV2
from data.dataloader import build_qm9_dataloader
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob, categorical_reverse_step
from RL.rewards import get_reward
from rdkit import RDLogger, Chem      
from torchani.models import ANI2x 
from diffusers import DDIMScheduler 
from config import DDIM_config
from RL.rewards import ValidityReward

RDLogger.DisableLog('rdApp.*')  

class RunningStats:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.mean = None
        self.std = None

    def update(self, x):
        batch_mean = x.mean()
        batch_std = x.std()

        if self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
        else:
            # Exponential Moving Average
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.std = self.momentum * self.std + (1 - self.momentum) * batch_std

    def normalize(self, x):
        if self.mean is None:
            return x
        # Subtract moving mean and divide by moving std
        return (x - self.mean) / (self.std + 1e-8)

def vdW():
    ptable = Chem.GetPeriodicTable()

    # Example: list of atomic numbers you care about
    atomic_numbers = [1, 6, 7, 8, 9,]  # H, C, N, O, F

    # Extract vdW radii
    vdW_radii = {}
    for Z in atomic_numbers:
        radius = ptable.GetRvdw(Z)  # Returns van der Waals radius in Å
        vdW_radii[Z] = radius
    return vdW_radii


def ema_avg_fn(averaged_model_parameter, current_model_parameter, num_averaged):
    # classic EMA formula: v = decay*v + (1-decay)*x
    decay = 0.999  # tune this later
    return decay * averaged_model_parameter + (1 - decay) * current_model_parameter
    
def train(args):
    checkpoint_callback = ModelCheckpoint(dirpath="logs/TrainingSDPO/ckpts/", save_top_k=5, monitor="Reward0_mean", mode='max', filename='{epoch}-{step}-{reward0_mean:.4f}', save_last=True, every_n_train_steps=100)
    EMA = StochasticWeightAveraging(1e-3, avg_fn=ema_avg_fn)
    torch.set_float32_matmul_precision('high')
    
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1, 
                         precision='32',
                         max_steps=args.max_steps,
                         enable_progress_bar=True, 
                         logger=CSVLogger("logs", name="TrainingSDPO", flush_logs_every_n_steps=1), 
                         log_every_n_steps=1, 
                         callbacks=[checkpoint_callback])
    module, vocab_enc2atom, vocab_atom2enc = build_qm9_dataloader(root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    ABSORB_IDX = len(vocab_enc2atom)
    checkpoint = torch.load('checkpoints/Pretrain.ckpt')['state_dict']
    #checkpoint = {k[6:]: v for k,v in checkpoint.items()}
    checkpoint = {k[7 + k[6:].index('.'):]: v for k,v in checkpoint.items() if 'model' in k}
    tabasco = TabascoV2(atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, pos_coord_dim=128, pair_rbf_centers=args.d_model//2, dropout=0.1)
    tabasco.load_state_dict(checkpoint)
    #ani = ANI2x(periodic_table_index=True)
    #ani = ani.eval()
    rewarder = ValidityReward()
    args.vdW = vdW()
    model = LightningTabascoPipe(tabasco, rewarder, args, ABSORB_IDX, vocab_enc2atom)
    
    trainer.fit(model=model, datamodule=module) 

class LightningTabascoPipe(pl.LightningModule):
    def __init__(self, tabasco: TabascoV2, rewarder, args: dict, ABSORB_IDX: int, vocab: Dict):
        super().__init__()
        self.args = args
        self.model = tabasco
        self.scheduler = DDIMScheduler.from_config(DDIM_config)
        
        self.ABSORB_IDX = ABSORB_IDX
        self.pad = torch.zeros((ABSORB_IDX, ), device=args.device)
        self.pad[0] = 1
        self.vocab = vocab
        lam = torch.tensor(self.args.lam, device=args.device)
        lam = lam.pow(torch.arange(self.args.sample_steps, dtype=torch.float, device=args.device))
        self.register_buffer('lam', lam.repeat(self.args.batch_size, 1))
        self.adv_normalizer = RunningStats()

        self.adv_clip_max = 1e-2
        self.clip_range = 1e-4

        self.eta = 1.0
        self.advantage_scale = 1.0
        self.best_loss = None
        self.paths = []
        self.accumulation = 0
        self.accumulation_steps = 4
        
        self.rewarder = rewarder
        self.vocab = self.vocab.to(args.device)
        
    def configure_optimizers(self):
        # Initialize optimizer with ONLY the trainable parameters
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=0.01, betas= (0.9, 0.95))
        return optimizer

    def _compute_log_probs_batch(self, coords, atoms, next_coords, next_atoms, timesteps):
        """Batched forward pass for computing log probs - compiled for speed"""
        coord_pred, types_pred = self.model(atoms, coords, timesteps, batched=True)
        
        _, _, log_prob_coord = ddim_step_with_logprob(
            self.scheduler, coord_pred, timesteps, coords,
            eta=self.eta, x_prev=next_coords, t_batched=True
        )
        _, _, log_prob_types = categorical_reverse_step(
            self.scheduler, types_pred, timesteps, atoms,
            eta=self.eta, x_prev=next_atoms, t_batched=True
        )
        
        return log_prob_coord, log_prob_types

    def training_step(self, batch, batch_idx):
        # Sample from policy (frozen for this step)
        self.model.eval()
        
        # Generate initial noise once and reuse
        x = torch.randn(self.args.batch_size, 29, 3, device=self.device)
        types = torch.randn(self.args.batch_size, 29, 6, device=self.device)
        
        # Generate two trajectories with same starting point
        with torch.no_grad():
            mols1, all_mols1, log_probs1, anchor_steps1, sim_first1, sim_anchor1, sim_last1 = pipeline_with_logprob(
                self.model, x, types,
                num_inference_steps=self.args.sample_steps,
                scheduler=self.scheduler,
                B=self.args.batch_size,
                device=self.device,
                eta=self.eta
            )
            
            mols2, all_mols2, log_probs2, anchor_steps2, sim_first2, sim_anchor2, sim_last2 = pipeline_with_logprob(
                self.model, x, types,
                num_inference_steps=self.args.sample_steps,
                scheduler=self.scheduler,
                B=self.args.batch_size,
                device=self.device,
                eta=self.eta
            )
        
        # Stack trajectories - optimize memory layout
        coords1 = torch.stack([m[0] for m in all_mols1], dim=1)  # [B, T, N, 3]
        atoms1 = torch.stack([m[1] for m in all_mols1], dim=1)   # [B, T, N, A]
        coords2 = torch.stack([m[0] for m in all_mols2], dim=1)
        atoms2 = torch.stack([m[1] for m in all_mols2], dim=1)
        
        # Split log probs - FIX: split along correct dimension
        # log_probs shape should be [B, T, 2] where last dim is [coord_logprob, atom_logprob]
        log_probs_coord1, log_probs_atom1 = log_probs1.split(1, dim=-1)
        log_probs_coord2, log_probs_atom2 = log_probs2.split(1, dim=-1)
        
        # Compute rewards
        with torch.no_grad():
            rewards1 = get_reward(mols1, self.rewarder, self.vocab, self.args.vdW).T  # [B, 3]
            rewards2 = get_reward(mols2, self.rewarder, self.vocab, self.args.vdW).T  # [B, 3]
            
            # FIX: Correct reward indexing - rewards are [R_last, R_anchor, R_start]
            # Index 0: last step, Index 1: anchor, Index 2: start step
            
            # Log mean rewards
            self.log('Rewards_1_start', rewards1[:, 2].mean())
            self.log('Rewards_1_anchor', rewards1[:, 1].mean())
            self.log('Rewards_1_last', rewards1[:, 0].mean())
            self.log('Rewards_2_start', rewards2[:, 2].mean())
            self.log('Rewards_2_anchor', rewards2[:, 1].mean())
            self.log('Rewards_2_last', rewards2[:, 0].mean())
            self.log('Reward0_mean', (rewards1[:, 2].mean() + rewards2[:, 2].mean()) / 2, on_step=True)
            
            # FIX: Interpolate rewards correctly across timesteps
            # rewards[:, 2] = R_start, rewards[:, 1] = R_anchor, rewards[:, 0] = R_last
            rewards_interp1 = (
                rewards1[:, 2:3].expand(-1, self.args.sample_steps) * sim_first1
                + rewards1[:, 1:2].expand(-1, self.args.sample_steps) * sim_anchor1
                + rewards1[:, 0:1].expand(-1, self.args.sample_steps) * sim_last1
            ) / (sim_first1 + sim_anchor1 + sim_last1 + 1e-8)
            
            rewards_interp2 = (
                rewards2[:, 2:3].expand(-1, self.args.sample_steps) * sim_first2
                + rewards2[:, 1:2].expand(-1, self.args.sample_steps) * sim_anchor2
                + rewards2[:, 0:1].expand(-1, self.args.sample_steps) * sim_last2
            ) / (sim_first2 + sim_anchor2 + sim_last2 + 1e-8)
            
            # Override exact rewards at anchor points
            rewards_interp1[:, 0] = rewards1[:, 2]  # Start
            rewards_interp1[:, -1] = rewards1[:, 0]  # Last
            batch_idx_range = torch.arange(self.args.batch_size, device=self.device)
            rewards_interp1[batch_idx_range, anchor_steps1] = rewards1[:, 1]  # Anchor
            
            rewards_interp2[:, 0] = rewards2[:, 2]  # Start
            rewards_interp2[:, -1] = rewards2[:, 0]  # Last
            rewards_interp2[batch_idx_range, anchor_steps2] = rewards2[:, 1]  # Anchor
            
            # FIX: Compute returns properly without in-place corruption
            returns1 = rewards_interp1.clone()
            returns2 = rewards_interp2.clone()
            
            for t in reversed(range(self.args.sample_steps - 1)):
                returns1[:, t] += self.args.gamma * returns1[:, t + 1]
                returns2[:, t] += self.args.gamma * returns2[:, t + 1]
            
            # Normalize advantages
            returns_all = torch.cat([returns1, returns2], dim=0)
            returns_normalized = (returns_all - returns_all.mean()) / (returns_all.std() + 1e-8)
            adv1, adv2 = returns_normalized.chunk(2, dim=0)
            
            # Clip advantages
            adv1 = torch.clamp(adv1, -self.adv_clip_max, self.adv_clip_max)
            adv2 = torch.clamp(adv2, -self.adv_clip_max, self.adv_clip_max)
        
        # Training mode for gradient computation
        self.model.train()
        
        # Prepare data for batched computation
        timesteps = self.scheduler.timesteps.unsqueeze(-1)  # [T, 1]
        # Batch both trajectories together for efficiency
        coords_batch = torch.cat([coords1[:, :-1], coords2[:, :-1]], dim=0)  # [2B, T-1, N, 3]
        atoms_batch = torch.cat([atoms1[:, :-1], atoms2[:, :-1]], dim=0)     # [2B, T-1, N, 6]
        next_coords_batch = torch.cat([coords1[:, 1:], coords2[:, 1:]], dim=0)
        next_atoms_batch = torch.cat([atoms1[:, 1:], atoms2[:, 1:]], dim=0)
        
        # Flatten batch and time dimensions for vectorized forward pass
        B, T_minus_1, N, C = coords_batch.shape
        coords_flat = coords_batch#.reshape(B * T_minus_1, N, C)
        atoms_flat = atoms_batch#.reshape(B * T_minus_1, N, -1)
        next_coords_flat = next_coords_batch#.reshape(B * T_minus_1, N, C)
        next_atoms_flat = next_atoms_batch#.reshape(B * T_minus_1, N, -1)
        timesteps_flat = timesteps.repeat(B, 1).reshape(-1)
        
        # Single batched forward pass
        log_prob_coord, log_prob_types = self._compute_log_probs_batch(
            coords_flat, atoms_flat, next_coords_flat, next_atoms_flat, timesteps_flat
        )
        
        # Split into trajectory 1 and 2
        log_prob_coord1_new, log_prob_coord2_new = log_prob_coord.chunk(2, dim=0)
        log_prob_types1_new, log_prob_types2_new = log_prob_types.chunk(2, dim=0)
        
        # Get old log probs
        logp_coord1_old = log_probs_coord1.squeeze(-1)
        logp_atom1_old = log_probs_atom1.squeeze(-1)
        logp_coord2_old = log_probs_coord2.squeeze(-1)
        logp_atom2_old = log_probs_atom2.squeeze(-1)
        
        # Compute log ratios with clipping
        logc1 = torch.clamp(log_prob_coord1_new, logp_coord1_old - self.clip_range, logp_coord1_old + self.clip_range)
        logc2 = torch.clamp(log_prob_coord2_new, logp_coord2_old - self.clip_range, logp_coord2_old + self.clip_range)
        loga1 = torch.clamp(log_prob_types1_new, logp_atom1_old - self.clip_range, logp_atom1_old + self.clip_range)
        loga2 = torch.clamp(log_prob_types2_new, logp_atom2_old - self.clip_range, logp_atom2_old + self.clip_range)
        
        # Compute log probability ratios
        log_ratio1 = (log_prob_coord1_new - logp_coord1_old) + (log_prob_types1_new - logp_atom1_old)
        log_ratio2 = (log_prob_coord2_new - logp_coord2_old) + (log_prob_types2_new - logp_atom2_old)
        log_diff = log_ratio1 - log_ratio2
        
        # FIX: Use correct old log probs for trajectory 2
        log_ratio1_clipped = (logc1 - logp_coord1_old) + (loga1 - logp_atom1_old)
        log_ratio2_clipped = (logc2 - logp_coord2_old) + (loga2 - logp_atom2_old)
        log_diff_clipped = log_ratio1_clipped - log_ratio2_clipped
        
        # Apply temporal weighting
        log_weights = self.lam[:self.args.batch_size, :T_minus_1] / self.args.log_scale
        
        # Compute advantage difference
        adv_diff = self.advantage_scale * (adv1 - adv2)
        
        # SDPO loss
        log_diff_weighted = log_weights * log_diff
        log_diff_clipped_weighted = log_weights * log_diff_clipped
        
        loss = torch.square(log_diff_weighted - adv_diff)
        loss_clipped = torch.square(log_diff_clipped_weighted - adv_diff)
        loss_sdpo = torch.mean(torch.maximum(loss_clipped, loss))
        
        # Logging
        if self.args.debug:
            print(f"\n--- Signal Diagnostics (Step {self.global_step}) ---")
            print(f"Adv Diff Mean: {adv_diff.mean().item():.6f}")
            print(f"Adv Diff Std:  {adv_diff.std().item():.6f}")
            print(f"Log_Diff Mean: {log_diff.mean().item():.6f}")
            print(f"Log_Diff Std:  {log_diff.std().item():.6f}")
            print(f"Correlation:   {torch.corrcoef(torch.stack([log_diff.flatten(), adv_diff.flatten()]))[0, 1].item():.4f}")
            print("---------------------------------------------------")
            print("rewards1 raw:", rewards1.min().item(), rewards1.max().item())
            print("returns1 before norm:", returns1.min().item(), returns1.max().item())
            print("adv_diff after clip:", adv_diff.min().item(), adv_diff.max().item())
            print("log_diff before weights:", log_diff.min().item(), log_diff.max().item())
            print("log_weights:", log_weights.min().item(), log_weights.max().item())
                    
        self.log('log_diff', log_diff.mean())
        self.log('advantage_diff', adv_diff.mean())
        self.log('Training_loss', loss_sdpo)
        
        return loss_sdpo


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=200_000)
    parser.add_argument('--inner_epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_scale', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--ckpt-dir', type=str, default='logs/TrainingSDPO/ckpts')
    parser.add_argument('--sample-steps', type=int, default=25)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--log-steps', type=int, default=100)
    parser.add_argument('--save_after', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)