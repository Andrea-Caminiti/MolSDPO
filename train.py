import os 
import torch
import argparse
from typing import Dict
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.model import TabascoV2
from data.dataloader import build_qm9_dataloader, load_qm9_smiles
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob, categorical_reverse_step, recompute_log_probs
from RL.energy_rewards import get_reward, EnergyRewarder
from RL.validation import ValidationMixin
from rdkit import RDLogger, Chem      
from diffusers import DDIMScheduler 
from config import DDIM_config

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
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.std  = self.momentum * self.std  + (1 - self.momentum) * batch_std

    def normalize(self, x):
        if self.mean is None:
            return x
        return (x - self.mean) / (self.std + 1e-8)


def vdW():
    ptable = Chem.GetPeriodicTable()
    return {Z: ptable.GetRvdw(Z) for Z in [1, 6, 7, 8, 9]}


def train(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/TrainingSDPO/ckpts/",
        save_top_k=5, monitor="Reward0_mean", mode='max',
        filename='{epoch}-{step}-{Reward0_mean:.4f}',
        save_last=True, every_n_train_steps=100
    )
    early_stop = EarlyStopping(
                            monitor='val/stopping_score',
                            mode='max',
                            patience=8,
                            min_delta=0.001,
                        )
    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        accelerator='gpu', devices=1,
        # OPTIMIZATION: bf16-mixed uses BF16 for forward/backward but keeps
        # master weights in FP32.  BF16 has the same dynamic range as FP32
        # (unlike FP16) so training stability is maintained.  Combined with
        # FlashAttention in model.py this typically gives 30-50% wall-clock
        # speedup on Ampere/Hopper GPUs with no accuracy regression.
        precision='bf16-mixed',
        max_steps=args.max_steps,
        enable_progress_bar=True,
        logger=CSVLogger("logs", name="TrainingSDPO", flush_logs_every_n_steps=1),
        log_every_n_steps=1,
        val_check_interval=300,
        callbacks=[checkpoint_callback, early_stop],
    )
    module, vocab_enc2atom, vocab_atom2enc = build_qm9_dataloader(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    module.setup()
    train_dset = load_qm9_smiles(dataset=module.train_dataset)
    ABSORB_IDX = len(vocab_enc2atom)
    checkpoint = torch.load('checkpoints/Pretrain_fast.ckpt')['state_dict']
    checkpoint = {k[7 + k[6:].index('.'):]: v for k, v in checkpoint.items() if 'model' in k}
    tabasco = TabascoV2(
        atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, pos_coord_dim=128,
        pair_rbf_centers=args.d_model // 2, dropout=0.1
    )
    tabasco.load_state_dict(checkpoint)
    tabasco = torch.compile(tabasco)
    rewarder = EnergyRewarder(batch_size=args.batch_size)
    args.vdW = vdW()
    model = LightningTabascoPipe(tabasco, rewarder, args, ABSORB_IDX, vocab_enc2atom, train_dset)
    trainer.fit(model=model, datamodule=module)


class LightningTabascoPipe(ValidationMixin, pl.LightningModule):
    def __init__(self, tabasco: TabascoV2, rewarder, args, ABSORB_IDX: int, vocab: Dict, train_dset):
        super().__init__()
        self.train_smiles  = set(train_dset)
        self.val_n_samples = 256   
        self.automatic_optimization = False

        self.args = args
        self.model = tabasco
        self.scheduler = DDIMScheduler.from_config(DDIM_config)
        self.ABSORB_IDX = ABSORB_IDX
        self.vocab = vocab

        lam = torch.tensor(self.args.lam, device=args.device)
        lam = lam.pow(torch.arange(self.args.sample_steps, dtype=torch.float, device=args.device))
        self.register_buffer('lam', lam.repeat(self.args.batch_size, 1))

        self.adv_normalizer = RunningStats()

        self.clip_range   = 0.2
        self.adv_clip_max = 1.0

        self.eta              = 1.0
        self.advantage_scale  = 0.3
        self.rewarder         = rewarder
        self.vocab            = self.vocab.to(args.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.args.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        return optimizer

    # ── helpers ──────────────────────────────────────────────────────────

    def _compute_log_probs_batch(self, coords, atoms, next_coords, next_atoms, timesteps_BT):
        """Forward pass under the *current* (updating) model."""
        coord_pred, types_pred = self.model(atoms, coords, timesteps_BT, batched=True)
        _, _, log_prob_coord = ddim_step_with_logprob(
            self.scheduler, coord_pred, timesteps_BT, coords,
            eta=self.eta, x_prev=next_coords, t_batched=True
        )
        _, _, log_prob_types = categorical_reverse_step(
            self.scheduler, types_pred, timesteps_BT, atoms,
            eta=self.eta, x_prev=next_atoms, t_batched=True
        )
        return log_prob_coord, log_prob_types   # [B, T]

    def _sdpo_loss(
        self,
        lp_coord1_new, lp_types1_new, lp_coord1_old, lp_types1_old,
        lp_coord2_new, lp_types2_new, lp_coord2_old, lp_types2_old,
        adv1, adv2, B, T
    ):
        log_ratio1 = (lp_coord1_new - lp_coord1_old) + (lp_types1_new - lp_types1_old)
        log_ratio2 = (lp_coord2_new - lp_coord2_old) + (lp_types2_new - lp_types2_old)

        log_ratio1_clipped = torch.clamp(log_ratio1, -self.clip_range, self.clip_range)
        log_ratio2_clipped = torch.clamp(log_ratio2, -self.clip_range, self.clip_range)

        log_diff         = log_ratio1         - log_ratio2
        log_diff_clipped = log_ratio1_clipped - log_ratio2_clipped

        log_weights = self.lam[:B, :T] / self.args.log_scale
        adv_diff    = self.advantage_scale * (adv1 - adv2)

        loss         = torch.square(log_weights * log_diff         - adv_diff)
        loss_clipped = torch.square(log_weights * log_diff_clipped - adv_diff)
        return torch.mean(torch.maximum(loss_clipped, loss)), log_diff, adv_diff

    # ── main training step ────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        """
        Three phases:
          1. ROLLOUT     — two stochastic trajectories, frozen model, no_grad.
          2. FREEZE      — compute log_prob_old once from the frozen model.
          3. INNER LOOP  — K gradient steps; model updates between each step,
                           so log_prob_new diverges from log_prob_old and
                           log_diff becomes nonzero and informative.
        """
        opt = self.optimizers()
        B   = self.args.batch_size

        # ── Phase 1: rollout ─────────────────────────────────────────────
        self.model.eval()
        x1     = torch.randn(B, 29, 3, device=self.device)
        types1 = torch.randn(B, 29, 6, device=self.device)
        x2     = torch.randn(B, 29, 3, device=self.device)
        types2 = torch.randn(B, 29, 6, device=self.device)

        with torch.no_grad():
            mols1, all_mols1, _, anchor_steps1, sim_first1, sim_anchor1, sim_last1 = \
                pipeline_with_logprob(
                    self.model, x1, types1,
                    num_inference_steps=self.args.sample_steps,
                    scheduler=self.scheduler, B=B,
                    device=self.device, eta=self.eta
                )
            mols2, all_mols2, _, anchor_steps2, sim_first2, sim_anchor2, sim_last2 = \
                pipeline_with_logprob(
                    self.model, x2, types2,
                    num_inference_steps=self.args.sample_steps,
                    scheduler=self.scheduler, B=B,
                    device=self.device, eta=self.eta
                )

            coords1 = torch.stack([m[0] for m in all_mols1], dim=1)  # [B, T+1, N, 3]
            atoms1  = torch.stack([m[1] for m in all_mols1], dim=1)
            coords2 = torch.stack([m[0] for m in all_mols2], dim=1)
            atoms2  = torch.stack([m[1] for m in all_mols2], dim=1)

            timesteps_1d = self.scheduler.timesteps                     # [T]
            T            = len(timesteps_1d)
            timesteps_BT = timesteps_1d.unsqueeze(0).expand(B, -1)     # [B, T]

            # ── Phase 2: frozen old log_probs ────────────────────────────
            # OPTIMIZATION: concatenate both trajectories along the batch dim
            # so a single forward pass replaces two separate calls.
            # This halves the number of frozen forward passes per training step.
            coords_cat = torch.cat([coords1, coords2], dim=0)   # [2B, T+1, N, 3]
            atoms_cat  = torch.cat([atoms1,  atoms2],  dim=0)   # [2B, T+1, N, A]
            lp_coord_old, lp_types_old = recompute_log_probs(
                self.model, self.scheduler, coords_cat, atoms_cat, eta=self.eta
            )
            lp_coord1_old, lp_coord2_old = lp_coord_old.chunk(2, dim=0)
            lp_types1_old, lp_types2_old = lp_types_old.chunk(2, dim=0)
            lp_coord1_old = lp_coord1_old.detach()
            lp_types1_old = lp_types1_old.detach()
            lp_coord2_old = lp_coord2_old.detach()
            lp_types2_old = lp_types2_old.detach()

            if self.global_step % 500 == 0 and self.args.debug:
                get_reward(mols1, self.rewarder, self.vocab, self.args.vdW, debug=True)

            # ── Rewards ──────────────────────────────────────────────────
            progress = min(self.global_step / 5000.0, 1.0)
            w_quality_curr = 0.5 + 1.5 * progress
            rewards1 = get_reward(mols1, self.rewarder, self.vocab, self.args.vdW, w_quality=w_quality_curr).T
            rewards2 = get_reward(mols2, self.rewarder, self.vocab, self.args.vdW, w_quality=w_quality_curr).T

            self.log('Rewards_1_start',  rewards1[:, 2].mean())
            self.log('Rewards_1_anchor', rewards1[:, 1].mean())
            self.log('Rewards_1_last',   rewards1[:, 0].mean())
            self.log('Rewards_2_start',  rewards2[:, 2].mean())
            self.log('Rewards_2_anchor', rewards2[:, 1].mean())
            self.log('Rewards_2_last',   rewards2[:, 0].mean())
            self.log('Reward0_mean',
                     (rewards1[:, 2].mean() + rewards2[:, 2].mean()) / 2, on_step=True)
            self.log('Reward_gap_start', (rewards1[:, 2] - rewards2[:, 2]).abs().mean())
            self.log('Reward_gap_last',  (rewards1[:, 0] - rewards2[:, 0]).abs().mean())

            # Interpolate + discount
            def interp_rewards(rw, sim_f, sim_a, sim_l, anchor_steps):
                denom = sim_f + sim_a + sim_l + 1e-8
                ri = (rw[:, 2:3].expand(-1, T) * sim_f
                      + rw[:, 1:2].expand(-1, T) * sim_a
                      + rw[:, 0:1].expand(-1, T) * sim_l) / denom
                idx = torch.arange(B, device=self.device)
                ri[:, 0]              = rw[:, 2]
                ri[:, -1]             = rw[:, 0]
                ri[idx, anchor_steps] = rw[:, 1]
                return ri

            ri1 = interp_rewards(rewards1, sim_first1, sim_anchor1, sim_last1, anchor_steps1)
            ri2 = interp_rewards(rewards2, sim_first2, sim_anchor2, sim_last2, anchor_steps2)

            ret1 = ri1.clone()
            ret2 = ri2.clone()
            for t in reversed(range(T - 1)):
                ret1[:, t] += self.args.gamma * ret1[:, t + 1]
                ret2[:, t] += self.args.gamma * ret2[:, t + 1]

            ret_all  = torch.cat([ret1, ret2], dim=0)
            ret_norm = (ret_all - ret_all.mean()) / (ret_all.std() + 1e-8)
            adv1, adv2 = ret_norm.chunk(2, dim=0)
            adv1 = torch.clamp(adv1, -self.adv_clip_max, self.adv_clip_max)
            adv2 = torch.clamp(adv2, -self.adv_clip_max, self.adv_clip_max)

        # Pre-slice trajectory arrays (shared across inner steps)
        coords_cur1  = coords1[:, :-1];  coords_next1 = coords1[:, 1:]
        atoms_cur1   = atoms1[:, :-1];   atoms_next1  = atoms1[:, 1:]
        coords_cur2  = coords2[:, :-1];  coords_next2 = coords2[:, 1:]
        atoms_cur2   = atoms2[:, :-1];   atoms_next2  = atoms2[:, 1:]
        timesteps_2BT = timesteps_BT.repeat(2, 1)

        # ── Phase 3: inner update loop ───────────────────────────────────
        self.model.train()
        log_diff_last = None
        adv_diff_last = None

        for _ in range(self.args.inner_epochs):
            opt.zero_grad()

            lp_coord_new, lp_types_new = self._compute_log_probs_batch(
                torch.cat([coords_cur1,  coords_cur2],  dim=0),
                torch.cat([atoms_cur1,   atoms_cur2],   dim=0),
                torch.cat([coords_next1, coords_next2], dim=0),
                torch.cat([atoms_next1,  atoms_next2],  dim=0),
                timesteps_2BT
            )
            lp_coord1_new, lp_coord2_new = lp_coord_new.chunk(2, dim=0)
            lp_types1_new, lp_types2_new = lp_types_new.chunk(2, dim=0)

            loss_sdpo, log_diff, adv_diff = self._sdpo_loss(
                lp_coord1_new, lp_types1_new, lp_coord1_old, lp_types1_old,
                lp_coord2_new, lp_types2_new, lp_coord2_old, lp_types2_old,
                adv1, adv2, B, T
            )

            self.manual_backward(loss_sdpo)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()

            log_diff_last = log_diff.detach()
            adv_diff_last = adv_diff.detach()

        # Logging from the final inner step
        with torch.no_grad():
            corr = torch.corrcoef(
                torch.stack([log_diff_last.flatten(), adv_diff_last.flatten()])
            )[0, 1]

        if self.args.debug:
            print(f"\n--- Step {self.global_step} ---")
            print(f"Adv Diff   mean={adv_diff_last.mean():.4f}  std={adv_diff_last.std():.4f}")
            print(f"Log Diff   mean={log_diff_last.mean():.4f}  std={log_diff_last.std():.4f}")
            print(f"Correlation: {corr:.4f}")
            print(f"Reward gap start: {(rewards1[:,2]-rewards2[:,2]).abs().mean():.4f}")
            print(f"Reward gap last:  {(rewards1[:,0]-rewards2[:,0]).abs().mean():.4f}")

        self.log('log_diff',       log_diff_last.mean())
        self.log('advantage_diff', adv_diff_last.mean())
        self.log('log_adv_corr',   corr)
        self.log('Training_loss',  loss_sdpo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       default='qm9')
    parser.add_argument('--data-root',     default='data/QM9')
    parser.add_argument('--max_steps',     type=int,   default=200_000)
    parser.add_argument('--inner_epochs',  type=int,   default=8)
    parser.add_argument('--batch-size',    type=int,   default=56)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--log_scale',     type=float, default=1.0)
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--lam',           type=float, default=0.99)
    parser.add_argument('--device',        type=str,   default='cuda')
    parser.add_argument('--T',             type=int,   default=1000)
    parser.add_argument('--d-model',       type=int,   default=384)
    parser.add_argument('--n-heads',       type=int,   default=8)
    parser.add_argument('--n-layers',      type=int,   default=6)
    parser.add_argument('--num-workers',   type=int,   default=6)
    parser.add_argument('--ckpt-dir',      type=str,   default='logs/TrainingSDPO/ckpts')
    parser.add_argument('--sample-steps',  type=int,   default=25)
    parser.add_argument('--sample-every',  type=int,   default=5)
    parser.add_argument('--log-steps',     type=int,   default=100)
    parser.add_argument('--save_after',    type=int,   default=5)
    parser.add_argument('--debug',         action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)