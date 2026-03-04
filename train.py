import os
import math
import torch
import argparse
from typing import Dict
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.model import TabascoV2
from data.dataloader import build_qm9_dataloader, load_qm9_smiles
from torch.utils.data import DataLoader, Dataset
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob, _compute_log_prob
from RL.reward import get_reward_batched, EnergyRewarder, RewardConfig, DEFAULT_PROP_SPECS
from RL.validation import ValidationMixin
from diffusers import DDIMScheduler
from config import DDIM_config
import faulthandler
faulthandler.enable()

class DummyDataset(Dataset):
    """Infinite dummy dataset that returns a single zero tensor per item."""
    def __init__(self, mode):
        match mode:
            case 'train': self.l = 10_000_000
            case 'val':   self.l = 1000

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        return torch.tensor(0)

class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(DummyDataset('train'), batch_size=1, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(DummyDataset('val'), batch_size=1, num_workers=0, shuffle=False)

def train(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath    = "logs/TrainingSDPO/ckpts/",
        save_top_k = 5,
        monitor    = "Reward0_mean",
        mode       = 'max',
        filename   = '{epoch}-{step}-{Reward0_mean}',
        save_last  = True,
        every_n_train_steps = 100,
    )
    early_stop = EarlyStopping(
        monitor   = 'val/stopping_score',
        mode      = 'max',
        patience  = 5,
        min_delta = 0.001,
    )
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        accelerator         = 'gpu',
        devices             = 1,
        precision           = '32',
        max_steps           = args.max_steps,
        enable_progress_bar = True,
        logger              = CSVLogger("logs", name="TrainingSDPO", flush_logs_every_n_steps=1),
        log_every_n_steps   = 1,
        val_check_interval  = 100,
        callbacks           = [checkpoint_callback, early_stop],
    )
    module, vocab_enc2atom, vocab_atom2enc, smiles = build_qm9_dataloader(
        root=args.data_root, batch_size=args.batch_size
    )

    ABSORB_IDX = len(vocab_enc2atom)
    checkpoint = torch.load('ckpt/Pretrain.ckpt')['state_dict']
    checkpoint = {k[7 + k[6:].index('.'):]: v for k, v in checkpoint.items() if 'model' in k}

    tabasco = TabascoV2(
        atom_vocab_size  = ABSORB_IDX,
        d_model          = args.d_model,
        n_heads          = args.n_heads,
        n_layers         = args.n_layers,
        pos_coord_dim    = 128,
        pair_rbf_centers = args.d_model // 2,
        dropout          = 0.1,
    )
    tabasco.load_state_dict(checkpoint)

    _TRAINABLE = ('crossAtomCoord.', 'crossCoordAtom.', 'coord_head.', 'type_head.')
    for name, param in tabasco.named_parameters():
        param.requires_grad_(any(name.startswith(k) for k in _TRAINABLE))

    tabasco = torch.compile(tabasco)

    rewarder = EnergyRewarder(batch_size=args.batch_size)
    model    = LightningTabascoPipe(tabasco, rewarder, args, ABSORB_IDX, vocab_enc2atom, smiles)
    trainer.fit(model=model, datamodule=DummyDataModule())


class LightningTabascoPipe(ValidationMixin, pl.LightningModule):
    def __init__(
        self,
        tabasco    : TabascoV2,
        rewarder,
        args,
        ABSORB_IDX : int,
        vocab      : Dict,
        train_dset,
    ):
        super().__init__()
        self.trajectories  = 64
        self.train_smiles  = set(train_dset)
        self.val_n_samples = 1000
        self.automatic_optimization = False

        self.args       = args
        self.model      = tabasco
        self.scheduler  = DDIMScheduler.from_config(DDIM_config)
        self.ABSORB_IDX = ABSORB_IDX
        self.vocab      = vocab

        # λ decay weights: lam[t] = lam^t, shape [1, T]
        lam = torch.tensor(self.args.lam)
        lam = lam.pow(torch.arange(self.args.sample_steps, dtype=torch.float))
        self.register_buffer('lam', lam.unsqueeze(0))   # [1, T] — broadcast in _sdpo_loss

        self.eta      = 1.0
        self.rewarder = rewarder
        self.rewarder.initialize_from_dataset(train_dset)
        self.vocab    = self.vocab.to(args.device)

        T_steps = self.args.sample_steps
        gw = self.args.gamma ** torch.arange(T_steps, dtype=torch.float, device=args.device)
        self.register_buffer('gamma_weights', gw)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.args.lr, weight_decay=0.01, betas=(0.9, 0.95),
        )

    # ── Log-prob recomputation ────────────────────────────────────────────────

    def _compute_log_probs_batch(
        self, coords, atoms, next_coords, next_atoms,
        timesteps_BT,
        means_coord, means_atoms,
        sigmas_coord, sigmas_atoms,
    ):
        """
        Recompute log-probs for stored transitions under the current (updated) model.
        Sigma is reused from the rollout — it only depends on the scheduler, not the model.
        """
        coord_pred, types_pred = self.model(atoms, coords, timesteps_BT, batched=True)

        _, _, _, x_prev_mean_coord, _ = ddim_step_with_logprob(
            self.scheduler, coord_pred, timesteps_BT, coords,
            eta=self.eta, x_prev=next_coords, t_batched=True,
        )
        _, _, _, x_prev_mean_types, _ = ddim_step_with_logprob(
            self.scheduler, types_pred, timesteps_BT, atoms,
            eta=self.eta, x_prev=next_atoms, t_batched=True,
        )

        lp_coord = _compute_log_prob(next_coords, x_prev_mean_coord, sigmas_coord, t_batched=True)
        lp_types = _compute_log_prob(next_atoms,  x_prev_mean_types, sigmas_atoms,  t_batched=True)

        return lp_coord, lp_types

    # ── SDPO loss with win-rate confidence weighting ──────────────────────────

    def _sdpo_loss(
        self,
        lp_coord_new, lp_types_new,   # [B, G, T]
        lp_coord_old, lp_types_old,   # [B, G, T]
        T,
        adv_diff,                     # [B, G, G, T]  rank-based pairwise target
        confidence,                   # [G, G]         |win_rate - 0.5| * 2, in [0, 1]
    ):
        log_ratio = (lp_coord_new - lp_coord_old) + (lp_types_new - lp_types_old)  # [B, G, T]

        # Pairwise log-ratio differences — never across batch elements
        log_diff = log_ratio.unsqueeze(2) - log_ratio.unsqueeze(1)   # [B, G, G, T]

        # λ time-decay: lam[0, :T] broadcasts over [B, G, G, T]
        lam_t = (self.lam[0, :T] * self.args.log_scale).reshape(1, 1, 1, T)
        weighted_log_diff = lam_t * log_diff                          # [B, G, G, T]

        # KL penalty against rollout policy — prevents mode collapse
        kl_penalty = (log_ratio.exp() - 1 - log_ratio).mean()

        # Confidence weight: high for clear winners/losers, near-zero for ties.
        # Shape [1, G, G, 1] broadcasts over [B, G, G, T].
        # Using confidence (not raw win_rate) avoids the symmetric cancellation
        # that occurs when win_rate[i,j] + win_rate[j,i] = 1.
        conf_w = confidence.unsqueeze(0).unsqueeze(-1)                # [1, G, G, 1]
        entropy_coord = -lp_coord_new.mean()
        entropy_types = -lp_types_new.mean()
        entropy_bonus = 0.01 * (entropy_coord + entropy_types)  # Encourage exploration
        
        loss = (conf_w * torch.square(weighted_log_diff - adv_diff)).mean() \
             + self.args.kl_beta * kl_penalty - entropy_bonus

        return loss, weighted_log_diff

    # ── Reward config ─────────────────────────────────────────────────────────

    def _make_reward_cfg(self) -> RewardConfig:
        return RewardConfig(active_props=DEFAULT_PROP_SPECS)

    # ── Main training step ────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        B   = self.args.batch_size
        G   = self.trajectories

        # ── Phase 1: rollout (no grad) ────────────────────────────────────────
        self.model.eval()
        steps, traj            = [], []
        all_lp                 = []
        all_anchor_steps       = []
        all_sim_first          = []
        all_sim_anchor         = []
        all_sim_last           = []
        all_means_list         = []
        all_sigmas_list        = []

        with torch.no_grad():
            for _ in range(G):
                x     = torch.randn(B, 29, 3,              device=self.device)
                types = torch.randn(B, 29, self.ABSORB_IDX, device=self.device)
                mols, all_mols, lp, anchor_steps, sim_first, sim_anchor, sim_last, means, sigmas = \
                    pipeline_with_logprob(
                        self.model, x, types,
                        num_inference_steps = self.args.sample_steps,
                        scheduler           = self.scheduler,
                        B                   = B,
                        device              = self.device,
                        eta                 = self.eta,
                    )
                steps.append(mols)
                traj.append(all_mols)
                all_lp.append(lp)
                all_anchor_steps.append(anchor_steps)
                all_sim_first.append(sim_first)
                all_sim_anchor.append(sim_anchor)
                all_sim_last.append(sim_last)
                all_means_list.append(means)
                all_sigmas_list.append(sigmas)

            all_lp           = torch.stack(all_lp)           # [G, B, T, 2]
            all_anchor_steps = torch.stack(all_anchor_steps) # [G, B]
            all_sim_first    = torch.stack(all_sim_first)    # [G, B, T]
            all_sim_anchor   = torch.stack(all_sim_anchor)   # [G, B, T]
            all_sim_last     = torch.stack(all_sim_last)     # [G, B, T]
            all_means_t      = torch.stack(all_means_list)   # [G, B, T, N, D]
            all_sigmas_t     = torch.stack(all_sigmas_list)  # [G, B, T, N, D]

            lps_coord_old = all_lp[:, :, :, 0].detach()     # [G, B, T]
            lps_types_old = all_lp[:, :, :, 1].detach()     # [G, B, T]
            T             = lps_coord_old.shape[2]

            # Build trajectory state tensors [G, T+1, B, N, D]
            coords_list, atoms_list = [], []
            for trajectory in traj:
                coords_list.append(torch.stack([c for c, _ in trajectory]))  # [T+1, B, N, 3]
                atoms_list.append(torch.stack([a for _, a in trajectory]))   # [T+1, B, N, A]
            coords = torch.stack(coords_list)   # [G, T+1, B, N, 3]
            atoms  = torch.stack(atoms_list)    # [G, T+1, B, N, A]

            # ── Rewards ───────────────────────────────────────────────────────
            cfg           = self._make_reward_cfg()
            rewards_stack = get_reward_batched(steps, self.rewarder, self.vocab, cfg=cfg)  # [G, 3, B]
            r0            = rewards_stack[:, 2].mean()
            self.log('Reward0_mean', r0)

            # ── Reward interpolation ──────────────────────────────────────────
            def interp_rewards(rw, sim_f, sim_a, sim_l, anc):
                sims = torch.stack([sim_f, sim_a, sim_l], dim=-1)
                # Use temperature-scaled softmax to preserve differences
                w = torch.softmax(sims / 0.5, dim=-1)  # Lower temperature = sharper weights
                rw_t = torch.stack([rw[0], rw[1], rw[2]], dim=-1)
                ri = (w * rw_t.unsqueeze(1)).sum(-1)
                # CRITICAL: Add noise to prevent exact ties
                ri = ri + torch.randn_like(ri) * 0.01
                idx = torch.arange(B, device=self.device)
                ri[idx, anc] = rw[1]
                ri[:, 0] = rw[0]
                ri[:, -1] = rw[2]
                return ri

            ri = list(map(interp_rewards,
                          rewards_stack, all_sim_first, all_sim_anchor,
                          all_sim_last, all_anchor_steps))
            ri = torch.stack(ri)   # [G, B, T]

            # ── Discounted returns ─────────────────────────────────────────────
            gw  = self.gamma_weights[:T]                              # [T]
            ret = torch.flip(
                torch.cumsum(torch.flip(ri * gw, [2]), dim=2), [2]
            ) / gw                                                    # [G, B, T]

            # ── Rank-based advantages in [-1, 1] ──────────────────────────────
            # argsort().argsort() gives competition ranks (0 = worst, G-1 = best)
            ranks = ret.argsort(dim=0).argsort(dim=0).float()        # [G, B, T]
            adv   = (ranks / (G - 1)) * 2.0 - 1.0                   # [G, B, T], in [-1, 1]

            # ── Win-rate confidence ────────────────────────────────────────────
            # win_rate[i, j] = fraction of (B, T) elements where ret[i] > ret[j]
            ret_i    = ret.unsqueeze(1)                               # [G, 1, B, T]
            ret_j    = ret.unsqueeze(0)                               # [1, G, B, T]
            win_rate = (ret_i > ret_j).float().mean(dim=(-2, -1))    # [G, G]

            # Confidence = distance from a tie: 0 for tied pairs, 1 for clear wins.
            # Symmetric: confidence[i,j] == confidence[j,i], avoiding cancellation.
            confidence = (win_rate - 0.5).abs() * 2.0
            confidence = 0.3 + 0.7 * confidence  # Range [0.3, 1.0]           # [G, G]

            # Pairwise advantage targets [B, G, G, T]
            adv_grouped  = adv.permute(1, 0, 2)                      # [B, G, T]
            adv_diff     = adv_grouped.unsqueeze(2) - adv_grouped.unsqueeze(1)  # [B, G, G, T]

        # ── Flatten trajectory × batch for batched model forward ─────────────
        coords = coords.permute(0, 2, 1, 3, 4)              # [G, B, T+1, N, 3]
        atoms  = atoms.permute(0, 2, 1, 3, 4)               # [G, B, T+1, N, A]
        coords_flat = coords.reshape(G * B, T + 1, -1, 3)   # [GB, T+1, N, 3]
        atoms_flat  = atoms.reshape(G * B, T + 1, -1, atoms.shape[-1])

        coords_cur  = coords_flat[:, :-1]   # [GB, T, N, 3]
        atoms_cur   = atoms_flat[:, :-1]
        coords_next = coords_flat[:, 1:]
        atoms_next  = atoms_flat[:, 1:]

        timesteps_1d  = self.scheduler.timesteps               # [T]
        timesteps_BT  = timesteps_1d.unsqueeze(0).expand(B, -1)
        timesteps_GBT = timesteps_BT.repeat(G, 1)             # [GB, T]

        # Means / sigmas: [G, B, T, N, D] → [GB, T, N, D]
        D_ms        = all_means_t.shape[-1]
        means_flat  = all_means_t.reshape(G * B, T, -1, D_ms)
        sigmas_flat = all_sigmas_t.reshape(G * B, T, -1, D_ms)
        means_coord  = means_flat[..., :3]
        means_atoms  = means_flat[..., 3:]
        sigmas_coord = sigmas_flat[..., :3]
        sigmas_atoms = sigmas_flat[..., 3:]

        # Old log-probs grouped as [B, G, T]
        lp_coord_old_g = lps_coord_old.permute(1, 0, 2)   # [B, G, T]
        lp_types_old_g = lps_types_old.permute(1, 0, 2)

        # ── Phase 2: inner update loop ────────────────────────────────────────
        self.model.train()
        weighted_log_diff_last = None

        for _ in range(self.args.inner_epochs):
            opt.zero_grad()

            lp_coord_new, lp_types_new = self._compute_log_probs_batch(
                coords_cur, atoms_cur, coords_next, atoms_next,
                timesteps_GBT,
                means_coord, means_atoms,
                sigmas_coord, sigmas_atoms,
            )   # both [GB, T]

            lp_coord_new_g = lp_coord_new.reshape(G, B, T).permute(1, 0, 2)  # [B, G, T]
            lp_types_new_g = lp_types_new.reshape(G, B, T).permute(1, 0, 2)

            loss_sdpo, weighted_log_diff_last = self._sdpo_loss(
                lp_coord_new_g, lp_types_new_g,
                lp_coord_old_g, lp_types_old_g,
                T, adv_diff, confidence,
            )

            self.manual_backward(loss_sdpo)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            opt.step()

        # ── Logging ───────────────────────────────────────────────────────────
        with torch.no_grad():
            log_ratio_total = (lp_coord_new_g - lp_coord_old_g) \
                            + (lp_types_new_g - lp_types_old_g)   # [B, G, T]

            approx_kl  = (log_ratio_total.exp() - 1 - log_ratio_total).mean()
            log_r_mean = log_ratio_total.mean()
            log_r_std  = log_ratio_total.std()

            adv_mean = adv.mean()
            adv_std  = adv.std()
            adv_max  = adv.abs().max()

            r_start  = rewards_stack[:, 0].mean()
            r_anchor = rewards_stack[:, 1].mean()
            r_last   = rewards_stack[:, 2].mean()
            r_std    = rewards_stack[:, 2].std()

            # Off-diagonal win-rate stats (diagonal is always 0.5)
            diag_mask = ~torch.eye(G, dtype=torch.bool, device=self.device)
            wr_off    = win_rate[diag_mask]
            wr_mean   = wr_off.mean()
            wr_std    = wr_off.std()
            conf_mean = confidence[diag_mask].mean()

            # Correlation between weighted log-diff and advantage diff
            diag_mask_2d = diag_mask
            wld_off = weighted_log_diff_last[:, diag_mask_2d, :]   # [B, G*(G-1), T]
            adv_off = adv_diff[:, diag_mask_2d, :]
            corr    = torch.corrcoef(
                torch.stack([wld_off.flatten(), adv_off.flatten()])
            )[0, 1]

            # Per-timestep correlation
            corr_per_t = []
            for t_i in range(T):
                w_t = weighted_log_diff_last[:, diag_mask_2d, t_i].flatten()
                a_t = adv_diff[:, diag_mask_2d, t_i].flatten()
                if w_t.numel() > 1:
                    corr_per_t.append(torch.corrcoef(torch.stack([w_t, a_t]))[0, 1].item())
            corr_early = sum(corr_per_t[:T//3])      / (T//3)       if T >= 3 else corr
            corr_mid   = sum(corr_per_t[T//3:2*T//3]) / (T//3)     if T >= 3 else corr
            corr_late  = sum(corr_per_t[2*T//3:])   / (T - 2*T//3) if T >= 3 else corr

            grad_norm = torch.stack([
                p.grad.norm() for p in self.model.parameters() if p.grad is not None
            ]).norm()

            entropy_coord = -lp_coord_new.mean()
            entropy_types = -lp_types_new.mean()
            trajectory_variance = ret.std(dim=0).mean()  # Reward variance across trajectories
            win_rate_entropy = -(win_rate * torch.log(win_rate + 1e-8)).mean()  # Win-rate entropy

        self.log_dict({
            "collapse/trajectory_variance": trajectory_variance,
            "collapse/win_rate_entropy": win_rate_entropy,
            "collapse/detection": (trajectory_variance < 0.1).float(),  # Alert when < 0.1
            "reward/start"          : r_start,
            "reward/anchor"         : r_anchor,
            "reward/last"           : r_last,
            "reward/last_std"       : r_std,
            "reward/progression"    : r_last - r_start,
            "loss/sdpo"             : loss_sdpo,
            "policy/approx_kl"      : approx_kl,
            "policy/log_ratio_mean" : log_r_mean,
            "policy/log_ratio_std"  : log_r_std,
            "advantage/mean"        : adv_mean,
            "advantage/std"         : adv_std,
            "advantage/abs_max"     : adv_max,
            "winrate/mean"          : wr_mean,
            "winrate/std"           : wr_std,
            "winrate/conf_mean"     : conf_mean,
            "alignment/corr"        : corr,
            "alignment/corr_early"  : corr_early,
            "alignment/corr_mid"    : corr_mid,
            "alignment/corr_late"   : corr_late,
            "grad/norm"             : grad_norm,
            "entropy/coord"         : entropy_coord,
            "entropy/types"         : entropy_types,
        })

        if self.args.debug:
            print(f"\n--- Step {self.global_step} ---")
            print(f"  reward      start={r_start:.3f}  anchor={r_anchor:.3f}  last={r_last:.3f}")
            print(f"  policy      KL={approx_kl:.4f}  log_r_std={log_r_std:.4f}")
            print(f"  advantage   mean={adv_mean:.4f}  std={adv_std:.4f}  max={adv_max:.4f}")
            print(f"  winrate     mean={wr_mean:.4f}  std={wr_std:.4f}  conf={conf_mean:.4f}")
            print(f"  alignment   corr={corr:.4f}  early={corr_early:.4f}  mid={corr_mid:.4f}  late={corr_late:.4f}")
            print(f"  grad_norm   {grad_norm:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      default='qm9')
    parser.add_argument('--data-root',    default='data/QM9')
    parser.add_argument('--max_steps',    type=int,   default=200_000)
    parser.add_argument('--inner_epochs', type=int,   default=1)
    parser.add_argument('--batch-size',   type=int,   default=12)
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--log_scale',    type=float, default=1.0,
                        help='Multiplier on lam_t to scale weighted_log_diff toward adv_diff range.')
    parser.add_argument('--kl_beta',      type=float, default=0.1,
                        help='KL penalty weight against rollout policy. Increase if collapse occurs.')
    parser.add_argument('--gamma',        type=float, default=0.99)
    parser.add_argument('--lam',          type=float, default=0.99)
    parser.add_argument('--device',       type=str,   default='cuda')
    parser.add_argument('--d-model',      type=int,   default=384)
    parser.add_argument('--n-heads',      type=int,   default=8)
    parser.add_argument('--n-layers',     type=int,   default=6)
    parser.add_argument('--ckpt-dir',     type=str,   default='logs/TrainingSDPO/ckpts')
    parser.add_argument('--sample-steps', type=int,   default=25)
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)