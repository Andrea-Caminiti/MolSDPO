import os
import torch
import argparse
from typing import Dict, Tuple
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.model import TabascoV2
from data.dataloader import build_qm9_dataloader
from torch.utils.data import DataLoader, Dataset
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob
from RL.reward import (get_reward_batched, MoleculeRewarder, AdaptiveWeighter,
                       NoveltyBuffer, compute_diversity, compute_novelty_batched, reward_log_dict)
from RL.validation import ValidationMixin
from diffusers import DDIMScheduler
from config import DDIM_config
import faulthandler

faulthandler.enable()

class _RLDummyDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(0)


class RLDataModule(pl.LightningDataModule):
    TRAIN_LEN = 10_000_000
    VAL_LEN   = 1

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_RLDummyDataset(self.TRAIN_LEN), batch_size=1, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(_RLDummyDataset(self.VAL_LEN), batch_size=1, num_workers=0)


class LightningTabascoPipe(ValidationMixin, pl.LightningModule):

    N_TRAJECTORIES: int = 32 #G: number of independent rollouts per step

    def __init__(
        self,
        tabasco      : TabascoV2,
        rewarder     : MoleculeRewarder,
        weighter     : AdaptiveWeighter,
        args         : argparse.Namespace,
        absorb_idx   : int,
        vocab        : Dict,
        train_smiles,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.args         = args
        self.model        = tabasco
        self.rewarder     = rewarder
        self.weighter     = weighter
        self.ABSORB_IDX   = absorb_idx
        self.vocab        = vocab       
        self.train_smiles = set(train_smiles)
        self.val_n_samples = 1_000
        self.eta           = 1.0

        self.scheduler = DDIMScheduler.from_config(DDIM_config)

        #Novelty buffer tracks recently generated SMILES so the reward
        #function can penalise the model for repeatedly generating the same
        #molecules.  Capacity default 5000
        self.novelty_buffer = NoveltyBuffer(capacity=args.novelty_buf_size)

        T     = args.sample_steps
        lam_t = args.lam ** torch.arange(T, dtype=torch.float)
        self.register_buffer('lam_t', lam_t) #[T]

        gamma_w = args.gamma ** torch.arange(T, dtype=torch.float)
        self.register_buffer('gamma_weights', gamma_w) #[T]


    def setup(self, stage: str) -> None:
        self.vocab = self.vocab.to(self.device)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable, lr=self.args.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )

    @torch.no_grad()
    def _rollout(self, B: int) -> dict:
        """
        Sample G independent trajectories under the current (frozen) policy.

        Returns a dict of stacked tensors:
          coords          [G, T+1, B, N, 3]
          atoms           [G, T+1, B, N, A]
          lp_coord_old    [G, B, T]
          lp_types_old    [G, B, T]
          means           [G, B, T, N, D]
          sigmas          [G, B, T, N, D]
          anchor_steps    [G, B]
          x0_pred_first   [G, B, N, D]
          x0_pred_anchor  [G, B, N, D]
          x0_pred_last    [G, B, N, D]
        """
        self.model.eval()
        G  = self.N_TRAJECTORIES
        GB = G * B

        x     = torch.randn(GB, 29, 3,               device=self.device)
        types = torch.randn(GB, 29, self.ABSORB_IDX, device=self.device)

        (mols_raw, traj_states, lp,
         anchor_steps, x0_pred_first, x0_pred_anchor, x0_pred_last,
         means, sigmas) = pipeline_with_logprob(
            self.model, x, types,
            num_inference_steps = self.args.sample_steps,
            scheduler           = self.scheduler,
            B                   = GB,
            device              = self.device,
            eta                 = self.eta,
        )

        T = lp.shape[1]
        N = x0_pred_first.shape[1]
        A = types.shape[-1]

        def _r(t: torch.Tensor) -> torch.Tensor:
            """Reshape leading G*B dim to (G, B, *rest)."""
            return t.reshape(G, B, *t.shape[1:])

        #Trajectory states: list of T+1 (coord [G*B,N,3], types [G*B,N,A]) pairs
        coords_traj = torch.stack([c for c, _ in traj_states])  #[T+1, G*B, N, 3]
        atoms_traj  = torch.stack([a for _, a in traj_states])  #[T+1, G*B, N, A]
        #Split G*B → G, B then move T axis to position 1  →  [G, T+1, B, N, 3/A]
        coords_stack = coords_traj.reshape(T + 1, G, B, N, 3).permute(1, 0, 2, 3, 4)
        atoms_stack  = atoms_traj.reshape(T + 1, G, B, N, A).permute(1, 0, 2, 3, 4)

        return dict(
            final_mols     = mols_raw,                       
            coords         = coords_stack, #[G, T+1, B, N, 3]
            atoms          = atoms_stack, #[G, T+1, B, N, A]
            lp_coord_old   = _r(lp)[..., 0].detach(), #[G, B, T]
            lp_types_old   = _r(lp)[..., 1].detach(), #[G, B, T]
            means          = _r(means), #[G, B, T, N, D]
            sigmas         = _r(sigmas), #[G, B, T, N, D]
            anchor_steps   = _r(anchor_steps), #[G, B]
            x0_pred_first  = _r(x0_pred_first), #[G, B, N, D]
            x0_pred_anchor = _r(x0_pred_anchor), #[G, B, N, D]
            x0_pred_last   = _r(x0_pred_last), #[G, B, N, D]
        )

    @torch.no_grad()
    def _compute_advantages(self, rollout: dict, T: int) -> dict:
        """
        Compute dense per-timestep advantages using x̂₀_pred checkpoints and
        piecewise-linear interpolation in α̅_t space.

        Interpolation scheme
        --------------------
        For each molecule b with anchor step a_b, the T timesteps are split
        into two segments:

          Segment 1  [0,   a_b] : linearly blend r_first  → r_anchor  in α̅ space
          Segment 2  [a_b, T-1] : linearly blend r_anchor → r_final   in α̅ space

        """
        G = self.N_TRAJECTORIES
        B = rollout['x0_pred_first'].shape[1]
        COORD_DIM = 3

       
        x0_first  = rollout['x0_pred_first'] #[G, B, N, D]
        x0_anchor = rollout['x0_pred_anchor'] #[G, B, N, D]
        x0_last   = rollout['x0_pred_last'] #[G, B, N, D]

        def split(x):
            return x[..., :COORD_DIM], x[..., COORD_DIM:]

        
        r_first,  _ = get_reward_batched(self.rewarder, self.weighter, *split(x0_first))
        r_anchor, _ = get_reward_batched(self.rewarder, self.weighter, *split(x0_anchor))
        r_final,  sub_final = get_reward_batched(self.rewarder, self.weighter, *split(x0_last))

        #Diversity bonus  encourages G trajectories to be structurally distinct
        #For each batch position b, how different is trajectory g's predicted
        #clean molecule from the other G-1 trajectories?  

        dw = self.args.diversity_weight
        div_first = compute_diversity(x0_first[..., COORD_DIM:]) #[G, B]
        div_anchor = compute_diversity(x0_anchor[..., COORD_DIM:])
        div_last = compute_diversity(x0_last[..., COORD_DIM:])

        #Novelty bonus penalises repeated generation of the same molecules
        #Score 1.0 if this molecule has not appeared in the last
        #`novelty_buf_size` unique SMILES; 0.0 otherwise.
        #Applied only at x̂₀_last (novelty is an end-state property).
        nw = self.args.novelty_weight
        nov_last = compute_novelty_batched(
            x0_last[..., :COORD_DIM], x0_last[..., COORD_DIM:],
            self.rewarder.vocab, self.novelty_buffer,
        )   #[G, B]

        r_first_aug  = r_first  + dw * div_first
        r_anchor_aug = r_anchor + dw * div_anchor
        r_final_aug  = r_final  + dw * div_last + nw * nov_last

        rewards_stack = torch.stack([r_first, r_anchor, r_final], dim=1)   #[G, 3, B]
        rewards_aug   = torch.stack([r_first_aug, r_anchor_aug, r_final_aug], dim=1)


        ab = self.scheduler.alphas_cumprod[
            self.scheduler.timesteps[:T].cpu()
        ].to(self.device)   

        def interp_rewards(
            rw        : torch.Tensor, #[3, B]  stacked (r_first, r_anchor, r_final)
            anc_steps : torch.Tensor, #[B]     anchor step index per molecule
        ) -> torch.Tensor: #[B, T]
            r_f = rw[0] #[B]
            r_a = rw[1] #[B]
            r_l = rw[2] #[B]

            ab_anc = ab[anc_steps].unsqueeze(-1) #[B, 1]
            ab_0 = ab[0] 
            ab_T = ab[-1] 
            ab_t = ab.unsqueeze(0) #[1, T]

            denom1 = (ab_anc - ab_0).clamp(min=1e-6) #[B, 1]
            w1 = ((ab_t - ab_0) / denom1).clamp(0.0, 1.0) #[B, T]
            seg1 = r_f.unsqueeze(-1) * (1 - w1) + r_a.unsqueeze(-1) * w1  #[B, T]

            denom2 = (ab_T - ab_anc).clamp(min=1e-6) #[B, 1]
            w2 = ((ab_t - ab_anc) / denom2).clamp(0.0, 1.0) #[B, T]
            seg2 = r_a.unsqueeze(-1) * (1 - w2) + r_l.unsqueeze(-1) * w2  #[B, T]

            t_idx = torch.arange(T, device=self.device).unsqueeze(0)  #[1, T]
            mask  = t_idx <= anc_steps.unsqueeze(-1) #[B, T]
            ri = torch.where(mask, seg1, seg2) #[B, T]

            idx = torch.arange(B, device=self.device)
            ri[idx, 0] = r_f
            ri[idx, anc_steps] = r_a
            ri[idx, -1] = r_l
            return ri

        ri_list = [
            interp_rewards(rewards_aug[g], rollout['anchor_steps'][g])
            for g in range(G)
        ]
        ri = torch.stack(ri_list) #[G, B, T]

        gw = self.gamma_weights[:T]
        ret = torch.flip(
            torch.cumsum(torch.flip(ri * gw, [2]), dim=2), [2]
        ) / gw #[G, B, T]

        ranks = ret.argsort(dim=0).argsort(dim=0).float()
        adv   = (ranks / max(G - 1, 1)) * 2.0 - 1.0   #[G, B, T]

        ret_f = ret[:, :, -1]  #[G, B]
        ret_f_i = ret_f.unsqueeze(1) #[G, 1, B]
        ret_f_j = ret_f.unsqueeze(0)  #[1, G, B]
        tau = ret_f.std(dim=0, keepdim=True).clamp(min=1e-3)   #[1, B]
        win_rate = torch.sigmoid(
            (ret_f_i - ret_f_j) / tau.unsqueeze(0)
        ).mean(dim=-1) #[G, G]
        confidence = (win_rate - 0.5).abs() * 2.0  #[G, G]

        adv_g    = adv.permute(1, 0, 2) #[B, G, T]
        adv_diff = adv_g.unsqueeze(2) - adv_g.unsqueeze(1) #[B, G, G, T]

        return dict(
            rewards_stack = rewards_stack,  #[G, 3, B]  
            sub_final = sub_final,  #[G, B, 4]  
            div_last = div_last, #[G, B]     
            nov_last = nov_last, #[G, B]     
            ret = ret, #[G, B, T]
            adv = adv, #[G, B, T]
            adv_diff = adv_diff, #[B, G, G, T]
            win_rate = win_rate, #[G, G]
            confidence = confidence, #[G, G]
        )


    def _recompute_log_probs(
        self,
        coords_cur  : torch.Tensor, #[GB, T, N, 3]
        atoms_cur   : torch.Tensor,
        coords_next : torch.Tensor,
        atoms_next  : torch.Tensor,
        timesteps   : torch.Tensor, #[GB, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass + DDIM step to get log-probs under the *current* policy.
        Sigmas come from the scheduler and are independent of model weights,
        so they are reused from the rollout without recomputation.
        """
        coord_pred, types_pred = self.model(atoms_cur, coords_cur, timesteps, batched=True)

        _, _, lp_coord, _, _ = ddim_step_with_logprob(
            self.scheduler, coord_pred, timesteps, coords_cur,
            eta=self.eta, x_prev=coords_next, t_batched=True,
        )
        _, _, lp_types, _, _ = ddim_step_with_logprob(
            self.scheduler, types_pred, timesteps, atoms_cur,
            eta=self.eta, x_prev=atoms_next, t_batched=True,
        )
        return lp_coord, lp_types #both [GB, T]

    def _sdpo_loss(
        self,
        lp_coord_new : torch.Tensor, #[B, G, T]
        lp_types_new : torch.Tensor,
        lp_coord_old : torch.Tensor,
        lp_types_old : torch.Tensor,
        adv_diff     : torch.Tensor, #[B, G, G, T]
        confidence   : torch.Tensor, #[G, G]
        ret          : torch.Tensor, #[G, B, T]
        T            : int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_ratio = (lp_coord_new - lp_coord_old) + \
                    (lp_types_new - lp_types_old) #[B, G, T]

        log_diff = log_ratio.unsqueeze(2) - log_ratio.unsqueeze(1) #[B, G, G, T]

        lam_t = (self.lam_t[:T] * self.args.log_scale).reshape(1, 1, 1, T)
        weighted_log_diff = lam_t * log_diff #[B, G, G, T]

        ret_bg = ret.permute(1, 0, 2) #[B, G, T]
        ret_std = ret_bg.std(dim=1, keepdim=True).clamp(min=1e-6)  #[B, 1, T]
        ret_i = ret_bg.unsqueeze(2) #[B, G, 1, T]
        ret_j = ret_bg.unsqueeze(1) #[B, 1, G, T]
        ret_gap = ((ret_i - ret_j).abs() / ret_std.unsqueeze(2)
                     ).clamp(0.0, 3.0) #[B, G, G, T]
        target = adv_diff * ret_gap #[B, G, G, T]

        log_ratio_clipped = log_ratio.clamp(-20.0, 20.0)
        kl_penalty = (log_ratio_clipped.exp() - 1 - log_ratio).mean()

        entropy_bonus = 0.01 * (-lp_coord_new.mean() + -lp_types_new.mean())

        conf_w = confidence.unsqueeze(0).unsqueeze(-1) #[1, G, G, 1]
        loss = (conf_w * torch.square(weighted_log_diff - target)).mean() \
             + self.args.kl_beta * kl_penalty \
             - entropy_bonus

        return loss, weighted_log_diff

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        B   = self.args.batch_size
        G   = self.N_TRAJECTORIES

        rollout = self._rollout(B)
        T       = rollout['lp_coord_old'].shape[2]

        adv_data = self._compute_advantages(rollout, T)

        self.log('Reward0_mean', adv_data['rewards_stack'][:, 2].mean())

        #Flatten [G, B, T+1, N, D] to [GB, T+1, N, D] for batched forward 
        #Permute from [G, T+1, B, N, D] to [G, B, T+1, N, D] first.
        coords = rollout['coords'].permute(0, 2, 1, 3, 4) #[G, B, T+1, N, 3]
        atoms = rollout['atoms'].permute(0, 2, 1, 3, 4) #[G, B, T+1, N, A]

        coords_flat = coords.reshape(G * B, T + 1, -1, 3)
        atoms_flat = atoms.reshape(G * B, T + 1, -1, atoms.shape[-1])

        coords_cur = coords_flat[:, :-1] #[GB, T, N, 3]
        atoms_cur = atoms_flat[:, :-1]
        coords_next = coords_flat[:, 1:]
        atoms_next = atoms_flat[:, 1:]

        timesteps_GBT = self.scheduler.timesteps \
                            .unsqueeze(0) \
                            .expand(coords_flat.shape[0], -1) #[GB, T]

        lp_coord_old_g = rollout['lp_coord_old'].permute(1, 0, 2)
        lp_types_old_g = rollout['lp_types_old'].permute(1, 0, 2)

        self.model.train()
        weighted_log_diff_last = None

        for _ in range(self.args.inner_epochs):
            opt.zero_grad()

            lp_coord_new, lp_types_new = self._recompute_log_probs(
                coords_cur, atoms_cur, coords_next, atoms_next, timesteps_GBT,
            ) #[GB, T]

            lp_coord_new_g = lp_coord_new.reshape(G, B, T).permute(1, 0, 2)
            lp_types_new_g = lp_types_new.reshape(G, B, T).permute(1, 0, 2)

            loss, weighted_log_diff_last = self._sdpo_loss(
                lp_coord_new_g, lp_types_new_g,
                lp_coord_old_g, lp_types_old_g,
                adv_data['adv_diff'],
                adv_data['confidence'],
                adv_data['ret'],
                T,
            )

            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            opt.step()

        self._log_metrics(
            loss, lp_coord_new_g, lp_types_new_g,
            lp_coord_old_g, lp_types_old_g,
            weighted_log_diff_last,
            adv_data, G, T,
        )

    @torch.no_grad()
    def _log_metrics(
        self,
        loss_sdpo    : torch.Tensor,
        lp_coord_new : torch.Tensor,  #[B, G, T]
        lp_types_new : torch.Tensor,
        lp_coord_old : torch.Tensor,
        lp_types_old : torch.Tensor,
        weighted_log_diff : torch.Tensor,#[B, G, G, T]
        adv_data : dict,
        G : int,
        T : int,
    ) -> None:
        rewards_stack = adv_data['rewards_stack']
        adv = adv_data['adv']
        adv_diff = adv_data['adv_diff']
        win_rate = adv_data['win_rate']
        confidence = adv_data['confidence']

        log_ratio = (lp_coord_new - lp_coord_old) + (lp_types_new - lp_types_old)
        approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()

        r_start = rewards_stack[:, 0].mean()
        r_anchor = rewards_stack[:, 1].mean()
        r_last = rewards_stack[:, 2].mean()
        r_std = rewards_stack[:, 2].std()

        diag_mask = ~torch.eye(G, dtype=torch.bool, device=self.device)
        wr_off = win_rate[diag_mask]
        conf_mean = confidence[diag_mask].mean()

        wld_off = weighted_log_diff[:, diag_mask, :]   #[B, G*(G-1), T]
        adv_off = adv_diff[:, diag_mask, :]
        corr = torch.corrcoef(
            torch.stack([wld_off.flatten(), adv_off.flatten()])
        )[0, 1]

        third = max(T // 3, 1)
        corr_per_t = []
        for t_i in range(T):
            w_t = wld_off[:, :, t_i].flatten()
            a_t = adv_off[:, :, t_i].flatten()
            if w_t.numel() > 1:
                corr_per_t.append(
                    torch.corrcoef(torch.stack([w_t, a_t]))[0, 1].item()
                )
        corr_early = sum(corr_per_t[:third])              / third
        corr_mid   = sum(corr_per_t[third : 2 * third])   / third
        corr_late  = sum(corr_per_t[2 * third:])          / max(T - 2 * third, 1)

        grad_norm = torch.stack([
            p.grad.norm() for p in self.model.parameters() if p.grad is not None
        ]).norm()

        trajectory_variance = adv_data['ret'].std(dim=0).mean()
        win_rate_entropy    = -(win_rate * torch.log(win_rate + 1e-8)).mean()

        self.log_dict({
            "collapse/trajectory_variance" : trajectory_variance,
            "collapse/win_rate_entropy"    : win_rate_entropy,
            "collapse/detection"           : (trajectory_variance < 0.1).float(),
            "reward/start"                 : r_start,
            "reward/anchor"                : r_anchor,
            "reward/last"                  : r_last,
            "reward/last_std"              : r_std,
            "reward/progression"           : r_last - r_start,
            "reward/diversity_mean"        : adv_data['div_last'].mean(),
            "reward/novelty_mean"          : adv_data['nov_last'].mean(),
            "reward/novelty_rate"          : (adv_data['nov_last'] > 0).float().mean(),
            "loss/sdpo"                    : loss_sdpo,
            "policy/approx_kl"             : approx_kl,
            "policy/log_ratio_mean"        : weighted_log_diff.mean(),
            "policy/log_ratio_std"         : weighted_log_diff.std(),
            "advantage/mean"               : adv.mean(),
            "advantage/std"                : adv.std(),
            "advantage/abs_max"            : adv.abs().max(),
            "winrate/mean"                 : wr_off.mean(),
            "winrate/std"                  : wr_off.std(),
            "winrate/conf_mean"            : conf_mean,
            "alignment/corr"               : corr,
            "alignment/corr_early"         : corr_early,
            "alignment/corr_mid"           : corr_mid,
            "alignment/corr_late"          : corr_late,
            "grad/norm"                    : grad_norm,
            "entropy/coord"                : -lp_coord_new.mean(),
            "entropy/types"                : -lp_types_new.mean(),
            **reward_log_dict(adv_data['sub_final'], self.weighter),
        })

        if self.args.debug:
            print(f"\n--- Step {self.global_step} ---")
            print(f"  reward    start={r_start:.3f}  anchor={r_anchor:.3f}  last={r_last:.3f}")
            print(f"  policy    KL={approx_kl:.4f}")
            print(f"  advantage mean={adv.mean():.4f}  std={adv.std():.4f}")
            print(f"  winrate   mean={wr_off.mean():.4f}  conf={conf_mean:.4f}")
            print(f"  alignment corr={corr:.4f}  early={corr_early:.4f}  "
                  f"mid={corr_mid:.4f}  late={corr_late:.4f}")
            print(f"  grad_norm {grad_norm:.4f}")


def train(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision('medium')

    checkpoint_cb = ModelCheckpoint(
        dirpath             = args.ckpt_dir,
        save_top_k          = 5,
        monitor             = "Reward0_mean",
        mode                = 'max',
        filename            = '{epoch}-{step}-{Reward0_mean:.4f}',
        save_last           = True,
        every_n_train_steps = 100,
    )
    early_stop_cb = EarlyStopping(
        monitor   = 'stopping_score',
        mode      = 'max',
        patience  = 5,
        min_delta = 0.001,
    )

    trainer = pl.Trainer(
        accelerator         = 'gpu',
        devices             = 1,
        precision           = "32",
        max_steps           = args.max_steps,
        enable_progress_bar = True,
        logger              = CSVLogger("logs", name="TrainingSDPO", flush_logs_every_n_steps=1),
        log_every_n_steps   = 1,
        val_check_interval  = 100,
        callbacks           = [checkpoint_cb, early_stop_cb],
    )

    module, vocab_enc2atom, vocab_atom2enc, smiles = build_qm9_dataloader(
        root=args.data_root, batch_size=args.batch_size
    )
    ABSORB_IDX = len(vocab_enc2atom)

    raw_ckpt   = torch.load('ckpt/Pretrain.ckpt')['state_dict']
    state_dict = {
        k[len('model._orig_mod.'):]: v
        for k, v in raw_ckpt.items()
        if k.startswith('model._orig_mod.')
    }

    tabasco = TabascoV2(
        atom_vocab_size  = ABSORB_IDX,
        d_model          = args.d_model,
        n_heads          = args.n_heads,
        n_layers         = args.n_layers,
        pos_coord_dim    = 128,
        pair_rbf_centers = args.d_model // 2,
        dropout          = 0.1,
    )
    tabasco.load_state_dict(state_dict)

    _TRAINABLE = ('crossCoordAtom.', 'crossAtomCoord.', 'coord_head.', 'type_head.')
    for name, param in tabasco.named_parameters():
        param.requires_grad_(any(name.startswith(k) for k in _TRAINABLE))

    rewarder = MoleculeRewarder(vocab_enc2atom)
    weighter = AdaptiveWeighter()
    model = LightningTabascoPipe(tabasco, rewarder, weighter, args, ABSORB_IDX, vocab_enc2atom, smiles)

    model.model = torch.compile(model.model)

    trainer.fit(model=model, datamodule=RLDataModule())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      default='qm9')
    parser.add_argument('--data-root',    default='data/QM9')
    parser.add_argument('--max_steps',    type=int,   default=200_000)
    parser.add_argument('--inner_epochs', type=int,   default=1)
    parser.add_argument('--batch-size',   type=int,   default=8)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--log_scale',    type=float, default=4.0,
                        help='Scales weighted_log_diff to match adv_diff range.')
    parser.add_argument('--kl_beta',      type=float, default=0.05,
                        help='KL penalty weight. Increase to 0.05–0.1 if collapse occurs.')
    parser.add_argument('--gamma',        type=float, default=0.99)
    parser.add_argument('--lam',          type=float, default=0.95)
    parser.add_argument('--device',       type=str,   default='cuda')
    parser.add_argument('--d-model',      type=int,   default=384)
    parser.add_argument('--n-heads',      type=int,   default=8)
    parser.add_argument('--n-layers',     type=int,   default=6)
    parser.add_argument('--ckpt-dir',     type=str,   default='logs/TrainingSDPO/ckpts')
    parser.add_argument('--sample-steps', type=int,   default=25)
    parser.add_argument('--diversity-weight', type=float, default=0.3,
                        help='Weight for intra-batch diversity bonus. '
                             '0 = disabled. Start at 0.3; raise to 0.5 if '
                             'uniqueness stays below 0.1 after 200 steps.')
    parser.add_argument('--novelty-weight',   type=float, default=0.2,
                        help='Weight for novelty buffer bonus (1.0 = new molecule). '
                             '0 = disabled. Raise to 0.3 if reward/novelty_rate '
                             'drops below 0.3 at steady state.')
    parser.add_argument('--novelty-buf-size', type=int,   default=5_000,
                        help='Unique SMILES retained in the novelty buffer. '
                             'Covers ~13 rollout steps at G=32, B=12 before eviction.')
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)