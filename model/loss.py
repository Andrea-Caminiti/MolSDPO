import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from model.util import q_sample_positions


def p_losses_joint_absorb_improved_efficient(
    model: torch.nn.Module,
    coords0: torch.Tensor,
    types0: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    device: torch.device,
    lambda_type: float = 1.0,
    lambda_dist: float = 0.01,
    lambda_com: float = 0.01,
) -> Tuple[torch.Tensor, Dict]:
    """
    DDPM epsilon-prediction loss with padding-masked geometry guidance.

    Loss components
    ---------------
    coord_loss  – MSE on coordinate noise, real atoms only
    type_loss   – MSE on type noise, real atoms only
    dist_loss   – MSE on pairwise distances of predicted x0, real pairs only
    com_loss    – Centre-of-mass preservation, real atoms only

    geom_loss (KL on distance histograms) has been removed: zeroing padding
    coordinates before cdist still leaves N*N - N_real*N_real spurious pairs
    in the histogram (padding-padding at d=0, padding-real at arbitrary d),
    making the KL target inconsistent across molecules of different sizes and
    causing the loss to spike unpredictably.  dist_loss captures the same
    pairwise geometry signal without this complication.
    """
    B, N, _ = coords0.shape

    # ── Padding mask ─────────────────────────────────────────────────────────
    # Real atoms have a one-hot encoding summing to 1; padded slots are
    # all-zero.  All losses are restricted to real atoms / real atom pairs.
    real_mask = types0.sum(dim=-1) > 0.5          # [B, N]  bool

    # ── Noise schedule ──────────────────────────────────────────────────────
    t_idx = t.squeeze().cpu() if t.ndim > 1 else t.cpu()
    alpha_prod_t         = scheduler.alphas_cumprod[t_idx].view(B, 1, 1).to(device)
    sqrt_alpha           = alpha_prod_t.sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt()

    # ── Forward diffusion with standard N(0,1) noise ─────────────────────────
    coord_noise = torch.randn_like(coords0)
    type_noise  = torch.randn_like(types0)
    x_t      = sqrt_alpha * coords0 + sqrt_one_minus_alpha * coord_noise
    types_t  = sqrt_alpha * types0  + sqrt_one_minus_alpha * type_noise

    # ── Single forward pass ──────────────────────────────────────────────────
    coord_pred, type_pred = model(types_t, x_t, t_idx.to(device))

    # ── Denoising losses – real atoms only ───────────────────────────────────
    coord_mask  = real_mask.unsqueeze(-1).expand_as(coord_noise)   # [B, N, 3]
    type_mask   = real_mask.unsqueeze(-1).expand_as(type_noise)    # [B, N, V]
    n_real      = real_mask.sum(dim=1).float().clamp(min=1)        # [B]

    coord_loss = ((coord_pred - coord_noise) ** 2 * coord_mask).sum(dim=(1, 2)) / (n_real * 3)
    type_loss  = ((type_pred  - type_noise)  ** 2 * type_mask ).sum(dim=(1, 2)) / (n_real * types0.shape[-1])

    # ── Reconstruct x0 ──────────────────────────────────────────────────────
    x0_pred = (x_t - sqrt_one_minus_alpha * coord_pred) / (sqrt_alpha + 1e-8)

    # ── Distance loss – real atom pairs only ─────────────────────────────────
    # pair_mask[b,i,j] = True iff both atom i and j are real in sample b.
    # Combined with upper-triangle to avoid double-counting.
    pair_mask = (real_mask.unsqueeze(2) & real_mask.unsqueeze(1))  # [B, N, N]
    triu_base = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    triu_mask = pair_mask & triu_base                               # [B, N, N]

    true_dists = torch.cdist(coords0, coords0)
    pred_dists = torch.cdist(x0_pred, x0_pred)
    n_pairs    = triu_mask.sum(dim=(1, 2)).float().clamp(min=1)    # [B]
    dist_loss  = (((pred_dists - true_dists) ** 2) * triu_mask).sum(dim=(1, 2)) / n_pairs

    # ── Centre-of-mass loss – real atoms only ────────────────────────────────
    n_real_3d = n_real.unsqueeze(-1)                               # [B, 1]
    com_true  = (coords0 * real_mask.unsqueeze(-1)).sum(dim=1) / n_real_3d   # [B, 3]
    com_pred  = (x0_pred * real_mask.unsqueeze(-1)).sum(dim=1) / n_real_3d
    com_loss  = ((com_pred - com_true) ** 2).mean(dim=-1)          # [B]

    # ── Adaptive weighting by timestep ──────────────────────────────────────
    # High t → focus on denoising; low t → focus on geometry quality.
    t_norm    = t_idx.float().to(device) / (scheduler.config.num_train_timesteps - 1)
    denoise_w = 0.3 + 0.7 * t_norm     # 0.3 at t=0  → 1.0 at t=T
    geom_w    = 1.0 - 0.9 * t_norm     # 1.0 at t=0  → 0.1 at t=T

    # ── Total loss ───────────────────────────────────────────────────────────
    loss = (
        denoise_w * coord_loss
        + lambda_type * type_loss
        + lambda_dist * geom_w * dist_loss
        + lambda_com  * com_loss
    )

    metrics = {
        'coord_loss': coord_loss.mean().item(),
        'type_loss':  type_loss.mean().item(),
        'dist_loss':  dist_loss.mean().item(),
        'com_loss':   com_loss.mean().item(),
    }

    return loss, metrics