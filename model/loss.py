import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from model.util import q_sample_positions


@torch.jit.script
def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances efficiently."""
    return torch.cdist(coords, coords)


@torch.jit.script
def pairwise_distance_distribution_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    bins: torch.Tensor,
    sigma: float = 0.15
) -> torch.Tensor:
    """
    Match the distribution of pairwise distances using a soft histogram (KL divergence).

    This encourages the model to reproduce realistic bond-length distributions
    rather than arbitrary point clouds.

    Args:
        pred_coords: [B, N, 3] predicted x0 coordinates
        true_coords: [B, N, 3] ground-truth coordinates
        bins: [num_bins] distance bin centres
        sigma: Gaussian kernel width for soft binning

    Returns:
        loss: [B] per-sample KL divergence
    """
    B, N = pred_coords.shape[:2]

    pred_dists = torch.cdist(pred_coords, pred_coords).view(B, -1)   # [B, N*N]
    true_dists = torch.cdist(true_coords, true_coords).view(B, -1)

    bins_exp = bins.view(1, 1, -1)   # [1, 1, num_bins]

    pred_hist = torch.exp(-0.5 * ((pred_dists.unsqueeze(-1) - bins_exp) / sigma) ** 2)
    true_hist = torch.exp(-0.5 * ((true_dists.unsqueeze(-1) - bins_exp) / sigma) ** 2)

    pred_hist = pred_hist.sum(dim=1) + 1e-8   # [B, num_bins]
    true_hist = true_hist.sum(dim=1) + 1e-8

    pred_hist = pred_hist / pred_hist.sum(dim=-1, keepdim=True)
    true_hist = true_hist / true_hist.sum(dim=-1, keepdim=True)

    kl = (true_hist * (true_hist / pred_hist).log()).sum(dim=-1)   # [B]
    return kl


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
    Standard DDPM epsilon-prediction loss with lightweight geometric guidance.

    Noise is always drawn from N(0, 1) — the standard DDPM convention — so
    that training and validation see the same distribution and the noise
    schedule is honoured.

    Loss components
    ---------------
    coord_loss  – MSE between predicted and actual coordinate noise  (main signal)
    type_loss   – MSE between predicted and actual type noise        (main signal)
    dist_loss   – MSE on pairwise distance matrix of predicted x0    (geometry)
    com_loss    – Centre-of-mass preservation for predicted x0

    Args:
        model        : Diffusion model
        coords0      : [B, N, 3] ground-truth coordinates
        types0       : [B, N, vocab_size] ground-truth atom types (one-hot / soft)
        t            : [B] or [B, 1] timesteps (integer, 0 … T-1)
        scheduler    : DDIMScheduler (provides alphas_cumprod)
        device       : torch.device
        lambda_*     : Loss-term weights

    Returns:
        loss    : [B] per-sample total loss
        metrics : dict of scalar diagnostics
    """
    B, N, _ = coords0.shape

    # ── Noise schedule ──────────────────────────────────────────────────────
    t_idx = t.squeeze().cpu() if t.ndim > 1 else t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t_idx].view(B, 1, 1).to(device)

    sqrt_alpha         = alpha_prod_t.sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt()

    # ── Sample noisy inputs with STANDARD Gaussian noise ────────────────────
    # Using non-unit variance here (e.g. * 2.7) breaks the noise schedule and
    # creates a mismatch with validation / generation which use N(0,1).
    coord_noise = torch.randn_like(coords0)
    type_noise  = torch.randn_like(types0)

    x_t      = sqrt_alpha * coords0 + sqrt_one_minus_alpha * coord_noise
    types_t  = sqrt_alpha * types0  + sqrt_one_minus_alpha * type_noise

    # ── Forward pass (single) ───────────────────────────────────────────────
    coord_pred, type_pred = model(types_t, x_t, t_idx.to(device))

    # ── Padding mask ─────────────────────────────────────────────────────────
    # Null/padding class is index 0; real atoms are indices 1-5.
    # We use this to apply different masking rules to different loss terms.
    real_mask = types0.argmax(dim=-1) != 0     # [B, N]  True = real atom

    # ── Denoising losses ─────────────────────────────────────────────────────
    # coord_loss: real atoms only — padding coords are artificial zeros and
    # should not constrain the coordinate prediction.
    coord_mask = real_mask.unsqueeze(-1).expand_as(coord_noise)
    n_real     = real_mask.sum(dim=1).float().clamp(min=1)
    coord_loss = ((coord_pred - coord_noise) ** 2 * coord_mask).sum(dim=(1, 2)) / (n_real * 3)

    # type_loss: ALL positions including padding.
    # Padding positions have a concrete target (null class one-hot).  Training
    # on them is what teaches the model to output low probability for null class
    # at real atom positions, and high probability at padding positions.
    # Without this, the model never learns to signal "no atom here" and
    # generation always produces max_N atoms.
    type_loss = F.mse_loss(type_pred, type_noise, reduction='none').mean(dim=(1, 2))

    # ── Reconstruct x0 ──────────────────────────────────────────────────────
    x0_pred = (x_t - sqrt_one_minus_alpha * coord_pred) / (sqrt_alpha + 1e-8)

    
    true_dists = torch.cdist(coords0, coords0)
    pred_dists = torch.cdist(x0_pred, x0_pred)
    real_mask = real_mask.float()
    mask_2d = torch.matmul(real_mask, real_mask.transpose(-1, -2))

    dist_loss = F.mse_loss(pred_dists, true_dists)
    dist_loss = (dist_loss * mask_2d).mean()
    # ── Distance loss — real atom pairs only ─────────────────────────────────
    #pair_mask = real_mask.unsqueeze(2) & real_mask.unsqueeze(1)
    #triu_base = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    #triu_mask = pair_mask & triu_base
    #n_pairs    = triu_mask.sum(dim=(1, 2)).float().clamp(min=1)
    #dist_loss  = (((pred_dists - true_dists) ** 2) * triu_mask).sum(dim=(1, 2)) / n_pairs

    # ── Centre-of-mass loss — real atoms only ────────────────────────────────
    n_real_3d = n_real.unsqueeze(-1)
    com_true  = (coords0 * real_mask.unsqueeze(-1)).sum(dim=1) / n_real_3d
    com_pred  = (x0_pred * real_mask.unsqueeze(-1)).sum(dim=1) / n_real_3d
    com_loss  = ((com_pred - com_true) ** 2).mean(dim=-1)
        
    t_norm    = t_idx.float().to(device) / (scheduler.config.num_train_timesteps - 1)
    denoise_w = 0.3 + 0.7 * t_norm     # 0.3 at t=0  → 1.0 at t=T
    geom_w    = 1.0 - 0.9 * t_norm     # 1.0 at t=0  → 0.1 at t=T
    # ── Total loss ───────────────────────────────────────────────────────────
    loss = (
        denoise_w * coord_loss +
        lambda_type * type_loss
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