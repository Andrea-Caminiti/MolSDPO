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
    sigma: float = 0.1
) -> torch.Tensor:
    """
    Match the distribution of pairwise distances using soft histogram.
    
    This encourages the model to generate geometries where the distribution
    of atom-atom distances matches the training data, which implicitly
    enforces realistic bond lengths and molecular shapes.
    
    Args:
        pred_coords: [B, N, 3] predicted coordinates
        true_coords: [B, N, 3] ground truth coordinates  
        bins: [num_bins] distance bin centers (e.g., [0.5, 1.0, 1.5, ...])
        sigma: Gaussian kernel width for soft binning
        
    Returns:
        loss: [B] KL divergence between distance distributions
    """
    B, N = pred_coords.shape[:2]
    
    # Compute all pairwise distances [B, N, N]
    pred_dists = torch.cdist(pred_coords, pred_coords)
    true_dists = torch.cdist(true_coords, true_coords)
    
    # Flatten to [B, N*N]
    pred_dists_flat = pred_dists.view(B, -1)
    true_dists_flat = true_dists.view(B, -1)
    
    # Soft histogram using Gaussian kernels [B, N*N, num_bins]
    pred_dists_expanded = pred_dists_flat.unsqueeze(-1)  # [B, N*N, 1]
    true_dists_expanded = true_dists_flat.unsqueeze(-1)
    bins_expanded = bins.view(1, 1, -1)  # [1, 1, num_bins]
    
    # Compute soft counts
    pred_hist = torch.exp(-0.5 * ((pred_dists_expanded - bins_expanded) / sigma) ** 2)
    true_hist = torch.exp(-0.5 * ((true_dists_expanded - bins_expanded) / sigma) ** 2)
    
    # Sum over distance pairs, normalize
    pred_hist = pred_hist.sum(dim=1) + 1e-8  # [B, num_bins]
    true_hist = true_hist.sum(dim=1) + 1e-8
    
    pred_hist = pred_hist / pred_hist.sum(dim=-1, keepdim=True)
    true_hist = true_hist / true_hist.sum(dim=-1, keepdim=True)
    
    # KL divergence
    kl = (true_hist * (true_hist / pred_hist).log()).sum(dim=-1)  # [B]
    
    return kl


def p_losses_joint_absorb_improved(
    model: torch.nn.Module,
    coords0: torch.Tensor,
    types0: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    device: torch.device,
    lambda_type: float = 1.0,
    lambda_dist: float = 0.001,
    lambda_geom: float = 0.1,
    lambda_com: float = 0.01,
    lambda_mag: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Improved loss with geometric guidance.
    
    Loss components:
    1. coord_loss: MSE on coordinate noise prediction (standard)
    2. type_loss: MSE on atom type noise prediction (standard)
    3. dist_loss: MSE on pairwise distance matrix (existing)
    4. geom_loss: Distance distribution matching (NEW - most important)
    5. com_loss: Center of mass preservation (NEW)
    6. mag_loss: Coordinate magnitude regularization (NEW)
    
    The geom_loss is the key addition - it teaches the model that
    realistic molecules have specific distance distributions centered
    around bond lengths (~1.5 Å), not random point clouds.
    
    Args:
        model: Diffusion model
        coords0: [B, N, 3] ground truth coordinates
        types0: [B, N, vocab_size] ground truth atom types
        t: [B, 1] or [B] timesteps
        scheduler: DDIM scheduler
        device: Device
        lambda_type: Weight for type loss
        lambda_dist: Weight for distance preservation loss
        lambda_geom: Weight for geometric distribution loss (NEW)
        lambda_com: Weight for center of mass loss (NEW)
        lambda_mag: Weight for magnitude regularization (NEW)
        
    Returns:
        loss: [B] total loss
        coord_loss: [B] coordinate noise prediction loss
        type_loss: [B] type noise prediction loss
        dist_loss: [B] distance matrix loss
        geom_loss: [B] geometric distribution loss
        com_loss: [B] center of mass loss
        metrics: Dict of diagnostics
    """
    B, N, _ = coords0.shape
    
    # Get noise schedule
    t_idx = t.squeeze().cpu() if t.ndim > 1 else t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t_idx].view(B, 1, 1).to(device)
    
    sqrt_alpha = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_prod_t)
    
    # Sample noisy versions
    types_t, types_noise = q_sample_positions(types0, sqrt_alpha, sqrt_one_minus_alpha)
    x_t, coord_noise = q_sample_positions(coords0, sqrt_alpha, sqrt_one_minus_alpha)
    
    # Model forward
    coord_pred, type_pred = model(types_t, x_t, t)
    
    # ========== Standard Losses ==========
    coord_loss = F.mse_loss(coord_pred, coord_noise, reduction='none').mean(dim=(1, 2))
    type_loss = F.mse_loss(type_pred, types_noise, reduction='none').mean(dim=(1, 2))
    
    # ========== Predict x0 ==========
    x0_pred = (x_t - coord_pred * sqrt_one_minus_alpha) / sqrt_alpha
    
    # ========== Distance Matrix Loss (existing) ==========
    triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    true_dists = torch.cdist(coords0, coords0, p=2)
    pred_dists = torch.cdist(x0_pred, x0_pred, p=2)
    
    dist_loss = F.mse_loss(
        pred_dists[:, triu_mask],
        true_dists[:, triu_mask],
        reduction='none'
    ).mean(dim=1)
    
    # ========== NEW: Geometric Distribution Loss ==========
    # Define distance bins in normalized space (coords are /2.2)
    # Realistic bond lengths: 1.4-1.6 Å → 0.64-0.73 in normalized space
    # Non-bonded: 2.5+ Å → 1.14+ in normalized space
    bins = torch.linspace(0.2, 2.5, 16, device=device)
    
    geom_loss = pairwise_distance_distribution_loss(
        x0_pred, coords0, bins, sigma=0.15
    )
    
    # ========== NEW: Center of Mass Preservation ==========
    # The center of mass should be preserved during denoising
    # This prevents the molecule from drifting during generation
    com_true = coords0.mean(dim=1)  # [B, 3]
    com_pred = x0_pred.mean(dim=1)  # [B, 3]
    com_loss = F.mse_loss(com_pred, com_true, reduction='none').mean(dim=-1)  # [B]
    
    # ========== NEW: Magnitude Regularization ==========
    # Penalize extremely large coordinates (prevents explosion)
    # Use L2 norm of coordinates as regularizer
    mag_loss = (x0_pred ** 2).mean(dim=(1, 2))  # [B]
    
    # ========== Adaptive Weighting by Timestep ==========
    # At high t (noisy), focus on denoising (coord_loss, type_loss)
    # At low t (clean), focus on geometry (geom_loss, dist_loss)
    t_normalized = t.squeeze().float() / 1000.0  # [B], range [0, 1]
    
    # Weight decreases from 1.0 at t=1000 to 0.3 at t=0
    denoise_weight = 0.3 + 0.7 * t_normalized.view(-1)
    
    # Weight increases from 0.1 at t=1000 to 1.0 at t=0
    geom_weight = 1.0 - 0.9 * t_normalized.view(-1)
    
    # ========== Total Loss ==========
    loss = (
        denoise_weight * coord_loss +
        lambda_type * type_loss +
        lambda_dist * geom_weight * dist_loss +
        lambda_geom * geom_weight * geom_loss +
        lambda_com * com_loss +
        lambda_mag * mag_loss
    )
    
    # ========== Metrics ==========
    metrics = {
        'coord_loss_mean': coord_loss.mean().item(),
        'type_loss_mean': type_loss.mean().item(),
        'dist_loss_mean': dist_loss.mean().item(),
        'geom_loss_mean': geom_loss.mean().item(),
        'com_loss_mean': com_loss.mean().item(),
        'mag_loss_mean': mag_loss.mean().item(),
        'denoise_weight_mean': denoise_weight.mean().item(),
        'geom_weight_mean': geom_weight.mean().item(),
    }
    
    # Add coordinate statistics
    pred_coords_mean = coord_pred.mean(dim=(0, 1))
    metrics.update({
        'pred_x': pred_coords_mean[0].item(),
        'pred_y': pred_coords_mean[1].item(),
        'pred_z': pred_coords_mean[2].item(),
    })
    
    # Add predicted x0 statistics
    x0_mean_dist = pred_dists[:, triu_mask].mean().item()
    x0_min_dist = pred_dists[:, triu_mask].min().item()
    metrics.update({
        'x0_mean_dist': x0_mean_dist,
        'x0_min_dist': x0_min_dist,
    })
    
    return loss, coord_loss, type_loss, dist_loss, geom_loss, com_loss, metrics


def p_losses_joint_absorb_improved_efficient(
    model: torch.nn.Module,
    coords0: torch.Tensor,
    types0: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    device: torch.device,
    lambda_type: float = 1.0,
    lambda_dist: float = 0.001,
    lambda_geom: float = 0.1,
    lambda_com: float = 0.01,
    lambda_mag: float = 0.001,
) -> Tuple[torch.Tensor, Dict]:
    """
    Memory-efficient version with geometric guidance.
    
    Use this for training. Returns only total loss and minimal metrics.
    """
    B, N, _ = coords0.shape
    
    # Get noise schedule
    t_idx = t.squeeze().cpu() if t.ndim > 1 else t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t_idx].view(B, 1, 1).to(device)
    
    sqrt_alpha = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_prod_t)
    
    # Sample noisy versions
    types_t, types_noise = q_sample_positions(types0, sqrt_alpha, sqrt_one_minus_alpha)
    x_t, coord_noise = q_sample_positions(coords0, sqrt_alpha, sqrt_one_minus_alpha)
    
    # Forward
    coord_pred, type_pred = model(types_t, x_t, t)
    
    # Standard losses
    coord_loss = F.mse_loss(coord_pred, coord_noise, reduction='none').mean(dim=(1, 2))
    type_loss = F.mse_loss(type_pred, types_noise, reduction='none').mean(dim=(1, 2))
    
    # Predict x0
    x0_pred = (x_t - coord_pred * sqrt_one_minus_alpha) / sqrt_alpha
    
    # Distance loss
    N = coords0.shape[1]
    triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    true_dists = torch.cdist(coords0, coords0, p=2)
    pred_dists = torch.cdist(x0_pred, x0_pred, p=2)
    dist_loss = F.mse_loss(pred_dists[:, triu_mask], true_dists[:, triu_mask], reduction='none').mean(dim=1)
    
    # Geometric distribution loss
    bins = torch.linspace(0.2, 2.5, 16, device=device)
    geom_loss = pairwise_distance_distribution_loss(x0_pred, coords0, bins, sigma=0.15)
    
    # Center of mass
    com_loss = F.mse_loss(x0_pred.mean(dim=1), coords0.mean(dim=1), reduction='none').mean(dim=-1)
    
    # Magnitude regularization
    mag_loss = (x0_pred ** 2).mean(dim=(1, 2))
    
    # Adaptive weighting
    t_normalized = t.squeeze().float() / 1000.0
    denoise_weight = 0.3 + 0.7 * t_normalized.view(-1)
    geom_weight = 1.0 - 0.9 * t_normalized.view(-1)
    
    # Total loss
    loss = (
        denoise_weight * coord_loss +
        lambda_type * type_loss +
        lambda_dist * geom_weight * dist_loss +
        lambda_geom * geom_weight * geom_loss +
        lambda_com * com_loss +
        lambda_mag * mag_loss
    )
    
    # Minimal metrics
    metrics = {
        'coord_loss': coord_loss.mean().item(),
        'type_loss': type_loss.mean().item(),
        'dist_loss': dist_loss.mean().item(),
        'geom_loss': geom_loss.mean().item(),
        'com_loss': com_loss.mean().item(),
        'mag_loss': mag_loss.mean().item(),
    }
    
    return loss, metrics