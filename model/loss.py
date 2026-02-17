
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from model.util import q_sample_positions


@torch.jit.script
def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances efficiently.
    
    Args:
        coords: [B, N, 3] coordinate tensor
        
    Returns:
        distances: [B, N, N] pairwise distance matrix
    """
    return torch.cdist(coords, coords)


def compute_type_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    atom_vocab_size: int,
    valid_mask: torch.Tensor,
    weights: torch.Tensor
) -> Dict[str, float]:
    """
    Compute accuracy metrics for atom type predictions.

    Args:
        logits: [B, N, vocab_size] predicted logits
        targets: [B, N] ground truth atom type indices
        atom_vocab_size: Number of atom types
        valid_mask: [B, N] mask for valid atoms (not padding)
        weights: [vocab_size] class weights

    Returns:
        metrics: Dict containing overall and per-class accuracy
    """
    with torch.no_grad():
        # Get predictions
        pred = logits.argmax(dim=-1)  # [B, N]
        
        # Flatten for easier computation
        pred_flat = pred.view(-1)
        true_flat = targets.view(-1)
        mask_flat = valid_mask.view(-1) if valid_mask is not None else None
        
        # Apply mask if provided
        if mask_flat is not None:
            pred_flat = pred_flat[mask_flat]
            true_flat = true_flat[mask_flat]
        
        # Overall accuracy
        correct = (pred_flat == true_flat).float()
        overall_acc = correct.mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for cls in range(atom_vocab_size):
            cls_mask = (true_flat == cls)
            if cls_mask.sum() > 0:
                cls_correct = (pred_flat[cls_mask] == cls).float().mean()
                per_class_acc[f'acc_class_{cls}'] = cls_correct.item()
            else:
                per_class_acc[f'acc_class_{cls}'] = float('nan')
        
        metrics = {
            "overall_acc": overall_acc.item()
        }
        metrics.update(per_class_acc)
    
    return metrics


def p_losses_joint_absorb(
    model: torch.nn.Module,
    coords0: torch.Tensor,
    types0: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    device: torch.device,
    lambda_type: float = 1.0,
    lambda_dist: float = 0.001,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute joint loss for coordinates and atom types with absorbing diffusion.
    
    Args:
        model: Diffusion model
        coords0: [B, N, 3] ground truth coordinates
        types0: [B, N, vocab_size] ground truth atom types (one-hot)
        t: [B, 1] or [B] timesteps
        scheduler: DDIM scheduler with alpha_cumprod
        device: Device for computation
        lambda_type: Weight for type prediction loss
        lambda_dist: Weight for distance preservation loss
        eps: Small constant for numerical stability
        
    Returns:
        loss: [B] total loss per sample
        coord_loss: [B] coordinate prediction loss
        type_loss: [B] type prediction loss
        dist_loss: [B] distance preservation loss
        metrics: Dict of diagnostic metrics
    """
    B, N, _ = coords0.shape
    
    # Get noise schedule parameters
    # Handle both [B, 1] and [B] shaped timesteps
    t_idx = t.squeeze().cpu() if t.ndim > 1 else t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t_idx]  # [B]
    
    # Reshape for broadcasting: [B, 1, 1]
    alpha_prod_t = alpha_prod_t.view(B, 1, 1).to(device)
    
    sqrt_alpha = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_prod_t)
    
    # Sample noisy versions (q_sample)
    types_t, types_noise = q_sample_positions(types0, sqrt_alpha, sqrt_one_minus_alpha)
    x_t, coord_noise = q_sample_positions(coords0, sqrt_alpha, sqrt_one_minus_alpha)
    
    # Model forward pass
    coord_pred, type_pred = model(types_t, x_t, t)
    
    # ========== Coordinate Loss ==========
    coord_loss = F.mse_loss(coord_pred, coord_noise, reduction='none')  # [B, N, 3]
    coord_loss = coord_loss.mean(dim=(1, 2))  # [B]
    
    # ========== Type Loss ==========
    type_loss = F.mse_loss(type_pred, types_noise, reduction='none')  # [B, N, vocab_size]
    type_loss = type_loss.mean(dim=(1, 2))  # [B]
    
    # ========== Distance Preservation Loss ==========
    # Predict x0 from noisy observation
    x0_pred = (x_t - coord_pred * sqrt_one_minus_alpha) / sqrt_alpha
    
    # Compute distance matrices (only upper triangle to avoid redundancy)
    # Create mask for upper triangle (excluding diagonal)
    triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    
    # Compute distances
    true_dists = compute_pairwise_distances(coords0)  # [B, N, N]
    pred_dists = compute_pairwise_distances(x0_pred)  # [B, N, N]
    
    # Apply mask and compute loss
    dist_loss = F.mse_loss(
        pred_dists[:, triu_mask], 
        true_dists[:, triu_mask], 
        reduction='none'
    )  # [B, N*(N-1)/2]
    dist_loss = dist_loss.mean(dim=1)  # [B]
    
    # ========== Total Loss ==========
    loss = coord_loss + lambda_type * type_loss + lambda_dist * dist_loss
    
    # ========== Diagnostic Metrics ==========
    metrics = {}
    
    # Coordinate prediction metrics (mean per dimension)
    pred_coords = coord_pred.mean(dim=(0, 1))  # [3]
    real_coords = coord_noise.mean(dim=(0, 1))  # [3]
    metrics.update({
        'pred_x': pred_coords[0].item(),
        'pred_y': pred_coords[1].item(),
        'pred_z': pred_coords[2].item(),
        'real_x': real_coords[0].item(),
        'real_y': real_coords[1].item(),
        'real_z': real_coords[2].item(),
    })
    
    # Type prediction metrics (mean per class)
    pred_types = type_pred.mean(dim=(0, 1))  # [vocab_size]
    real_types = types_noise.mean(dim=(0, 1))  # [vocab_size]
    
    vocab_size = type_pred.shape[-1]
    for i in range(vocab_size):
        metrics[f'pred_type_{i}'] = pred_types[i].item()
        metrics[f'noise_type_{i}'] = real_types[i].item()
    
    # Loss component metrics
    metrics.update({
        'coord_loss_mean': coord_loss.mean().item(),
        'type_loss_mean': type_loss.mean().item(),
        'dist_loss_mean': dist_loss.mean().item(),
    })
    
    return loss, coord_loss, type_loss, lambda_dist * dist_loss, metrics


def p_losses_joint_absorb_efficient(
    model: torch.nn.Module,
    coords0: torch.Tensor,
    types0: torch.Tensor,
    t: torch.Tensor,
    scheduler,
    device: torch.device,
    lambda_type: float = 1.0,
    lambda_dist: float = 0.001,
    compute_distance_loss: bool = True
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Memory-efficient version that only returns total loss and minimal metrics.
    
    Use this for training to save memory. Use full version for validation/debugging.
    
    Args:
        model: Diffusion model
        coords0: [B, N, 3] ground truth coordinates
        types0: [B, N, vocab_size] ground truth atom types
        t: [B] or [B, 1] timesteps
        scheduler: DDIM scheduler
        device: Computation device
        lambda_type: Weight for type loss
        lambda_dist: Weight for distance loss
        compute_distance_loss: Whether to compute distance loss (can be disabled for speed)
        
    Returns:
        loss: [B] total loss
        metrics: Minimal diagnostic metrics
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
    
    # Forward pass
    coord_pred, type_pred = model(types_t, x_t, t)
    
    # Compute losses
    coord_loss = F.mse_loss(coord_pred, coord_noise, reduction='none').mean(dim=(1, 2))
    type_loss = F.mse_loss(type_pred, types_noise, reduction='none').mean(dim=(1, 2))
    
    # Optional distance loss
    if compute_distance_loss:
        x0_pred = (x_t - coord_pred * sqrt_one_minus_alpha) / sqrt_alpha
        
        # Efficient distance computation
        triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        true_dists = torch.cdist(coords0, coords0, p=2)
        pred_dists = torch.cdist(x0_pred, x0_pred, p=2)
        
        dist_loss = F.mse_loss(
            pred_dists[:, triu_mask],
            true_dists[:, triu_mask],
            reduction='none'
        ).mean(dim=1)
        
        loss = coord_loss + lambda_type * type_loss + lambda_dist * dist_loss
    else:
        dist_loss = torch.zeros_like(coord_loss)
        loss = coord_loss + lambda_type * type_loss
    
    # Minimal metrics
    metrics = {
        'coord_loss': coord_loss.mean().item(),
        'type_loss': type_loss.mean().item(),
        'dist_loss': dist_loss.mean().item() if compute_distance_loss else 0.0,
    }
    
    return loss, metrics


# ==============================================================================
# Additional utility functions
# ==============================================================================

@torch.jit.script
def safe_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss with masking support.
    
    Args:
        pred: Predictions
        target: Targets
        mask: Boolean mask for valid elements
        
    Returns:
        loss: Masked MSE loss
    """
    diff = (pred - target) ** 2
    masked_diff = diff * mask.unsqueeze(-1).float()
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss


def compute_loss_weights(
    t: torch.Tensor,
    schedule_type: str = 'snr',
    min_snr_gamma: float = 5.0
) -> torch.Tensor:
    """
    Compute timestep-dependent loss weights.
    
    Args:
        t: [B] timesteps
        schedule_type: 'snr' for SNR weighting, 'uniform' for no weighting
        min_snr_gamma: Minimum SNR gamma for clipping
        
    Returns:
        weights: [B] loss weights
    """
    if schedule_type == 'uniform':
        return torch.ones_like(t, dtype=torch.float32)
    
    elif schedule_type == 'snr':
        # SNR weighting from "Elucidating the Design Space of Diffusion-Based Generative Models"
        # weight = min(SNR(t), gamma)
        # This prevents over-weighting of very clean samples
        alpha_t = 1.0 - t.float() / 1000.0  # Simple linear schedule
        snr = alpha_t / (1 - alpha_t + 1e-8)
        weights = torch.clamp(snr, max=min_snr_gamma)
        return weights
    
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")