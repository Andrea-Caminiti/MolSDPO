
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def gather_index(arr: torch.Tensor, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Gather values from array at indices t."""
    return arr[t].to(device)


def q_sample_positions(
    x0: torch.Tensor, 
    sqrt_alpha_cum: torch.Tensor, 
    sqrt_one_minus_alpha_cum: torch.Tensor, 
    noise: Optional[torch.Tensor] = None
):
    """
    Sample from q(x_t | x_0) - the forward diffusion process.
    
    Args:
        x0: Clean data [B, N, D]
        sqrt_alpha_cum: sqrt(alpha_bar_t) [B, 1, 1] or scalar
        sqrt_one_minus_alpha_cum: sqrt(1 - alpha_bar_t) [B, 1, 1] or scalar
        noise: Optional pre-sampled noise
        
    Returns:
        x_t: Noisy sample
        noise: The noise that was added
    """
    if noise is None:
        noise = torch.randn_like(x0, device=x0.device)

    x_t = sqrt_alpha_cum * x0 + sqrt_one_minus_alpha_cum * noise
    return x_t, noise


class TimeEmbed(nn.Module):
    """
    Sinusoidal time embedding with MLP projection.
    
    Fixed to work with torch.compile by removing batched parameter.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.SiLU(), 
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal time embeddings.
        
        Args:
            t: Timesteps [B] or [B*T] for batched mode
            
        Returns:
            emb: Time embeddings [B, dim] or [B*T, dim]
        """
        device = t.device
        half = self.dim // 2
        
        # Compute frequency bands
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        
        # Compute sinusoidal embeddings
        # Handle both [B] and [B, T, ...] shapes
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., dim]
        
        # Project through MLP
        return self.mlp(emb)


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax sampling for differentiable categorical sampling.
    
    Args:
        logits: Unnormalized log probabilities
        temperature: Temperature for softmax (lower = more discrete)
        hard: If True, return hard one-hot samples with straight-through gradients
        
    Returns:
        Soft or hard samples
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    y = (logits + gumbel_noise) / temperature
    y = F.softmax(y, dim=-1)

    if hard:
        # Straight-through estimator: hard forward, soft backward
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y

    return y


def cosine_warmup_lr(
    step: int, 
    warmup_steps: int, 
    total_steps: int, 
    min_lr_ratio: float = 0.1
) -> float:
    """
    Warmup + Cosine LR schedule with a minimum learning rate floor.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak LR
        
    Returns:
        lr_multiplier: Multiplier for base learning rate
    """
    if step < warmup_steps:
        # Linear warmup from min_lr_ratio → 1
        alpha = step / warmup_steps
        return min_lr_ratio + (1 - min_lr_ratio) * alpha
    
    # Cosine decay from 1 → min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay


def annealed_weight(
    step: int, 
    t_batch: torch.Tensor, 
    T: int, 
    total_training_steps: int
) -> torch.Tensor:
    """
    Compute annealed importance weights for timesteps.
    
    Early in training, focus on harder (higher t) timesteps.
    Later in training, give equal weight to all timesteps.
    
    Args:
        step: Current global training step
        t_batch: Batch of timesteps [B]
        T: Maximum diffusion timestep
        total_training_steps: Total number of training steps
        
    Returns:
        weights: Importance weights [B]
    """
    # Calculate training progress (0.0 to 1.0)
    progress = min(step / total_training_steps, 1.0)
    
    # Power starts at 4.0 (strict focus on high t) 
    # and decays to 1.0 (uniform focus)
    power = 4.0 - (3.0 * progress)
    
    # Higher t gets higher weight early in training
    # w_t = ((T - t) / T) ^ power
    w_t = ((T - t_batch.float()) / T) ** power
    
    return w_t


def get_indices(batch_size: int, num_steps: int = 25, max_t: int = 1000) -> torch.Tensor:
    """
    Generate jittered timestep indices for each sample in batch.
    
    Creates a grid of timesteps with random jitter to improve coverage.
    
    Args:
        batch_size: Number of samples in batch
        num_steps: Number of timesteps to generate per sample
        max_t: Maximum timestep value
        
    Returns:
        timesteps: [B, num_steps] jittered timestep indices
    """
    # Calculate stride between timesteps
    stride = max_t // num_steps
    
    # Generate base grid: [stride, 2*stride, ..., max_t-stride]
    base_grid = torch.arange(stride, max_t, stride)[:num_steps]
    
    # Add random jitter for each batch element
    offsets = torch.randint(0, stride, (batch_size, 1))
    
    # Apply offsets and flip to go from high t (noise) to low t (clean)
    jittered_t = (base_grid + offsets).flip(dims=[1])
    
    # Clamp to valid range
    return jittered_t.clamp(1, max_t - 1)


def top_k_filtering(
    log_diff: torch.Tensor, 
    adv_diff: torch.Tensor, 
    keep_ratio: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter to keep only top-k samples by advantage magnitude.
    
    Useful for focusing training on samples with largest reward gaps.
    
    Args:
        log_diff: Log probability differences
        adv_diff: Advantage differences
        keep_ratio: Fraction of samples to keep (0.0 to 1.0)
        
    Returns:
        filtered_log_diff: Masked log differences
        filtered_adv_diff: Masked advantage differences
    """
    # Find magnitude of advantage gaps
    abs_adv = adv_diff.abs()
    
    # Determine cutoff for top keep_ratio
    k = int(len(adv_diff) * keep_ratio)
    threshold = torch.topk(abs_adv, k, dim=0).values[-1]
    
    # Create mask for samples above threshold
    mask = (abs_adv >= threshold).float()
    
    return log_diff * mask, adv_diff * mask


@torch.jit.script
def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple[int]) -> torch.Tensor:
    """
    Extract values from array at timesteps and broadcast to target shape.
    
    Args:
        arr: Source array [T]
        timesteps: Indices to extract [B]
        broadcast_shape: Target shape (B, N, D)
        
    Returns:
        Broadcasted tensor of shape broadcast_shape
    """
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
