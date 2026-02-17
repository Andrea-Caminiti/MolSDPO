import torch
import math

def cosine_beta_schedule(T, min_clamp, max_clamp, s=0.008, device='cpu'):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_cum = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    #return betas
    return betas.clamp(min=min_clamp, max=max_clamp)

def beta_schedule_scaled_linear(
    T,
    beta_start=1e-4,
    beta_end=2e-2,
    device="cpu",
):
    betas = torch.linspace(
        beta_start ** 0.5,
        beta_end ** 0.5,
        T,
        device=device
    ) ** 2
    return betas
    

def precompute_schedule(T, min_clamp, max_clamp, device='cpu'):
    betas = cosine_beta_schedule(T, min_clamp, max_clamp,  device=device)
    
    alphas = 1.0 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)         # \bar{\alpha}_t
    sqrt_alpha_cum = torch.sqrt(alpha_cum)
    sqrt_one_minus_alpha_cum = torch.sqrt(1 - alpha_cum)
    return dict(
        T=T, betas=betas, alphas=alphas, alpha_cum=alpha_cum,
        sqrt_alpha_cum=sqrt_alpha_cum, 
        sqrt_one_minus_alpha_cum=sqrt_one_minus_alpha_cum)