import math
import torch
import torch.nn.functional as F
from typing import Optional
from model.model import TabascoV2


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _gather_alphas(alphas_cumprod, t):
    """Gather from a 1-D alphas_cumprod using an index tensor `t` of any shape."""
    alphas = alphas_cumprod.to(t.device)
    return alphas.gather(0, t.flatten()).reshape(t.shape)


def _get_variance(scheduler, timestep, prev_timestep):
    alpha_prod_t = _gather_alphas(scheduler.alphas_cumprod, timestep).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        _gather_alphas(scheduler.alphas_cumprod, prev_timestep.clamp(min=0)).to(timestep.device),
        scheduler.final_alpha_cumprod.expand_as(prev_timestep).to(timestep.device),
    )
    beta_prod_t      = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


def _compute_log_prob(x_prev, x_prev_mean, sigma_t, t_batched=False):
    std_dev  = torch.clamp(sigma_t, min=1e-4)
    log_prob = (
        -((x_prev.detach() - x_prev_mean) ** 2) / (2 * std_dev ** 2)
        - torch.log(std_dev)
        - math.log(math.sqrt(2 * math.pi))
    )
    dims = tuple(range(2, log_prob.ndim)) if t_batched else tuple(range(1, log_prob.ndim))
    return log_prob.mean(dim=dims)


def ddim_step_with_logprob(
    scheduler,
    model_output,
    t,
    x_t,
    eta       = 0.2,
    generator = None,
    t_batched = False,
    x_prev    = None,
):
    """Single DDIM reverse step with Gaussian log-probability."""
    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    alpha_prod_t = _gather_alphas(scheduler.alphas_cumprod, t).to(x_t.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        _gather_alphas(scheduler.alphas_cumprod, prev_timestep.clamp(min=0)).to(x_t.device),
        scheduler.final_alpha_cumprod.expand_as(prev_timestep).to(x_t.device),
    )
    alpha_prod_t      = _left_broadcast(alpha_prod_t,      x_t.shape)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x_t.shape)
    beta_prod_t = 1 - alpha_prod_t

    variance = _get_variance(scheduler, t, prev_timestep)
    sigma_t  = _left_broadcast(eta * variance ** 0.5, x_t.shape)

    x0_pred     = (x_t - model_output * torch.sqrt(beta_prod_t)) / torch.sqrt(alpha_prod_t)
    x_prev_mean = torch.sqrt(alpha_prod_t_prev) * x0_pred \
                + torch.sqrt(1 - alpha_prod_t_prev - sigma_t ** 2) * model_output

    if x_prev is None:
        noise  = torch.randn_like(x_t) if generator is None else torch.randn_like(x_t, generator=generator)
        x_prev = x_prev_mean + sigma_t * noise

    log_prob = _compute_log_prob(x_prev, x_prev_mean, sigma_t, t_batched=t_batched)

    return x_prev.type(x_t.dtype), x0_pred.type(x_t.dtype), log_prob, x_prev_mean, sigma_t


categorical_reverse_step = ddim_step_with_logprob


def _select_anchor(all_clean: torch.Tensor, device: torch.device):
    """
    Select the anchor step per molecule as the timestep where the trajectory
    diverges most from both its start and end — i.e. the point of maximum
    structural change, used as the mid-trajectory reward checkpoint.

    Returns
    -------
    anchor_steps : [B]   — per-molecule index into the T axis of all_clean
    """
    first_expanded = all_clean[:, 0:1].expand_as(all_clean)
    last_expanded  = all_clean[:, -1:].expand_as(all_clean)

    def _cos_sim(a, b):
        return F.cosine_similarity(a.flatten(2, 3), b.flatten(2, 3), dim=2)

    sim_first = _cos_sim(all_clean, first_expanded)   #[B, T]
    sim_last  = _cos_sim(all_clean, last_expanded)    #[B, T]

    divergence   = sim_first + sim_last               #[B, T]
    anchor_steps = torch.argmin(divergence, dim=1)    #[B]
    return anchor_steps


def pipeline_with_logprob(
    self        : TabascoV2,
    x, types,
    scheduler,
    device,
    B           : int   = 32,
    N           : int   = 29,
    num_inference_steps : int   = 50,
    eta         : float = 0.2,
    coord_dim   : int   = 3,
):
    """
    Full denoising rollout with per-step log-probabilities and
    x_pred snapshots at three trajectory checkpoints.

    Returns
    -------
    mols            : list of 3 tensors [B, N, 3+A]   hard-decoded molecules
                      at t=first, t=anchor, t=last (kept for validation use)
    all_mols        : list of T+1 (coord, types) pairs full trajectory states
    all_log_probs   : [B, T, 2]   (log_p_coord, log_p_types) per step
    anchor_steps    : [B]         per-molecule anchor step index
    x0_pred_first   : [B, N, D]   x_pred at step 0 (D = coord_dim + A)
    x0_pred_anchor  : [B, N, D]   x_pred at anchor step
    x0_pred_last    : [B, N, D]   x_pred at step T-1
    all_means       : [B, T, N, D]
    all_sigmas      : [B, T, N, D]
    """
    all_clean     = []
    all_log_probs = []
    all_mols      = [(x, types)]
    mols          = []
    all_means     = []
    all_sigmas    = []

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    for i, t in enumerate(timesteps):
        coord_logits, type_logits = self(types, x, t)

        x,     clean,   log_prob_coord, mean_coord, sig_coord = ddim_step_with_logprob(
            scheduler, coord_logits, t, x,     eta=eta
        )
        types, clean_t, log_prob_type,  mean_types, sig_types = ddim_step_with_logprob(
            scheduler, type_logits,  t, types, eta=eta
        )

        all_mols.append((x, types))

        all_clean.append(torch.cat((clean, clean_t), dim=-1))          #[B, N, D]
        all_log_probs.append(torch.stack((log_prob_coord, log_prob_type), dim=-1))
        all_means.append(torch.cat([mean_coord, mean_types], dim=-1))
        all_sigmas.append(torch.cat([sig_coord,  sig_types], dim=-1))

        if i == 0 or i == num_inference_steps - 1:
            clean_t_soft = F.softmax(clean_t, dim=-1)
            y_hard = torch.zeros_like(clean_t_soft).scatter_(
                -1, clean_t_soft.argmax(dim=-1, keepdim=True), 1.0
            )
            mols.append(torch.cat([clean, y_hard], dim=-1))

    all_clean     = torch.stack(all_clean,     dim=1)   #[B, T, N, D]
    all_log_probs = torch.stack(all_log_probs, dim=1)   #[B, T, 2]
    all_means     = torch.stack(all_means,     dim=1)   #[B, T, N, D]
    all_sigmas    = torch.stack(all_sigmas,    dim=1)   #[B, T, N, D]

    anchor_steps = _select_anchor(all_clean, device)   #[B]

    ori_latents_anchor = all_clean[torch.arange(B, device=device), anchor_steps]  #[B, N, D]
    c_anc, a_anc = ori_latents_anchor[..., :coord_dim], ori_latents_anchor[..., coord_dim:]
    a_soft = F.softmax(a_anc, dim=-1)
    y_hard = torch.zeros_like(a_soft).scatter_(-1, a_soft.argmax(dim=-1, keepdim=True), 1.0)
    mols.insert(1, torch.cat([c_anc, y_hard], dim=-1))

    x0_pred_first  = all_clean[:, 0]                                               #[B, N, D]
    x0_pred_anchor = all_clean[torch.arange(B, device=device), anchor_steps]      #[B, N, D]
    x0_pred_last   = all_clean[:, -1]                                              #[B, N, D]

    return (
        mols,
        all_mols,
        all_log_probs,
        anchor_steps,
        x0_pred_first,
        x0_pred_anchor,
        x0_pred_last,
        all_means,
        all_sigmas,
    )