import math 
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict
from model.model import TabascoV2
from model.util import gumbel_softmax, get_indices


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

def categorical_reverse_step(
    scheduler, 
    model_output,      # predicted ε OR x0 OR v
    t,                 # current timestep
    x_t,               # noisy sample at time t
    eta=0.2,
    generator=None,
    t_batched = False,
    x_prev = None
):
    """
    Minimal DDIM step using simple schedule.
    Returns:
        x_prev              -- sample at t-1
        x0_pred             -- predicted clean sample
        log_prob            -- mean log-prob of x_prev under transition distribution
    """
    # ---- 1. Setup ----
    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod.gather(0, t.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()), scheduler.final_alpha_cumprod
    )
    if t_batched:
        B, T = x_t.shape[:2]
        alpha_prod_t = alpha_prod_t.reshape(B, T)
        alpha_prod_t_prev = alpha_prod_t_prev.reshape(B, T)
    alpha_prod_t = _left_broadcast(alpha_prod_t, x_t.shape).to(x_t.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x_t.shape).to(x_t.device)
    # ---- 4. Add noise if eta > 0 ---
    beta_prod_t = 1 - alpha_prod_t

    variance = _get_variance(scheduler, t, prev_timestep)
    sigma_t = eta * variance **0.5
    if t_batched:
        sigma_t = sigma_t.reshape(B, T)
    sigma_t = _left_broadcast(sigma_t, x_t.shape)
    x0_pred = (x_t - model_output * torch.sqrt(beta_prod_t)) / torch.sqrt(alpha_prod_t)

    # ---- 3. Deterministic DDIM update ----
    coef_x0  = torch.sqrt(alpha_prod_t_prev)
    coef_eps = torch.sqrt(1 - alpha_prod_t_prev - sigma_t**2)

    x_prev_mean = coef_x0 * x0_pred + coef_eps * model_output
    noise = torch.randn_like(x_t) if generator is None else torch.randn_like(x_t, generator=generator)
    if  x_prev is None:
        x_prev =  (x_prev_mean + sigma_t * noise)
        
    # ---- 5. Compute log-prob (under Gaussian transition) ----
    # If sigma_t = 0 (deterministic DDIM), std_dev is tiny → avoid NaNs by eps shift
    std_dev = torch.clamp(sigma_t, min=1e-2)
    log_prob = (
        -((x_prev.detach() - x_prev_mean) ** 2) / (2 * std_dev**2)
        - torch.log(torch.as_tensor(std_dev))
        - math.log(math.sqrt(2 * math.pi))
    )
    # mean over all dimensions except batch dim
    if t_batched:
        dims = tuple(range(2, log_prob.ndim))
    else:
        dims = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=dims)

    return x_prev.type(x_t.dtype), x0_pred, log_prob


def ddim_step_with_logprob(
    scheduler, 
    model_output,      # predicted ε OR x0 OR v
    t,                 # current timestep
    x_t,               # noisy sample at time t
    eta=0.2,
    generator=None,
    t_batched = False,
    x_prev = None
):
    """
    Minimal DDIM step using simple schedule.
    Returns:
        x_prev              -- sample at t-1
        x0_pred             -- predicted clean sample
        log_prob            -- mean log-prob of x_prev under transition distribution
    """
    # ---- 1. Setup ----
    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod.gather(0, t.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()), scheduler.final_alpha_cumprod
    )
    if t_batched:
        B, T = x_t.shape[:2]
        alpha_prod_t = alpha_prod_t.reshape(B, T)
        alpha_prod_t_prev = alpha_prod_t_prev.reshape(B, T)
    alpha_prod_t = _left_broadcast(alpha_prod_t, x_t.shape).to(x_t.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x_t.shape).to(x_t.device)
    # ---- 4. Add noise if eta > 0 ---
    beta_prod_t = 1 - alpha_prod_t

    variance = _get_variance(scheduler, t, prev_timestep)
    sigma_t = eta * variance **0.5
    if t_batched:
        sigma_t = sigma_t.reshape(B, T)
    sigma_t = _left_broadcast(sigma_t, x_t.shape)
    x0_pred = (x_t - model_output * torch.sqrt(beta_prod_t)) / torch.sqrt(alpha_prod_t)

    # ---- 3. Deterministic DDIM update ----
    coef_x0  = torch.sqrt(alpha_prod_t_prev)
    coef_eps = torch.sqrt(1 - alpha_prod_t_prev - sigma_t**2)

    x_prev_mean = coef_x0 * x0_pred + coef_eps * model_output
    noise = torch.randn_like(x_t) if generator is None else torch.randn_like(x_t, generator=generator)
    if  x_prev is None:
        x_prev =  (x_prev_mean + sigma_t * noise)
        
    # ---- 5. Compute log-prob (under Gaussian transition) ----
    # If sigma_t = 0 (deterministic DDIM), std_dev is tiny → avoid NaNs by eps shift
    std_dev = torch.clamp(sigma_t, min=1e-2)
    log_prob = (
        -((x_prev.detach() - x_prev_mean) ** 2) / (2 * std_dev**2)
        - torch.log(torch.as_tensor(std_dev))
        - math.log(math.sqrt(2 * math.pi))
    )
    # mean over all dimensions except batch dim
    if t_batched:
        dims = tuple(range(2, log_prob.ndim))
    else:
        dims = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=dims)
    return x_prev.type(x_t.dtype), x0_pred.type(x_t.dtype), log_prob

def pipeline_with_logprob(
    self: TabascoV2,
    x, types,
    scheduler, 
    device,
    B: int = 32, 
    N: int = 29,
    num_inference_steps: int = 50,
    eta = 0.2
):    
    all_clean = []
    all_clean_t = []
    all_log_probs = []
    all_mols = [(x, types)] #All produced molecules
    mols = [] #First, last and anchor molecules
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    for i, t in enumerate(timesteps):
        # predict the noise residual
        coord_logits, type_logits = self(types, x, t)

        # compute the previous noisy sample x_t -> x_t-1
        x, clean, log_prob_coord = ddim_step_with_logprob(scheduler, coord_logits, t, x, eta=eta)
        types, clean_t, log_prob_type = categorical_reverse_step(scheduler, type_logits,  t, types, eta=eta)
        
        all_mols.append((x, types))
        all_clean.append(torch.cat((clean, clean_t), -1))
        all_log_probs.append(torch.cat((log_prob_coord.unsqueeze(-1), log_prob_type.unsqueeze(-1)), -1))
        if i == 0 or i == num_inference_steps - 1:
            #print(clean_t)

            clean_t = F.softmax(clean_t, dim=-1)
            y_hard = torch.zeros_like(clean_t)
            y_hard = torch.scatter(y_hard, -1, clean_t.argmax(dim=-1, keepdim=True), 1.0)
            mols.append([clean, y_hard])

    all_clean = torch.stack(all_clean, dim=1)

    sim_first = torch.cosine_similarity(
        all_clean.flatten(2, 3),
        all_clean[:, 0:1].repeat(1, num_inference_steps, 1, 1).flatten(2, 3), dim=2
    )
    sim_last = torch.cosine_similarity(
        all_clean.flatten(2, 3),
        all_clean[:, -1:].repeat(1, num_inference_steps, 1, 1).flatten(2, 3), dim=2
    )

# (You'll need to handle the time/batch dimensions carefully here)
    divergence = sim_first + sim_last

# Find the minimum on the smooth signal
    anchor_steps = torch.argmin(divergence, dim=1)
    ori_latents_anchor = all_clean[torch.arange(len(anchor_steps), device=device), anchor_steps]

    sim_anchor = torch.cosine_similarity(
        all_clean.flatten(2, 3),
        ori_latents_anchor.unsqueeze(1).repeat(1, num_inference_steps, 1, 1).flatten(2, 3), dim=2
    )
    c, a = torch.split(ori_latents_anchor, [3, 6], dim=-1)
    ori_latents_a = F.softmax(a, dim=-1)
    y_hard = torch.zeros_like(ori_latents_a)
    y_hard = torch.scatter(y_hard, -1, ori_latents_a.argmax(dim=-1, keepdim=True), 1.0)
    mols.insert(1, [c, y_hard])
    all_log_probs = torch.stack(all_log_probs, dim=1)
    return mols, all_mols, all_log_probs, anchor_steps, sim_first, sim_anchor, sim_last
