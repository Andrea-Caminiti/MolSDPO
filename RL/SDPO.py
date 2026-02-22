import math 
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict
from model.model import TabascoV2
from model.util import gumbel_softmax, get_indices


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _gather_alphas(alphas_cumprod, t):
    """
    Gather from a 1-D alphas_cumprod using an index tensor `t` of any shape.
    Returns a tensor with the same shape as `t`.

    OPTIMIZATION: moved alphas to t.device here rather than calling .cpu() on t,
    which was causing a GPU→CPU sync on every single call (hundreds of times per
    training step).  alphas_cumprod is small (1000 floats) so the one-time .to()
    is essentially free, and no CPU round-trip occurs.
    """
    alphas = alphas_cumprod.to(t.device)
    return alphas.gather(0, t.flatten()).reshape(t.shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = _gather_alphas(self.alphas_cumprod, timestep).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        _gather_alphas(self.alphas_cumprod, prev_timestep.clamp(min=0)).to(timestep.device),
        self.final_alpha_cumprod.expand_as(prev_timestep).to(timestep.device)
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def _compute_log_prob(x_prev, x_prev_mean, sigma_t, t_batched=False):
    std_dev = torch.clamp(sigma_t, min=1e-4)
    log_prob = (
        -((x_prev.detach() - x_prev_mean) ** 2) / (2 * std_dev ** 2)
        - torch.log(std_dev)
        - math.log(math.sqrt(2 * math.pi))
    )
    if t_batched:
        dims = tuple(range(2, log_prob.ndim))
    else:
        dims = tuple(range(1, log_prob.ndim))
    return log_prob.sum(dim=dims)   # ← sum, not mean


def categorical_reverse_step(
    scheduler,
    model_output,
    t,
    x_t,
    eta=0.2,
    generator=None,
    t_batched=False,
    x_prev=None
):
    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    # _gather_alphas handles t of any shape (scalar, 1D, or 2D [B,T])
    alpha_prod_t = _gather_alphas(scheduler.alphas_cumprod, t).to(x_t.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        _gather_alphas(scheduler.alphas_cumprod, prev_timestep.clamp(min=0)).to(x_t.device),
        scheduler.final_alpha_cumprod.expand_as(prev_timestep).to(x_t.device)
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, x_t.shape).to(x_t.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x_t.shape).to(x_t.device)
    beta_prod_t = 1 - alpha_prod_t

    variance = _get_variance(scheduler, t, prev_timestep)
    sigma_t = eta * variance ** 0.5
    sigma_t = _left_broadcast(sigma_t, x_t.shape).to(x_t.device)

    x0_pred = (x_t - model_output * torch.sqrt(beta_prod_t)) / torch.sqrt(alpha_prod_t)
    coef_x0 = torch.sqrt(alpha_prod_t_prev)
    coef_eps = torch.sqrt(1 - alpha_prod_t_prev - sigma_t ** 2)
    x_prev_mean = coef_x0 * x0_pred + coef_eps * model_output

    if x_prev is None:
        noise = torch.randn_like(x_t) if generator is None else torch.randn_like(x_t, generator=generator)
        x_prev = x_prev_mean + sigma_t * noise

    log_prob = _compute_log_prob(x_prev, x_prev_mean, sigma_t, t_batched=t_batched)

    return x_prev.type(x_t.dtype), x0_pred, log_prob


def ddim_step_with_logprob(
    scheduler,
    model_output,
    t,
    x_t,
    eta=0.2,
    generator=None,
    t_batched=False,
    x_prev=None
):
    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    alpha_prod_t = _gather_alphas(scheduler.alphas_cumprod, t).to(x_t.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        _gather_alphas(scheduler.alphas_cumprod, prev_timestep.clamp(min=0)).to(x_t.device),
        scheduler.final_alpha_cumprod.expand_as(prev_timestep).to(x_t.device)
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, x_t.shape).to(x_t.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x_t.shape).to(x_t.device)
    beta_prod_t = 1 - alpha_prod_t

    variance = _get_variance(scheduler, t, prev_timestep)
    sigma_t = eta * variance ** 0.5
    sigma_t = _left_broadcast(sigma_t, x_t.shape).to(x_t.device)

    x0_pred = (x_t - model_output * torch.sqrt(beta_prod_t)) / torch.sqrt(alpha_prod_t)
    coef_x0 = torch.sqrt(alpha_prod_t_prev)
    coef_eps = torch.sqrt(1 - alpha_prod_t_prev - sigma_t ** 2)
    x_prev_mean = coef_x0 * x0_pred + coef_eps * model_output

    if x_prev is None:
        noise = torch.randn_like(x_t) if generator is None else torch.randn_like(x_t, generator=generator)
        x_prev = x_prev_mean + sigma_t * noise

    log_prob = _compute_log_prob(x_prev, x_prev_mean, sigma_t, t_batched=t_batched)

    return x_prev.type(x_t.dtype), x0_pred.type(x_t.dtype), log_prob


def recompute_log_probs(
    model: TabascoV2,
    scheduler,
    coords_traj,    # [B, T+1, N, 3]  — full trajectory including x_0
    atoms_traj,     # [B, T+1, N, A]
    eta: float = 0.2
):
    """
    Re-evaluates log P(x_{t-1} | x_t, model) for every transition in a stored
    trajectory using a BATCHED forward pass.

    OPTIMIZATION NOTE: callers in train.py should pass both trajectories
    concatenated along dim=0 (doubling B) so a single call replaces two
    separate calls — halving the number of frozen forward passes per step.

    Args:
        coords_traj:  [B, T+1, N, 3]
        atoms_traj:   [B, T+1, N, A]
    Returns:
        log_probs_coord: [B, T]
        log_probs_atoms: [B, T]
    """
    B, Tp1, N, _ = coords_traj.shape
    T = Tp1 - 1

    coords_cur  = coords_traj[:, :-1]   # [B, T, N, 3]
    coords_next = coords_traj[:, 1:]    # [B, T, N, 3]
    atoms_cur   = atoms_traj[:, :-1]    # [B, T, N, A]
    atoms_next  = atoms_traj[:, 1:]     # [B, T, N, A]

    # Build per-step timesteps correctly: [T] repeated for each batch item → [B, T]
    timesteps_1d = scheduler.timesteps              # [T]
    timesteps_BT = timesteps_1d.unsqueeze(0).expand(B, -1)  # [B, T]

    coord_pred, types_pred = model(atoms_cur, coords_cur, timesteps_BT, batched=True)

    _, _, log_prob_coord = ddim_step_with_logprob(
        scheduler, coord_pred, timesteps_BT, coords_cur,
        eta=eta, x_prev=coords_next, t_batched=True
    )
    _, _, log_prob_atoms = categorical_reverse_step(
        scheduler, types_pred, timesteps_BT, atoms_cur,
        eta=eta, x_prev=atoms_next, t_batched=True
    )

    return log_prob_coord, log_prob_atoms   # both [B, T]


def _bimodal_cosine_sim(
    all_clean: torch.Tensor,        # [B, T, N, D_coord + D_atom]
    reference: torch.Tensor,        # [B, T, N, D_coord + D_atom]  (already expanded)
    coord_dim: int = 3,
    coord_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute cosine similarity between all_clean and reference with the two
    modalities (coords and atom types) handled separately and then combined.
    """
    atom_weight = 1.0 - coord_weight

    coords     = all_clean[..., :coord_dim]
    atoms      = all_clean[..., coord_dim:]
    ref_coords = reference[..., :coord_dim]
    ref_atoms  = reference[..., coord_dim:]

    coords     = F.normalize(coords,     dim=-1, eps=1e-8)
    ref_coords = F.normalize(ref_coords, dim=-1, eps=1e-8)
    atoms      = F.normalize(atoms,      dim=-1, eps=1e-8)
    ref_atoms  = F.normalize(ref_atoms,  dim=-1, eps=1e-8)

    sim_coord = torch.cosine_similarity(
        coords.flatten(2, 3), ref_coords.flatten(2, 3), dim=2
    )   # [B, T]
    sim_atom = torch.cosine_similarity(
        atoms.flatten(2, 3), ref_atoms.flatten(2, 3), dim=2
    )   # [B, T]

    return coord_weight * sim_coord + atom_weight * sim_atom


def pipeline_with_logprob(
    self: TabascoV2,
    x, types,
    scheduler,
    device,
    B: int = 32,
    N: int = 29,
    num_inference_steps: int = 50,
    eta=0.2,
    coord_weight: float = 0.5,
    coord_dim: int = 3,
):
    all_clean = []
    all_clean_t = []
    all_log_probs = []
    all_mols = [(x, types)]
    mols = []
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    for i, t in enumerate(timesteps):
        coord_logits, type_logits = self(types, x, t)

        x, clean, log_prob_coord = ddim_step_with_logprob(scheduler, coord_logits, t, x, eta=eta)
        types, clean_t, log_prob_type = categorical_reverse_step(scheduler, type_logits, t, types, eta=eta)

        all_mols.append((x, types))
        all_clean.append(torch.cat((clean, clean_t), -1))
        all_log_probs.append(torch.cat((log_prob_coord.unsqueeze(-1), log_prob_type.unsqueeze(-1)), -1))

        if i == 0 or i == num_inference_steps - 1:
            clean_t = F.softmax(clean_t, dim=-1)
            y_hard = torch.zeros_like(clean_t)
            y_hard = torch.scatter(y_hard, -1, clean_t.argmax(dim=-1, keepdim=True), 1.0)
            mols.append([clean, y_hard])

    all_clean = torch.stack(all_clean, dim=1)   # [B, T, N, D]

    # --- Bimodal anchor selection -------------------------------------------
    first_expanded  = all_clean[:, 0:1].expand_as(all_clean)
    last_expanded   = all_clean[:, -1:].expand_as(all_clean)

    sim_first = _bimodal_cosine_sim(
        all_clean, first_expanded,
        coord_dim=coord_dim, coord_weight=coord_weight
    )
    sim_last = _bimodal_cosine_sim(
        all_clean, last_expanded,
        coord_dim=coord_dim, coord_weight=coord_weight
    )

    divergence  = sim_first + sim_last
    anchor_steps = torch.argmin(divergence, dim=1)
    ori_latents_anchor = all_clean[
        torch.arange(len(anchor_steps), device=device), anchor_steps
    ]

    anchor_expanded = ori_latents_anchor.unsqueeze(1).expand_as(all_clean)
    sim_anchor = _bimodal_cosine_sim(
        all_clean, anchor_expanded,
        coord_dim=coord_dim, coord_weight=coord_weight
    )
    # ------------------------------------------------------------------------
    c, a = torch.split(ori_latents_anchor, [3, 6], dim=-1)
    ori_latents_a = F.softmax(a, dim=-1)
    y_hard = torch.zeros_like(ori_latents_a)
    y_hard = torch.scatter(y_hard, -1, ori_latents_a.argmax(dim=-1, keepdim=True), 1.0)
    mols.insert(1, [c, y_hard])

    all_log_probs = torch.stack(all_log_probs, dim=1)
    return mols, all_mols, all_log_probs, anchor_steps, sim_first, sim_anchor, sim_last