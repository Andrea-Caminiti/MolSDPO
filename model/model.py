import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Optional
from model.util import TimeEmbed, q_sample_positions
from model.RBF import PairwiseRBF
from model.loss import p_losses_joint_absorb_improved_efficient
from model.scheduler import precompute_schedule, cosine_beta_schedule
from model.util import cosine_warmup_lr
from diffusers import DDIMScheduler
from config import DDIM_config
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds, Descriptors

RDLogger.DisableLog('rdApp.*')  


# ---------- Small helpers ----------

try:
    _NativeRMSNorm = nn.RMSNorm   # PyTorch >= 2.4

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-8):
            super().__init__()
            self._norm = _NativeRMSNorm(dim, eps=eps)

        def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
            return self._norm(x)

except AttributeError:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.scale


# ---------- Transformer block (pre-LN) ----------

class SimpleTransformerBlock(nn.Module):
    """Transformer block using F.scaled_dot_product_attention (→ FlashAttention 2)."""
    
    def __init__(self, dim: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout_p = dropout

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_mult, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.norm2 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, batched: bool = False) -> torch.Tensor:
        if batched:
            B, T, N, D = x.shape
            x_ln = self.norm1(x)
            qkv = self.qkv(x_ln).reshape(B, T, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(3, 0, 1, 4, 2, 5)
            q, k, v = qkv.unbind(0)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

            dropout_p = self.attn_dropout_p if self.training else 0.0
            assert q.shape[-2] == attn_bias.shape[-1] == attn_bias.shape[-2], \
                f"Seq mismatch: q={q.shape}, attn_bias={attn_bias.shape}"
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N, D)
        else:
            B, N, D = x.shape
            x_ln = self.norm1(x)
            qkv = self.qkv(x_ln).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            dropout_p = self.attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        x = x + self.out(out)
        x = x + self.ff(self.norm2(x))
        return x


# ---------- Cross-attention block ----------

class SimpleCrossAttentionTransformerBlock(nn.Module):
    """Cross-attention block using F.scaled_dot_product_attention."""
    
    def __init__(self, dim_q: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.dim_q = dim_q
        self.n_heads = n_heads
        self.head_dim = dim_q // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim_q, dim_q, bias=False)
        self.kv_proj = nn.Linear(dim_q, dim_q * 2, bias=False)
        self.out = nn.Linear(dim_q, dim_q)
        self.attn_dropout_p = dropout
        
        self.norm_q = RMSNorm(dim_q) if use_rmsnorm else nn.LayerNorm(dim_q)
        self.norm_kv = RMSNorm(dim_q) if use_rmsnorm else nn.LayerNorm(dim_q)
        
        self.ff = nn.Sequential(
            nn.Linear(dim_q, dim_q * mlp_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * mlp_mult, dim_q),
            nn.Dropout(dropout)
        )
        # Missing in original: norm after cross-attention residual
        self.norm_out = RMSNorm(dim_q) if use_rmsnorm else nn.LayerNorm(dim_q)

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, batched: bool = False) -> torch.Tensor:
        if attn_bias is not None:
            if attn_bias.shape[-1] == 1:
                Nq, Nk = q_x.shape[-2], kv_x.shape[-2]
                attn_bias = attn_bias.expand(*attn_bias.shape[:-2], Nq, Nk)
            attn_bias = attn_bias.contiguous()

        if batched:
            B, T, Nq, Dq = q_x.shape
            Nk = kv_x.shape[2]

            q = self.q_proj(self.norm_q(q_x))
            q = q.reshape(B, T, Nq, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

            kv = self.kv_proj(self.norm_kv(kv_x))
            kv = kv.reshape(B, T, Nk, 2, self.n_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
            k, v = kv.unbind(0)   
            k, v = k.contiguous(), v.contiguous()

            dropout_p = self.attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            out = out.permute(0, 1, 3, 2, 4).reshape(B, T, Nq, Dq)
        else:
            B, Nq, Dq = q_x.shape
            Nk = kv_x.shape[1]

            q = self.q_proj(self.norm_q(q_x))
            q = q.reshape(B, Nq, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

            kv = self.kv_proj(self.norm_kv(kv_x))
            kv = kv.reshape(B, Nk, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)   
            k, v = k.contiguous(), v.contiguous()

            dropout_p = self.attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            out = out.permute(0, 2, 1, 3).reshape(B, Nq, Dq)
        
        x = q_x + self.out(out)
        x = x + self.ff(self.norm_out(x))
        return x
    
# ---------- TabascoV2 ----------

class TabascoV2(nn.Module):
    """TabascoV2 model with FlashAttention and fused RMSNorm."""
    
    def __init__(self,
                 atom_vocab_size: int = 100,
                 d_model: int = 192,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 pos_coord_dim: int = 64,
                 pair_rbf_centers: int = 16,
                 dropout: float = 0.1,
                 use_rmsnorm: bool = True):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        
        self.atom_emb = nn.Sequential(
            nn.Linear(atom_vocab_size, pos_coord_dim),
            nn.SiLU(),
            nn.Linear(pos_coord_dim, half)
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, pos_coord_dim),
            nn.SiLU(),
            nn.Linear(pos_coord_dim, half)
        )
        
        self.input_proj = nn.Linear(d_model, d_model)
        
        self.time_emb = TimeEmbed(d_model // 4)
        self.time_proj = nn.Linear(d_model // 4, d_model)

        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm)
            for _ in range(n_layers)
        ])

        self.rbf_dim = pair_rbf_centers
        self.pair_rbf = PairwiseRBF(num_rbf=pair_rbf_centers)
        # Projects per-pair RBF features → per-head scalar bias (preserves spatial structure)
        self.rbf_to_bias = nn.Linear(pair_rbf_centers, n_heads, bias=False)

        self.crossAtomCoord = SimpleCrossAttentionTransformerBlock(
            half, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm
        )
        self.crossCoordAtom = SimpleCrossAttentionTransformerBlock(
            half, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm
        )

        hidden = d_model * 2
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3)
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, atom_vocab_size)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, atom_idxs: torch.Tensor, coords: torch.Tensor, t: torch.Tensor, batched: bool = False):
        """
        Args:
            atom_idxs: [B, N, vocab_size]
            coords:    [B, N, 3]
            t:         [B]
            batched:   Whether inputs have a time dimension
        """
        atom_e  = self.atom_emb(atom_idxs)
        coord_e = self.coord_proj(coords)
        N = coords.shape[-2]
        x = torch.cat([atom_e, coord_e], dim=-1)
        x = self.input_proj(x)
        
        if batched:
            B, T = coords.shape[0], coords.shape[1]
            temb = self.time_proj(self.time_emb(t))
            temb = temb.view(B, T, -1).unsqueeze(2)
        else:
            temb = self.time_proj(self.time_emb(t))
            temb = temb.unsqueeze(1)
            
        x = x + temb
        
        # ── Pairwise geometric attention bias ──────────────────────────────
        # rbf: (B, N, N, num_rbf)  →  head_bias: (B, N, N, n_heads)
        # Transposed to (B, n_heads, N, N) to match attention shape.
        # FIX: previously used rbf.mean(dim=(1,2)) which collapsed all spatial
        # structure into a single scalar per head — every token pair received
        # the same bias, defeating the purpose of geometry-aware attention.
        rbf = self.pair_rbf(coords, batched=batched)           # (B, N, N, rbf_dim)
        head_bias = self.rbf_to_bias(rbf)       
        # (B, N, N, n_heads)
        if batched: 
            attn_bias  = head_bias.permute(0, 1, 4, 2, 3).contiguous()  # (B, n_heads, N, N)
        else:
            attn_bias  = head_bias.permute(0, 3, 1, 2).contiguous()  # (B, n_heads, N, N)

        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias, batched=batched)
        
        half = self.d_model // 2
        a = x[..., :half]
        c = x[..., half:]

        ac = self.crossAtomCoord(a, c, batched=batched)
        ca = self.crossCoordAtom(c, a, batched=batched)
        
        x = torch.cat([ac, ca], dim=-1)
        
        coord_pred  = self.coord_head(x)
        type_logits = self.type_head(x)
        
        return coord_pred, type_logits


class LightningTabasco(pl.LightningModule):
    """Lightning module for training TabascoV2."""
    
    def __init__(self, args, vocab_enc2atom: torch.Tensor):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.scheduler = DDIMScheduler.from_config(DDIM_config)
        
        self.model = TabascoV2(
            atom_vocab_size=self.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            pos_coord_dim=128,
            pair_rbf_centers=args.d_model // 2,
            dropout=0.1
        )
        
        self.register_buffer("loss_history", torch.ones(args.T) * 1000.0)
        self.register_buffer("sampling_probs", torch.ones(args.T) / args.T)
        self.alpha_ema = 0.01
        self.register_buffer("vocab_enc2atom", vocab_enc2atom)

    @torch.no_grad()
    def update_loss_history(self, t: torch.Tensor, losses: torch.Tensor):
        for i in range(len(t)):
            step_t = t[i].item()
            self.loss_history[step_t] = (
                (1 - self.alpha_ema) * self.loss_history[step_t] + 
                self.alpha_ema * losses[i].item()
            )
        self.sampling_probs.copy_(self.get_sampling_weights())

    def get_sampling_weights(self) -> torch.Tensor:
        weights = torch.sqrt(self.loss_history.clamp(min=0)) + 1e-6
        return weights / weights.sum()
    
    def configure_optimizers(self):
        base_lr = self.args.lr
        
        param_groups = [
            {"params": self.model.atom_emb.parameters(),        "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.coord_proj.parameters(),      "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.input_proj.parameters(),      "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.time_emb.parameters(),        "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.time_proj.parameters(),       "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.pair_rbf.parameters(),        "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.rbf_to_bias.parameters(),     "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.blocks.parameters(),          "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.crossAtomCoord.parameters(),  "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.crossCoordAtom.parameters(),  "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.coord_head.parameters(),      "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.type_head.parameters(),       "lr": base_lr, "weight_decay": 0.01},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / 5000, 1.0)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """Training step with importance sampling."""
        self.model = self.model.train()
        coords, atom_types = batch
        B = coords.shape[0]
        
        t = torch.multinomial(
            self.sampling_probs, 
            B, 
            replacement=True
        ).to(self.device)

        # FIX: previously, a full forward pass was computed here and DISCARDED,
        # then p_losses ran its own forward pass with *different* noise.
        # Now we call p_losses once — single forward pass, consistent noise.
        loss, metrics = p_losses_joint_absorb_improved_efficient(
            self.model, coords, atom_types, t, 
            self.scheduler, self.device, lambda_type=1.0
        )
        
        p_t = self.sampling_probs[t.squeeze()]
        weights = 1.0 / (p_t * self.args.T)
        
        loss_weighted = (loss * weights).mean()
        
        self.update_loss_history(t.squeeze(), loss.detach())
        
        self.log_dict({
            'loss': loss_weighted,
            **metrics
        }, on_step=True, prog_bar=False)
        
        return loss_weighted

    def validation_step(self, batch, batch_idx):
        """Per-batch validation: denoising loss at 4 canonical timesteps."""
        self.model = self.model.eval()
        
        coords, atom_types = batch
        B      = coords.shape[0]
        device = self.device

        with torch.no_grad():
            for t_val in [10, 100, 500, 900]:
                t = torch.full((B,), t_val, device=device).long()

                alpha      = self.scheduler.alphas_cumprod[t_val]
                sqrt_a     = alpha ** 0.5
                sqrt_one_a = (1.0 - alpha) ** 0.5

                # Standard N(0,1) noise — must match training convention
                noise_c = torch.randn_like(coords)
                noise_a = torch.randn_like(atom_types)

                noisy_coords = sqrt_a * coords     + sqrt_one_a * noise_c
                noisy_atoms  = sqrt_a * atom_types + sqrt_one_a * noise_a

                coord_pred, atom_pred = self.model(noisy_atoms, noisy_coords, t)

                # Mask coord loss to real atoms; type loss over all positions
                # (consistent with training: model learns null class for padding)
                real_mask  = atom_types.argmax(dim=-1) != 0
                coord_mask = real_mask.unsqueeze(-1).expand_as(noise_c)
                n_real     = real_mask.sum().float().clamp(min=1)
                loss = (
                    ((coord_pred - noise_c) ** 2 * coord_mask).sum() / (n_real * 3)
                    + F.mse_loss(atom_pred, noise_a)
                )

                self.log(
                    f"val/denoise_loss_t{t_val}", loss,
                    on_step=False, on_epoch=True,
                    prog_bar=(t_val == 100), sync_dist=True,
                )

    def on_validation_epoch_end(self):
        """End-of-epoch generation check."""
        NUM_SAMPLES  = 32
        DDIM_STEPS   = 50          # fewer steps for speed during validation
        device       = self.device

        self.model = self.model.eval()

        # FIX: generation must start from N(0,1) — the stationary distribution
        # of the forward process.  Using * 2.7 / * 3.0 puts the starting point
        # outside the learned support and the DDIM steps cannot recover.
        coords = torch.randn(NUM_SAMPLES, 29, 3,               device=device)
        atoms  = torch.randn(NUM_SAMPLES, 29, self.vocab_size,  device=device)

        self.scheduler.set_timesteps(DDIM_STEPS, device=device)

        timesteps = self.scheduler.timesteps   # e.g. [980, 960, 940, ..., 20, 0]

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((NUM_SAMPLES,), t, device=device).long()

                coord_pred, atom_pred = self.model(atoms, coords, t_batch)

                alpha_t = self.scheduler.alphas_cumprod[t].to(device)

                # BUG WAS HERE: using alphas_cumprod[t - 1] when t is e.g. 960
                # gives alphas_cumprod[959], advancing only 1/1000 of the schedule.
                # The correct previous timestep is the NEXT entry in the DDIM
                # sequence (e.g. 940), not t minus one.
                if i + 1 < len(timesteps):
                    t_prev     = timesteps[i + 1]
                    alpha_prev = self.scheduler.alphas_cumprod[t_prev].to(device)
                else:
                    # Final step: x_{-1} = x0
                    alpha_prev = torch.ones(1, device=device)

                # Deterministic DDIM update
                x0_c = (coords - (1 - alpha_t).sqrt() * coord_pred) / alpha_t.sqrt()
                x0_a = (atoms  - (1 - alpha_t).sqrt() * atom_pred)  / alpha_t.sqrt()

                coords = alpha_prev.sqrt() * x0_c + (1 - alpha_prev).sqrt() * coord_pred
                atoms  = alpha_prev.sqrt() * x0_a + (1 - alpha_prev).sqrt() * atom_pred

        atom_indices = atoms.argmax(dim=-1)
        atom_nums    = self.vocab_enc2atom[atom_indices]

        n_valid = n_connected = n_realistic = 0
        all_atom_counts = []
        all_min_dists   = []
        frags = []

        for i in range(NUM_SAMPLES):
            mask   = atom_nums[i] != 0
            n_atoms = mask.sum().item()
            if n_atoms < 2:
                continue

            all_atom_counts.append(n_atoms)

            c = coords[i][mask].cpu()
            z = atom_nums[i][mask].cpu()

            dists   = torch.cdist(c.unsqueeze(0), c.unsqueeze(0))[0]
            nn_dist = dists[dists > 0].min().item()
            all_min_dists.append(nn_dist)

            mol  = Chem.RWMol()
            conf = Chem.Conformer(n_atoms)
            for j, (atomic_num, pos) in enumerate(zip(z.tolist(), c.tolist())):
                mol.AddAtom(Chem.Atom(int(atomic_num)))
                conf.SetAtomPosition(j, pos)
            mol.AddConformer(conf)

            try:
                rdDetermineBonds.DetermineConnectivity(mol)
                Chem.SanitizeMol(mol)
            except Exception:
                continue

            n_valid += 1

            if len(Chem.GetMolFrags(mol)) == 1:
                n_connected += 1
            frags.append(len(Chem.GetMolFrags(mol)))
            
            try:
                mw   = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                if 30 < mw < 500 and -3 < logp < 6:
                    n_realistic += 1
            except Exception:
                pass

        denom = max(len(all_atom_counts), 1)

        metrics = {
            "val/valid_ratio":       n_valid     / denom,
            "val/connected_ratio":   n_connected / denom,
            "val/realistic_ratio":   n_realistic / denom,
            "val/mean_atoms":        float(np.mean(all_atom_counts)) if all_atom_counts else 0.0,
            "val/mean_min_dist_A":   float(np.mean(all_min_dists))   if all_min_dists   else 0.0,
            "val/mean_number_frags": np.mean(frags).item() if frags else 0.0,
        }
        self.log_dict(metrics, on_epoch=True)

        print(
            f"\n── val epoch {self.current_epoch} ──  "
            + "  ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in metrics.items())
        )