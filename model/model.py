import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Optional
from model.util import TimeEmbed
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

class RMSNorm(nn.Module):
    """Optimized RMSNorm implementation with fused operations."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
        # Fused normalization for better performance
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


# ---------- Transformer block (pre-LN) ----------

class SimpleTransformerBlock(nn.Module):
    """Optimized transformer block with efficient attention."""
    
    def __init__(self, dim: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Efficient MLP with fused operations
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
        """
        Optimized forward pass with optional batched mode.
        
        Args:
            x: Input tensor [B, N, D] or [B, T, N, D] if batched
            attn_bias: Optional attention bias
            batched: Whether input has time dimension
        """
        if batched:
            B, T, N, D = x.shape
            # Normalize and project
            x_ln = self.norm1(x, batched)
            qkv = self.qkv(x_ln).reshape(B, T, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, T, heads, N, head_dim]
            q, k, v = qkv.unbind(0)
            
            # Scaled dot-product attention
            attn = torch.einsum("bthid,bthjd->bthij", q, k) * self.scale
            
            if attn_bias is not None:
                attn = attn + attn_bias
            
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            out = torch.einsum("bthij,bthjd->bthid", attn, v)
            out = out.reshape(B, T, N, D)
        else:
            B, N, D = x.shape
            # Normalize and project
            x_ln = self.norm1(x)
            qkv = self.qkv(x_ln).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
            q, k, v = qkv.unbind(0)
            
            # Scaled dot-product attention
            attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
            
            if attn_bias is not None:
                attn = attn + attn_bias
            
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            out = torch.einsum("bhij,bhjd->bhid", attn, v)
            out = out.reshape(B, N, D)

        # Residual connections
        x = x + self.out(out)
        x = x + self.ff(self.norm2(x, batched) if batched else self.norm2(x))
        return x


# ---------- Cross-attention block ----------

class SimpleCrossAttentionTransformerBlock(nn.Module):
    """Optimized cross-attention block."""
    
    def __init__(self, dim_q: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.dim_q = dim_q
        self.n_heads = n_heads
        self.head_dim = dim_q // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim_q, dim_q, bias=False)
        self.kv_proj = nn.Linear(dim_q, dim_q * 2, bias=False)
        self.out = nn.Linear(dim_q, dim_q)
        
        self.norm_q = RMSNorm(dim_q) if use_rmsnorm else nn.LayerNorm(dim_q)
        self.norm_kv = RMSNorm(dim_q) if use_rmsnorm else nn.LayerNorm(dim_q)
        
        self.ff = nn.Sequential(
            nn.Linear(dim_q, dim_q * mlp_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * mlp_mult, dim_q),
            nn.Dropout(dropout)
        )

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, batched: bool = False) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            q_x: Query tensor [B, Nq, D] or [B, T, Nq, D]
            kv_x: Key-value tensor [B, Nk, D] or [B, T, Nk, D]
            attn_bias: Optional attention bias
            batched: Whether input has time dimension
        """
        if batched:
            B, T, Nq, Dq = q_x.shape
            Nk = kv_x.shape[2]
            
            # Project queries
            q = self.q_proj(self.norm_q(q_x, batched))
            q = q.reshape(B, T, Nq, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            
            # Project keys and values
            kv = self.kv_proj(self.norm_kv(kv_x, batched))
            kv = kv.reshape(B, T, Nk, 2, self.n_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
            k, v = kv.unbind(0)
            
            # Attention
            attn = torch.einsum("bthqd,bthkd->bthqk", q, k) * self.scale
            
            if attn_bias is not None:
                attn = attn + attn_bias
                
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum("bthqk,bthkd->bthqd", attn, v)
            out = out.permute(0, 1, 3, 2, 4).reshape(B, T, Nq, Dq)
        else:
            B, Nq, Dq = q_x.shape
            Nk = kv_x.shape[1]
            
            # Project queries
            q = self.q_proj(self.norm_q(q_x))
            q = q.reshape(B, Nq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Project keys and values
            kv = self.kv_proj(self.norm_kv(kv_x))
            kv = kv.reshape(B, Nk, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            
            # Attention
            attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
            
            if attn_bias is not None:
                attn = attn + attn_bias
                
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
            out = out.permute(0, 2, 1, 3).reshape(B, Nq, Dq)
        
        # Residual connections
        x = q_x + self.out(out)
        x = x + self.ff(x)
        return x


# ---------- TabascoV2 ----------

class TabascoV2(nn.Module):
    """Optimized TabascoV2 model with improved efficiency."""
    
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
        
        # Atom and coordinate embeddings with reduced layers
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
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Time embedding
        self.time_emb = TimeEmbed(d_model // 4)
        self.time_proj = nn.Linear(d_model // 4, d_model)

        # Transformer stack
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm)
            for _ in range(n_layers)
        ])

        # Pairwise RBF for geometry-aware attention
        self.rbf_dim = pair_rbf_centers
        self.pair_rbf = PairwiseRBF(num_rbf=pair_rbf_centers)
        self.rbf_to_bias = nn.Linear(pair_rbf_centers, n_heads, bias=False)

        # Cross-attention blocks
        self.crossAtomCoord = SimpleCrossAttentionTransformerBlock(
            half, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm
        )
        self.crossCoordAtom = SimpleCrossAttentionTransformerBlock(
            half, n_heads, dropout=dropout, use_rmsnorm=use_rmsnorm
        )

        # Output heads with reduced hidden size
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
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, atom_idxs: torch.Tensor, coords: torch.Tensor, t: torch.Tensor, batched: bool = False):
        """
        Forward pass with optional batched mode for efficiency.
        
        Args:
            atom_idxs: [B, N, vocab_size] one-hot or logits
            coords: [B, N, 3] or [B, T, N, 3] coordinates
            t: [B] or [B*T] timesteps
            batched: Whether inputs have time dimension
            
        Returns:
            coord_pred: Predicted coordinate noise/update
            type_logits: Predicted atom type logits
        """
        # Embed atoms and coordinates
        atom_e = self.atom_emb(atom_idxs)  # [B, N, half] or [B, T, N, half]
        coord_e = self.coord_proj(coords)  # [B, N, half] or [B, T, N, half]
        
        # Concatenate and project
        x = torch.cat([atom_e, coord_e], dim=-1)  # [B, N, d_model] or [B, T, N, d_model]
        x = self.input_proj(x)
        
        # Add time embedding
        if batched:
            B, T = coords.shape[0], coords.shape[1]
            temb = self.time_proj(self.time_emb(t))  # [B*T, d_model]
            temb = temb.view(B, T, -1).unsqueeze(2)  # [B, T, 1, d_model]
        else:
            # For normal mode, t shape is [B]
            temb = self.time_proj(self.time_emb(t))  # [B, d_model]
            temb = temb.unsqueeze(1)  # [B, 1, d_model]
            
        x = x + temb
        
        # Create pairwise geometric bias
        rbf = self.pair_rbf(coords, batched)  # [B, N, N, num_rbf] or [B, T, N, N, num_rbf]
        
        # Aggregate RBF features and project to attention bias
        if batched:
            rbf_pool = rbf.mean(dim=(2, 3))  # [B, T, num_rbf]
            head_bias = self.rbf_to_bias(rbf_pool)  # [B, T, n_heads]
            attn_bias = head_bias[:, :, :, None, None]  # [B, T, n_heads, 1, 1]
        else:
            rbf_pool = rbf.mean(dim=(1, 2))  # [B, num_rbf]
            head_bias = self.rbf_to_bias(rbf_pool)  # [B, n_heads]
            attn_bias = head_bias[:, :, None, None]  # [B, n_heads, 1, 1]

        # Transformer blocks with geometric bias
        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias, batched=batched)
        
        # Split into atom and coord streams
        half = self.d_model // 2
        a = x[..., :half]
        c = x[..., half:]

        # Cross-attention refinement
        ac = self.crossAtomCoord(a, c, batched=batched)
        ca = self.crossCoordAtom(c, a, batched=batched)
        
        # Recombine
        x = torch.cat([ac, ca], dim=-1)
        
        # Output heads
        coord_pred = self.coord_head(x)
        type_logits = self.type_head(x)
        
        return coord_pred, type_logits


class LightningTabasco(pl.LightningModule):
    """Lightning module for training TabascoV2."""
    
    def __init__(self, args, vocab_enc2atom: torch.Tensor):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.vocab_size = args.vocab_size
        
        # Scheduler
        self.scheduler = DDIMScheduler.from_config(DDIM_config)
        
        # Model
        self.model = TabascoV2(
            atom_vocab_size=self.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            pos_coord_dim=128,
            pair_rbf_centers=args.d_model // 2,
            dropout=0.1
        )
        
        # Loss history for importance sampling
        self.register_buffer("loss_history", torch.ones(args.T) * 1000.0)
        self.register_buffer("sampling_probs", torch.ones(args.T) / args.T)
        self.alpha_ema = 0.01  # EMA decay for loss history
        self.register_buffer("vocab_enc2atom", vocab_enc2atom)  # ← add this line

    @torch.no_grad()
    def update_loss_history(self, t: torch.Tensor, losses: torch.Tensor):
        """Update loss history with exponential moving average."""
        for i in range(len(t)):
            step_t = t[i].item()
            self.loss_history[step_t] = (
                (1 - self.alpha_ema) * self.loss_history[step_t] + 
                self.alpha_ema * losses[i].item()
            )
        
        # Update sampling probabilities
        self.sampling_probs.copy_(self.get_sampling_weights())

    def get_sampling_weights(self) -> torch.Tensor:
        """Compute importance sampling weights based on loss history."""
        weights = torch.sqrt(self.loss_history.clamp(min=0)) + 1e-6
        return weights / weights.sum()
    
    def configure_optimizers(self):
        """Configure optimizer with layer-wise learning rates."""
        base_lr = self.args.lr
        
        # Group parameters by component with different learning rates
        param_groups = [
            {"params": self.model.atom_emb.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.coord_proj.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.input_proj.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.time_emb.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.time_proj.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.pair_rbf.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.rbf_to_bias.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.blocks.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.crossAtomCoord.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.crossCoordAtom.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.coord_head.parameters(), "lr": base_lr, "weight_decay": 0.01},
            {"params": self.model.type_head.parameters(), "lr": base_lr, "weight_decay": 0.01},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_warmup_lr(step, 5000, 200000)
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
        coords, atom_types = batch
        B = coords.shape[0]
        
        # Importance sampling of timesteps
        t = torch.multinomial(
            self.sampling_probs, 
            B, 
            replacement=True
        ).to(self.device)
        
        # Compute loss
        loss, metrics = p_losses_joint_absorb_improved_efficient(
            self.model, coords, atom_types, t, 
            self.scheduler, self.device, lambda_type=1.0
        )
        
        # Importance weighting
        p_t = self.sampling_probs[t.squeeze()]
        weights = 1.0 / (p_t * self.args.T)
        
        # Weighted losses
        loss_weighted = (loss * weights).mean()
        
        # Update loss history
        self.update_loss_history(t.squeeze(), loss.detach())
        
        # Logging
        self.log_dict({
            'loss': loss_weighted,
            **metrics
        }, on_step=True, prog_bar=False)
        
        return loss_weighted
    def validation_step(self, batch, batch_idx):
        """
        Per-batch validation: denoising loss at 4 canonical timesteps.

        Lightning calls this for every batch in val_dataloader automatically.
        Results are averaged over the epoch and logged as val/denoise_loss_t*.
        The t=100 value is monitored by ModelCheckpoint.
        """
        coords, atom_types = batch          # coords: [B,N,3]  atom_types: [B,N,6]
        B      = coords.shape[0]
        device = self.device

        with torch.no_grad():
            for t_val in [10, 100, 500, 900]:
                t = torch.full((B, 1), t_val, device=device).long()

                alpha      = self.scheduler.alphas_cumprod[t_val]
                sqrt_a     = alpha ** 0.5
                sqrt_one_a = (1.0 - alpha) ** 0.5

                noise_c = torch.randn_like(coords)
                noise_a = torch.randn_like(atom_types)

                noisy_coords = sqrt_a * coords    + sqrt_one_a * noise_c
                noisy_atoms  = sqrt_a * atom_types + sqrt_one_a * noise_a

                coord_pred, atom_pred = self.model(noisy_atoms, noisy_coords, t.squeeze())

                loss = (
                    F.mse_loss(coord_pred, noise_c)
                    + F.mse_loss(atom_pred, noise_a)
                )

                # on_epoch=True → Lightning averages over all val batches
                self.log(
                    f"val/denoise_loss_t{t_val}", loss,
                    on_step=False, on_epoch=True,
                    prog_bar=(t_val == 100), sync_dist=True,
                )


    def on_validation_epoch_end(self):
        """
        End-of-epoch generation check.

        Generates 32 molecules via 10-step DDIM, rescales coordinates to
        Angstroms, runs DetermineConnectivity + SanitizeMol, and logs:

        val/valid_ratio      — fraction passing SanitizeMol
        val/connected_ratio  — fraction that are a single fragment
        val/realistic_ratio  — fraction with drug-like MW / logP
        val/mean_atoms       — average heavy atom count
        val/mean_min_dist_A  — average nearest-neighbour distance in Å
                                (should be 1.4–1.6 Å for a converged model)
        """
        NUM_SAMPLES  = 32
        DDIM_STEPS   = 10
        COORD_SCALE  = 2.2     # dataloader divided by this; we multiply back
        device       = self.device

        # ------------------------------------------------------------------
        # 1. DDIM sampling from pure noise
        # ------------------------------------------------------------------
        self.model.eval()

        coords = torch.randn(NUM_SAMPLES, 29, 3,              device=device)
        atoms  = torch.randn(NUM_SAMPLES, 29, self.vocab_size, device=device)

        self.scheduler.set_timesteps(DDIM_STEPS, device=device)

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_batch = torch.full((NUM_SAMPLES,), t, device=device).long()

                coord_pred, atom_pred = self.model(atoms, coords, t_batch)

                alpha_t      = self.scheduler.alphas_cumprod[t].to(device)
                alpha_t_prev = (
                    self.scheduler.alphas_cumprod[t - 1].to(device)
                    if t > 0
                    else torch.ones(1, device=device)
                )

                # Predict x0 then step back
                x0_c = (coords - (1 - alpha_t).sqrt() * coord_pred) / alpha_t.sqrt()
                x0_a = (atoms  - (1 - alpha_t).sqrt() * atom_pred)  / alpha_t.sqrt()

                coords = alpha_t_prev.sqrt() * x0_c + (1 - alpha_t_prev).sqrt() * coord_pred
                atoms  = alpha_t_prev.sqrt() * x0_a + (1 - alpha_t_prev).sqrt() * atom_pred

        # ------------------------------------------------------------------
        # 2. Decode atom types; apply padding mask; rescale to Angstroms
        # ------------------------------------------------------------------
        atom_indices = atoms.argmax(dim=-1)               # [B, N]
        atom_nums    = self.vocab_enc2atom[atom_indices]  # [B, N]  (atomic numbers)

        n_valid = n_connected = n_realistic = 0
        all_atom_counts = []
        all_min_dists   = []

        for i in range(NUM_SAMPLES):
            mask = atom_nums[i] != 0          # True for non-padding atoms
            n_atoms = mask.sum().item()
            if n_atoms < 2:
                continue

            all_atom_counts.append(n_atoms)

            # Rescale to Angstroms
            c = coords[i][mask].cpu() * COORD_SCALE    # [M, 3]
            z = atom_nums[i][mask].cpu()               # [M]

            # Nearest-neighbour distance (excludes self via >0 filter)
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0))[0]
            nn_dist = dists[dists > 0].min().item()
            all_min_dists.append(nn_dist)

            # --------------------------------------------------------------
            # 3. Build RDKit mol and check chemistry
            # --------------------------------------------------------------
            mol  = Chem.RWMol()
            conf = Chem.Conformer(n_atoms)
            for j, (atomic_num, pos) in enumerate(zip(z.tolist(), c.tolist())):
                mol.AddAtom(Chem.Atom(int(atomic_num)))
                conf.SetAtomPosition(j, pos)
            mol.AddConformer(conf)

            try:
                # DetermineConnectivity uses covalent radii + 3D geometry.
                # Must be called before SanitizeMol.
                rdDetermineBonds.DetermineConnectivity(mol)
                Chem.SanitizeMol(mol)
            except Exception:
                continue

            n_valid += 1

            if len(Chem.GetMolFrags(mol)) == 1:
                n_connected += 1

            try:
                mw   = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                if 30 < mw < 500 and -3 < logp < 6:
                    n_realistic += 1
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 4. Log
        # ------------------------------------------------------------------
        denom = max(len(all_atom_counts), 1)

        metrics = {
            "val/valid_ratio":      n_valid     / denom,
            "val/connected_ratio":  n_connected / denom,
            "val/realistic_ratio":  n_realistic / denom,
            "val/mean_atoms":       float(np.mean(all_atom_counts)) if all_atom_counts else 0.0,
            "val/mean_min_dist_A":  float(np.mean(all_min_dists))   if all_min_dists   else 0.0,
        }
        self.log_dict(metrics, on_epoch=True)

        print(
            f"\n── val epoch {self.current_epoch} ──  "
            + "  ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in metrics.items())
        )
