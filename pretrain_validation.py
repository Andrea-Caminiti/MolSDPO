
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
from typing import Dict, Any


class DiffusionValidationCallback(Callback):
    """
    Callback to validate diffusion model during training.
    
    Automatically uses the DataModule's val_dataloader - no need to pass it manually!
    
    Usage:
        trainer = pl.Trainer(
            callbacks=[
                DiffusionValidationCallback(
                    vocab_enc2atom=vocab_enc2atom,
                    validate_every_n_steps=1000
                )
            ]
        )
        
        # Will automatically use datamodule.val_dataloader()
        trainer.fit(model, datamodule=module)
    """
    
    def __init__(
        self, 
        vocab_enc2atom: torch.Tensor,
        validate_every_n_steps: int = 1000,
        num_val_batches: int = 10,
        num_gen_samples: int = 50,
        use_wandb: bool = False
    ):
        """
        Args:
            vocab_enc2atom: Vocabulary tensor mapping encoding index to atomic number
            validate_every_n_steps: Run validation every N training steps
            num_val_batches: Number of validation batches to use
            num_gen_samples: Number of samples to generate for quality check
            use_wandb: Whether to log to Weights & Biases
        """
        super().__init__()
        self.vocab = vocab_enc2atom
        self.validate_every_n_steps = validate_every_n_steps
        self.num_val_batches = num_val_batches
        self.num_gen_samples = num_gen_samples
        self.use_wandb = use_wandb
        
        self._val_loader = None
        self._val_iter = None
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize validation iterator from DataModule."""
        if trainer.datamodule is not None:
            # Get validation dataloader from DataModule
            self._val_loader = trainer.datamodule.val_dataloader()
            self._val_iter = iter(self._val_loader)
            print(f"✓ Validation callback initialized with {len(self._val_loader)} batches")
        else:
            print("⚠️  Warning: No DataModule found - validation callback will be disabled")
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs, 
        batch, 
        batch_idx
    ):
        """Run validation checks periodically."""
        # Skip if no validation loader
        if self._val_loader is None:
            return
        
        # Skip if not at validation step
        if trainer.global_step % self.validate_every_n_steps != 0:
            return
        
        if trainer.global_step == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"Running validation at step {trainer.global_step}")
        print(f"{'='*60}")
        
        # Run quick validation
        metrics = self._quick_validation(pl_module)
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                pl_module.log(f'{key}', value, on_step=True, prog_bar=False)
        
    
    def _get_val_batch(self):
        """Get next validation batch, resetting iterator if needed."""
        try:
            return next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self._val_loader)
            return next(self._val_iter)
    
    def _quick_validation(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """Run quick validation checks."""
        pl_module.model.eval()
        device = pl_module.device
        
        metrics = {}
        
        # 1. Denoising quality at different timesteps
        with torch.no_grad():
            denoise_metrics = self._validate_denoising(pl_module)
            metrics.update(denoise_metrics)
        
        # 2. Generation quality (sample a few molecules)
        with torch.no_grad():
            gen_metrics = self._validate_generation(pl_module)
            metrics.update(gen_metrics)
        
        pl_module.model.train()
        return metrics
    
    def _validate_denoising(self, pl_module: pl.LightningModule) -> Dict[str, float]:
        """Validate denoising at different timesteps."""
        device = pl_module.device
        losses = {10: [], 100: [], 500: [], 900: []}
        
        for i in range(min(self.num_val_batches, 5)):
            coords, atom_types = self._get_val_batch()
            
            coords = coords.to(device)
            atom_types = atom_types.to(device)
            B = coords.shape[0]
            
            for t_val in losses.keys():
                t = torch.full((B, 1), t_val, device=device).long()
                
                # Add noise
                noise_coord = torch.randn_like(coords)
                noise_atom = torch.randn_like(atom_types.float())
                
                alpha_t = pl_module.scheduler.alphas_cumprod[t_val]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                
                noisy_coords = sqrt_alpha_t * coords + sqrt_one_minus_alpha_t * noise_coord
                noisy_atoms = sqrt_alpha_t * atom_types + sqrt_one_minus_alpha_t * noise_atom
                
                # Predict
                coord_pred, atom_pred = pl_module.model(noisy_atoms, noisy_coords, t.squeeze())
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(coord_pred, noise_coord)
                losses[t_val].append(loss.item())
        
        # Average and check monotonicity
        avg_losses = {f'denoise_loss_t{t}': np.mean(l) for t, l in losses.items()}
        
        # Check if monotonic (should decrease with timestep)
        timesteps = [10, 100, 500, 900]
        is_monotonic = all(
            avg_losses[f'denoise_loss_t{timesteps[i]}'] >= avg_losses[f'denoise_loss_t{timesteps[i+1]}']
            for i in range(len(timesteps)-1)
        )
        avg_losses['denoise_monotonic'] = float(is_monotonic)
        
        return avg_losses
    
    def _validate_generation(self, pl_module: pl.LightningModule) -> Dict[str, float]:
        """Generate samples and check validity."""
        device = pl_module.device
        
        # Generate a small batch
        batch_size = min(10, self.num_gen_samples)
        coords = torch.randn(batch_size, 29, 3, device=device)
        atom_types = torch.randn(batch_size, 29, 6, device=device)
        
        # DDIM sampling (fast, 10 steps)
        pl_module.scheduler.set_timesteps(10, device=device)
        
        for t in pl_module.scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device).long()
            
            coord_pred, atom_pred = pl_module.model(atom_types, coords, t_batch)
            
            # DDIM step
            alpha_t = pl_module.scheduler.alphas_cumprod[t]
            alpha_t_prev = pl_module.scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            x0_coords = (coords - torch.sqrt(1 - alpha_t) * coord_pred) / torch.sqrt(alpha_t)
            x0_atoms = (atom_types - torch.sqrt(1 - alpha_t) * atom_pred) / torch.sqrt(alpha_t)
            
            coords = torch.sqrt(alpha_t_prev) * x0_coords + torch.sqrt(1 - alpha_t_prev) * coord_pred
            atom_types = torch.sqrt(alpha_t_prev) * x0_atoms + torch.sqrt(1 - alpha_t_prev) * atom_pred
        
        # Analyze samples
        valid_count = 0
        atom_counts = []
        
        for i in range(batch_size):
            atom_logits = atom_types[i]
            atom_indices = atom_logits.argmax(dim=-1)
            atom_nums = self.vocab[atom_indices]
            
            mask = atom_nums != 0
            num_atoms = mask.sum().item()
            
            if num_atoms > 0:
                valid_count += 1
                atom_counts.append(num_atoms)
        
        return {
            'gen_valid_ratio': valid_count / batch_size,
            'gen_avg_atoms': np.mean(atom_counts) if atom_counts else 0.0
        }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print validation metrics."""
        print("\nValidation Metrics:")
        print("-" * 60)
        
        # Denoising metrics
        print("Denoising Quality:")
        for t in [10, 100, 500, 900]:
            key = f'denoise_loss_t{t}'
            if key in metrics:
                print(f"  t={t:4d}: {metrics[key]:.6f}")
        
        if 'denoise_monotonic' in metrics:
            status = "✓ PASS" if metrics['denoise_monotonic'] else "✗ FAIL"
            print(f"  Monotonic: {status}")
        
        # Generation metrics
        print("\nGeneration Quality:")
        if 'gen_valid_ratio' in metrics:
            print(f"  Valid ratio: {metrics['gen_valid_ratio']:.1%}")
        if 'gen_avg_atoms' in metrics:
            print(f"  Avg atoms: {metrics['gen_avg_atoms']:.1f}")
        
        print("-" * 60)


class EarlyStoppingOnValidation(Callback):
    """
    Early stopping based on validation metrics.
    
    Stops training if:
    1. Denoising is not monotonic for too long
    2. Generation quality drops significantly
    
    Usage:
        trainer = pl.Trainer(
            callbacks=[
                DiffusionValidationCallback(vocab_enc2atom=vocab),
                EarlyStoppingOnValidation(patience=5)
            ]
        )
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_valid_ratio: float = 0.3
    ):
        """
        Args:
            patience: Number of failed validations before stopping
            min_valid_ratio: Minimum generation validity ratio
        """
        super().__init__()
        self.patience = patience
        self.min_valid_ratio = min_valid_ratio
        self.bad_count = 0
        self.best_loss = float('inf')
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx
    ):
        """Check validation metrics and decide whether to stop."""
        # Only check when validation callback has run
        if trainer.global_step % 1000 != 0:  # Should match validation frequency
            return
        
        # Get latest logged metrics
        logged_metrics = trainer.callback_metrics
        
        # Check monotonicity
        if 'denoise_monotonic' in logged_metrics:
            if logged_metrics['denoise_monotonic'].item() < 0.5:  # Not monotonic
                self.bad_count += 1
                print(f"⚠️  Warning: Denoising not monotonic ({self.bad_count}/{self.patience})")
            else:
                # Reset if monotonic
                if 'gen_valid_ratio' in logged_metrics:
                    if logged_metrics['gen_valid_ratio'].item() >= self.min_valid_ratio:
                        self.bad_count = 0
        
        # Check generation quality
        if 'gen_valid_ratio' in logged_metrics:
            if logged_metrics['gen_valid_ratio'].item() < self.min_valid_ratio:
                self.bad_count += 1
                print(f"⚠️  Warning: Low generation quality ({self.bad_count}/{self.patience})")
        
        # Stop if too many bad validations
        if self.bad_count >= self.patience:
            print(f"\n🛑 Stopping training: validation failed {self.patience} times")
            trainer.should_stop = True