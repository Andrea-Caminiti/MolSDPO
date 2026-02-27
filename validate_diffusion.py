"""
Validation script for molecular diffusion model pretraining.

This script provides multiple validation metrics to ensure your diffusion model
is learning properly before moving to RL fine-tuning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdDetermineBonds
from tqdm import tqdm
from diffusers import DDIMScheduler
import torch
import argparse
from model.model import TabascoV2
from data.dataloader import build_qm9_dataloader
from config import DDIM_config

RDLogger.DisableLog('rdApp.*')  



class DiffusionValidator:
    """Comprehensive validation for molecular diffusion models."""
    
    def __init__(self, model, scheduler, vocab_enc2atom, device='cuda'):
        self.model = model
        self.scheduler = scheduler
        self.vocab = vocab_enc2atom
        self.device = device
        
    def validate_all(self, val_loader, num_samples=100, save_dir='validation'):
        """Run all validation metrics."""
        print("="*80)
        print("DIFFUSION MODEL VALIDATION")
        print("="*80)
        
        results = {}
        # 1. Denoising Quality
        print("\n[1/7] Validating denoising quality...")
        results['denoising'] = self.validate_denoising(val_loader)
        
        # 2. Reconstruction Error
        print("\n[2/7] Validating reconstruction...")
        results['reconstruction'] = self.validate_reconstruction(val_loader)
        
        # 3. Generation Quality
        print("\n[3/7] Validating generation quality...")
        results['generation'] = self.validate_generation(num_samples)
        
        # 4. Timestep Consistency
        print("\n[4/7] Validating timestep consistency...")
        results['timestep'] = self.validate_timestep_consistency(val_loader)
        
        # 5. Diversity
        #print("\n[5/7] Validating diversity...")
        #results['diversity'] = self.validate_diversity(num_samples)
        
        # 6. Chemical Validity
        print("\n[6/7] Validating chemical validity...")
        results['chemistry'] = self.validate_chemistry(num_samples)
        
        # 7. Distribution Matching
        print("\n[7/7] Validating distribution matching...")
        results['distribution'] = self.validate_distribution(val_loader, num_samples)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def validate_denoising(self, val_loader, num_batches=10):
        """
        Validate that the model can denoise at different timesteps.
        
        Key insight: Loss should decrease with timestep (easier to denoise at small t).
        """
        self.model.eval()
        losses_by_timestep = {t: [] for t in [10, 100, 500, 900]}
        
        with torch.no_grad():
            for i, (coords, atom_types) in enumerate(val_loader):
                if i >= num_batches:
                    break
                    
                coords = coords.to(self.device)
                atom_types = atom_types.to(self.device)
                B = coords.shape[0]
                
                for t_val in losses_by_timestep.keys():
                    t = torch.full((B, 1), t_val, device=self.device).long()
                    
                    # Add noise
                    noise_coord = torch.randn_like(coords)
                    noise_atom = torch.randn_like(atom_types.float())
                    
                    alpha_t = self.scheduler.alphas_cumprod[t_val]
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                    
                    noisy_coords = sqrt_alpha_t * coords + sqrt_one_minus_alpha_t * noise_coord
                    noisy_atoms = sqrt_alpha_t * atom_types + sqrt_one_minus_alpha_t * noise_atom
                    
                    # Predict
                    coord_pred, atom_pred = self.model(noisy_atoms, noisy_coords, t.squeeze())
                    
                    # Compute loss
                    loss_coord = F.mse_loss(coord_pred, noise_coord)
                    loss_atom = F.mse_loss(atom_pred, noise_atom)
                    loss = loss_coord + loss_atom
                    
                    losses_by_timestep[t_val].append(loss.item())
        
        # Average losses
        avg_losses = {t: np.mean(losses) for t, losses in losses_by_timestep.items()}
        
        # Check monotonicity: loss should decrease as t decreases
        timesteps = sorted(avg_losses.keys())
        is_monotonic = all(avg_losses[timesteps[i]] >= avg_losses[timesteps[i+1]] 
                          for i in range(len(timesteps)-1))
        
        print(f"  Losses by timestep: {avg_losses}")
        print(f"  ✓ Monotonic: {is_monotonic}" if is_monotonic else f"  ✗ NOT monotonic (BAD)")
        
        return {
            'losses': avg_losses,
            'monotonic': is_monotonic,
            'pass': is_monotonic
        }
    
    def validate_reconstruction(self, val_loader, num_batches=5):
        """
        Test if model can reconstruct clean samples from slightly noised ones.
        
        Key insight: At low noise levels (t=10-50), model should recover original well.
        """
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for i, (coords, atom_types) in enumerate(val_loader):
                if i >= num_batches:
                    break
                    
                coords = coords.to(self.device)
                atom_types = atom_types.to(self.device)
                
                # Use DDIM to denoise
                reconstructed = self._ddim_reconstruct(coords, atom_types, num_steps=25)
                
                # Compute error
                error_coord = F.mse_loss(reconstructed[0], coords)
                error_atom = F.mse_loss(reconstructed[1], atom_types.float())
                
                reconstruction_errors.append(error_coord.item() + error_atom.item())
        
        avg_error = np.mean(reconstruction_errors)
        threshold = 0.1  # Adjust based on your scale
        
        print(f"  Average reconstruction error: {avg_error:.6f}")
        print(f"  ✓ Pass (< {threshold})" if avg_error < threshold else f"  ✗ High error (BAD)")
        
        return {
            'error': avg_error,
            'threshold': threshold,
            'pass': avg_error < threshold
        }
    
    def validate_generation(self, num_samples=100):
        """
        Generate samples from pure noise and check basic properties.
        
        Key insight: Generated molecules should have reasonable properties.
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples // 10):
                # Start from noise
                coords = torch.randn(10, 29, 3, device=self.device)
                atom_types = torch.randn(10, 29, 6, device=self.device)
                
                # DDIM sampling
                samples_batch = self._ddim_sample(coords, atom_types, num_steps=25)
                samples.extend(samples_batch)
        
        # Analyze samples
        stats = self._analyze_samples(samples)
        
        # Check criteria
        valid_ratio = stats['valid_ratio']
        unique_ratio = stats['unique_ratio']
        
        print(f"  Valid molecules: {valid_ratio:.1%}")
        print(f"  Unique molecules: {unique_ratio:.1%}")
        print(f"  Avg atoms: {stats['avg_atoms']:.1f}")
        print(f"  ✓ Pass" if valid_ratio > 0.5 else f"  ✗ Low validity (BAD)")
        
        return {
            **stats,
            'pass': valid_ratio > 0.5
        }
    
    def validate_timestep_consistency(self, val_loader, num_batches=5):
        """
        Check that predictions are consistent across timesteps.
        
        Key insight: x0 prediction should be similar for nearby timesteps.
        """
        self.model.eval()
        consistency_scores = []
        
        with torch.no_grad():
            for i, (coords, atom_types) in enumerate(val_loader):
                if i >= num_batches:
                    break
                    
                coords = coords.to(self.device)
                atom_types = atom_types.to(self.device)
                B = coords.shape[0]
                
                # Test pairs of nearby timesteps
                for t_val in [100, 300, 500, 700]:
                    t1 = torch.full((B, 1), t_val, device=self.device).long()
                    t2 = torch.full((B, 1), t_val + 10, device=self.device).long()
                    
                    # Add same noise to both
                    noise_coord = torch.randn_like(coords)
                    noise_atom = torch.randn_like(atom_types.float())
                    
                    # Noise at t1
                    alpha_t1 = self.scheduler.alphas_cumprod[t_val]
                    noisy_coords_1 = torch.sqrt(alpha_t1) * coords + torch.sqrt(1 - alpha_t1) * noise_coord
                    noisy_atoms_1 = torch.sqrt(alpha_t1) * atom_types + torch.sqrt(1 - alpha_t1) * noise_atom
                    
                    # Noise at t2
                    alpha_t2 = self.scheduler.alphas_cumprod[t_val + 10]
                    noisy_coords_2 = torch.sqrt(alpha_t2) * coords + torch.sqrt(1 - alpha_t2) * noise_coord
                    noisy_atoms_2 = torch.sqrt(alpha_t2) * atom_types + torch.sqrt(1 - alpha_t2) * noise_atom
                    
                    # Predict x0 from both
                    coord_pred_1, atom_pred_1 = self.model(noisy_atoms_1, noisy_coords_1, t1.squeeze())
                    coord_pred_2, atom_pred_2 = self.model(noisy_atoms_2, noisy_coords_2, t2.squeeze())
                    
                    # Compute x0 estimates
                    x0_coords_1 = (noisy_coords_1 - torch.sqrt(1 - alpha_t1) * coord_pred_1) / torch.sqrt(alpha_t1)
                    x0_coords_2 = (noisy_coords_2 - torch.sqrt(1 - alpha_t2) * coord_pred_2) / torch.sqrt(alpha_t2)
                    
                    # Consistency: x0 estimates should be similar
                    consistency = F.cosine_similarity(
                        x0_coords_1.flatten(1), 
                        x0_coords_2.flatten(1), 
                        dim=1
                    ).mean().item()
                    
                    consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        threshold = 0.95  # Should be very consistent
        
        print(f"  Average consistency: {avg_consistency:.4f}")
        print(f"  ✓ Pass (> {threshold})" if avg_consistency > threshold else f"  ✗ Inconsistent (BAD)")
        
        return {
            'consistency': avg_consistency,
            'threshold': threshold,
            'pass': avg_consistency > threshold
        }
    
    def validate_diversity(self, num_samples=100):
        """
        Check that generated samples are diverse.
        
        Key insight: Model shouldn't collapse to generating same molecule.
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples // 10), desc="  Generating"):
                coords = torch.randn(10, 29, 3, device=self.device)
                atom_types = torch.randn(10, 29, 6, device=self.device)
                
                samples_batch = self._ddim_sample(coords, atom_types, num_steps=25)
                samples.extend(samples_batch)
        
        # Compute pairwise distances
        num_valid = len([s for s in samples if s is not None])
        
        if num_valid < 2:
            print(f"  ✗ Too few valid samples")
            return {'pass': False, 'diversity': 0.0}
        
        # Compare coordinates
        coords_list = [s[0] for s in samples if s is not None]
        
        # Compute average pairwise distance
        total_dist = 0
        count = 0
        for i in range(len(coords_list)):
            for j in range(i+1, min(i+10, len(coords_list))):  # Compare with 10 neighbors
                print(coords_list[i].shape, coords_list[j].shape)
                dist = torch.norm(coords_list[i] - coords_list[j]).item()
                total_dist += dist
                count += 1
        
        avg_diversity = total_dist / count if count > 0 else 0
        threshold = 5.0  # Should be reasonably different
        
        print(f"  Average pairwise distance: {avg_diversity:.2f}")
        print(f"  ✓ Diverse" if avg_diversity > threshold else f"  ✗ Low diversity (mode collapse?)")
        
        return {
            'diversity': avg_diversity,
            'threshold': threshold,
            'pass': avg_diversity > threshold
        }
    
    def validate_chemistry(self, num_samples=100):
        """
        Validate chemical properties of generated molecules.
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples // 10), desc="  Generating"):
                coords = torch.randn(10, 29, 3, device=self.device)
                atom_types = torch.randn(10, 29, 6, device=self.device)
                
                samples_batch = self._ddim_sample(coords, atom_types, num_steps=1000)
                samples.extend(samples_batch)
        
        # Convert to RDKit molecules
        mols = []
        for sample in samples:
            if sample is not None:
                mol = self._to_rdkit_mol(sample)
                if mol is not None:
                    mols.append(mol)
        
        if len(mols) == 0:
            print(f"  ✗ No valid RDKit molecules")
            return {'pass': False}
        
        # Analyze properties
        valid_count = 0
        connected_count = 0
        realistic_count = 0
        
        for mol in mols:
            # Check validity
            try:
                Chem.SanitizeMol(mol)
                valid_count += 1
            except:
                continue
            
            # Check connectivity
            if len(Chem.GetMolFrags(mol)) == 1:
                connected_count += 1
            
            # Check realistic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            if 50 < mw < 500 and -3 < logp < 5:
                realistic_count += 1
        
        valid_ratio = valid_count / len(mols)
        connected_ratio = connected_count / len(mols)
        realistic_ratio = realistic_count / len(mols)
        
        print(f"  Valid (sanitized): {valid_ratio:.1%}")
        print(f"  Connected: {connected_ratio:.1%}")
        print(f"  Realistic properties: {realistic_ratio:.1%}")
        
        pass_threshold = valid_ratio > 0.3 and connected_ratio > 0.2
        print(f"  ✓ Pass" if pass_threshold else f"  ✗ Poor chemistry")
        
        return {
            'valid_ratio': valid_ratio,
            'connected_ratio': connected_ratio,
            'realistic_ratio': realistic_ratio,
            'pass': pass_threshold
        }
    
    def validate_distribution(self, val_loader, num_samples=100):
        """
        Check if generated distribution matches training distribution.
        
        Key insight: Generated samples should have similar statistics to training data.
        """
        self.model.eval()
        
        # Get statistics from validation set
        real_stats = self._compute_distribution_stats(val_loader, num_samples)
        
        # Generate samples
        samples = []
        with torch.no_grad():
            for _ in tqdm(range(num_samples // 10), desc="  Generating"):
                coords = torch.randn(10, 29, 3, device=self.device)
                atom_types = torch.randn(10, 29, 6, device=self.device)
                
                samples_batch = self._ddim_sample(coords, atom_types, num_steps=25)
                samples.extend(samples_batch)
        
        gen_stats = self._compute_sample_stats(samples)
        
        # Compare distributions
        atom_count_diff = abs(real_stats['avg_atoms'] - gen_stats['avg_atoms'])
        bond_length_diff = abs(real_stats['avg_bond_length'] - gen_stats['avg_bond_length'])
        
        print(f"  Real avg atoms: {real_stats['avg_atoms']:.1f}, Generated: {gen_stats['avg_atoms']:.1f}")
        print(f"  Real avg bond length: {real_stats['avg_bond_length']:.3f}, Generated: {gen_stats['avg_bond_length']:.3f}")
        
        pass_criteria = atom_count_diff < 3 and bond_length_diff < 0.3
        print(f"  ✓ Distributions match" if pass_criteria else f"  ✗ Distribution mismatch")
        
        return {
            'real_stats': real_stats,
            'gen_stats': gen_stats,
            'atom_count_diff': atom_count_diff,
            'bond_length_diff': bond_length_diff,
            'pass': pass_criteria
        }
    
    # ==================== Helper Methods ====================
    
    def _ddim_sample(self, coords, atom_types, num_steps=25):
        """Generate samples using DDIM."""
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        for t in self.scheduler.timesteps:
            t_batch = torch.full((coords.shape[0],), t, device=self.device).long()
            
            with torch.no_grad():
                coord_pred, atom_pred = self.model(atom_types, coords, t_batch)
            
            # DDIM step
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_prev = self.scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            # Predict x0
            x0_coords = (coords - torch.sqrt(1 - alpha_t) * coord_pred) / torch.sqrt(alpha_t)
            x0_atoms = (atom_types - torch.sqrt(1 - alpha_t) * atom_pred) / torch.sqrt(alpha_t)
            
            # DDIM update
            coords = torch.sqrt(alpha_t_prev) * x0_coords + torch.sqrt(1 - alpha_t_prev) * coord_pred
            atom_types = torch.sqrt(alpha_t_prev) * x0_atoms + torch.sqrt(1 - alpha_t_prev) * atom_pred
        
        # Convert to samples
        samples = []
        for i in range(coords.shape[0]):
            atom_logits = atom_types[i]
            atom_indices = atom_logits.argmax(dim=-1)
            atom_nums = self.vocab[atom_indices]
            
            mask = atom_nums != 0
            if mask.sum() > 0:
                samples.append((coords[i][mask].cpu() * 2.2, atom_nums[mask].cpu()))
            else:
                samples.append(None)
        return samples
    
    def _ddim_reconstruct(self, coords, atom_types, num_steps=25):
        """Reconstruct from slightly noised sample."""
        # Add small noise
        t = 50
        alpha_t = self.scheduler.alphas_cumprod[t]
        noise_coord = torch.randn_like(coords) * 0.1
        noise_atom = torch.randn_like(atom_types.float()) * 0.1
        
        noisy_coords = torch.sqrt(alpha_t) * coords + torch.sqrt(1 - alpha_t) * noise_coord
        noisy_atoms = torch.sqrt(alpha_t) * atom_types + torch.sqrt(1 - alpha_t) * noise_atom
        
        # Denoise
        for i in range(t, 0, -1):
            t_batch = torch.full((coords.shape[0],), i, device=self.device).long()
            
            with torch.no_grad():
                coord_pred, atom_pred = self.model(noisy_atoms, noisy_coords, t_batch)
            
            alpha_t = self.scheduler.alphas_cumprod[i]
            alpha_t_prev = self.scheduler.alphas_cumprod[i-1]
            
            x0_coords = (noisy_coords - torch.sqrt(1 - alpha_t) * coord_pred) / torch.sqrt(alpha_t)
            x0_atoms = (noisy_atoms - torch.sqrt(1 - alpha_t) * atom_pred) / torch.sqrt(alpha_t)
            
            noisy_coords = torch.sqrt(alpha_t_prev) * x0_coords + torch.sqrt(1 - alpha_t_prev) * coord_pred
            noisy_atoms = torch.sqrt(alpha_t_prev) * x0_atoms + torch.sqrt(1 - alpha_t_prev) * atom_pred
        
        return (noisy_coords, noisy_atoms)
    
    
    def _to_rdkit_mol(self, sample):
        coords, atom_nums = sample
        # coords should already be in Angstroms here
        coords_ang = coords * 2.2
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(atom_nums))
        for i, z in enumerate(atom_nums):
            mol.AddAtom(Chem.Atom(int(z)))
            conf.SetAtomPosition(i, coords[i].tolist())
        mol.AddConformer(conf)
        try:
            rdDetermineBonds.DetermineConnectivity(mol)  # ← This is the key missing call
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    
    def _analyze_samples(self, samples):
        """Analyze generated samples."""
        valid_samples = [s for s in samples if s is not None]
        
        if len(valid_samples) == 0:
            return {
                'valid_ratio': 0.0,
                'unique_ratio': 0.0,
                'avg_atoms': 0.0
            }
        
        # Convert to SMILES for uniqueness
        smiles_list = []
        for sample in valid_samples:
            mol = self._to_rdkit_mol(sample)
            if mol is not None:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi)
                except:
                    pass
        
        valid_ratio = len(smiles_list) / len(samples)
        unique_ratio = len(set(smiles_list)) / len(smiles_list) if len(smiles_list) > 0 else 0
        avg_atoms = np.mean([len(s[1]) for s in valid_samples])
        
        return {
            'valid_ratio': valid_ratio,
            'unique_ratio': unique_ratio,
            'avg_atoms': avg_atoms
        }
    
    def _compute_distribution_stats(self, val_loader, num_samples):
        """Compute statistics from validation set."""
        atom_counts = []
        bond_lengths = []
        
        count = 0
        for coords, atom_types in val_loader:
            coords = coords.to(self.device)
            c_raw = coords[0]  # First molecule, raw from dataloader
            mask = atom_types[0].sum(-1) > 0
            c = c_raw[mask]
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0))[0]
            dists = dists[dists > 0]
            
            for i in range(coords.shape[0]):
                atom_indices = atom_types[i].argmax(dim=-1)
                atom_nums = self.vocab[atom_indices]
                mask = atom_nums != 0  # 0 is your padding atomic number
                c = coords[i][mask] * 2.2  # rescale to Angstroms
                num_atoms = mask.sum().item()
                atom_counts.append(num_atoms)
                
                # Compute average bond length (pairwise distances)
                
                if len(c) > 1:
                    dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0))[0]
                    dists = dists[dists > 0]
                    if len(dists) > 0:
                        bond_lengths.append(dists.min().item())
                
                count += 1
                if count >= num_samples:
                    break
            
            if count >= num_samples:
                break
        
        return {
            'avg_atoms': np.mean(atom_counts),
            'avg_bond_length': np.mean(bond_lengths) if bond_lengths else 0.0
        }
    
    def _compute_sample_stats(self, samples):
        """Compute statistics from generated samples."""
        valid_samples = [s for s in samples if s is not None]
        
        if len(valid_samples) == 0:
            return {'avg_atoms': 0.0, 'avg_bond_length': 0.0}
        
        atom_counts = [len(s[1]) for s in valid_samples]
        
        bond_lengths = []
        for coords, _ in valid_samples:
            if len(coords) > 1:
                coords_tensor = torch.tensor(coords) if not torch.is_tensor(coords) else coords
                dists = torch.cdist(coords_tensor.unsqueeze(0), coords_tensor.unsqueeze(0))[0]
                dists = dists[dists > 0]
                if len(dists) > 0:
                    bond_lengths.append(dists.min().item())
        
        return {
            'avg_atoms': np.mean(atom_counts),
            'avg_bond_length': np.mean(bond_lengths) if bond_lengths else 0.0
        }
    
    def _print_summary(self, results):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        all_pass = all(r.get('pass', False) for r in results.values())
        
        for name, result in results.items():
            status = "✓ PASS" if result.get('pass', False) else "✗ FAIL"
            print(f"{name.upper():20s}: {status}")
        
        print("="*80)
        
        if all_pass:
            print("🎉 ALL VALIDATION CHECKS PASSED - Model is ready for RL fine-tuning!")
        else:
            print("⚠️  SOME CHECKS FAILED - Review failed tests before proceeding to RL")
        
        print("="*80)


# ==============================================================================
# Quick validation function for checkpoints
# ==============================================================================

def quick_validate(checkpoint_path, val_loader, vocab_enc2atom, device='cuda'):
    """
    Quick validation to run during training.
    
    Usage:
        quick_validate('checkpoints/epoch_10.ckpt', val_loader, vocab_enc2atom)
    """
    from model.model import TabascoV2
    from diffusers import DDIMScheduler
    from config import DDIM_config
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = TabascoV2(...)  # Initialize with your config
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create scheduler
    scheduler = DDIMScheduler.from_config(DDIM_config)
    
    # Run validation
    validator = DiffusionValidator(model, scheduler, vocab_enc2atom, device)
    results = validator.validate_all(val_loader, num_samples=50)
    
    return results


if __name__ == '__main__':
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=200_000)
    parser.add_argument('--inner_epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_scale', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--ckpt-dir', type=str, default='logs/TrainingSDPO/ckpts')
    parser.add_argument('--sample-steps', type=int, default=25)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--log-steps', type=int, default=100)
    parser.add_argument('--save_after', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    module, vocab_enc2atom, vocab_atom2enc = build_qm9_dataloader(root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    ABSORB_IDX = len(vocab_enc2atom)
    checkpoint = torch.load(r'logs\Pretrain\ckpts\epoch=79-step=18400-loss=1.28.ckpt')['state_dict']
    #checkpoint = {k[6:]: v for k,v in checkpoint.items()}
    checkpoint = {k[7 + k[6:].index('.'):]: v for k,v in checkpoint.items() if 'model' in k}
    tabasco = TabascoV2(atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, pos_coord_dim=128, pair_rbf_centers=args.d_model//2, dropout=0.1)
    tabasco.load_state_dict(checkpoint)
    tabasco = tabasco.to(args.device)
    scheduler = DDIMScheduler.from_config(DDIM_config)
    module.setup()
    val_loader = module.val_dataloader()
    vocab_enc2atom = vocab_enc2atom.to(args.device)
    validator = DiffusionValidator(tabasco, scheduler, vocab_enc2atom)
    results = validator.validate_all(val_loader, num_samples=100)
    
