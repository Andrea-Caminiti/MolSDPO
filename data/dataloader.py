import torch 
import torch.nn.functional as F 
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as pl
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import os
import pickle


class Collate_qm9:
    """Optimized collation function for QM9 molecules."""
    
    def __init__(self, vocab: torch.Tensor, max_N: int):
        """
        Args:
            vocab: Mapping tensor from atomic numbers to one-hot encodings [max_z+1, vocab_size]
            max_N: Maximum number of atoms to pad to
        """
        self.max_N = max_N
        self.mapping = vocab

    def __call__(self, batch):
        """
        Collate batch of molecules with padding.
        
        Returns:
            coords: [B, max_N, 3] padded coordinates
            atom_types: [B, max_N, vocab_size] padded one-hot atom types
        """
        B = len(batch)
        max_N = self.max_N
        
        # Pre-allocate tensors for efficiency
        coords = torch.zeros(B, max_N, 3, dtype=torch.float32)
        atom_types = torch.zeros(B, max_N, dtype=torch.long)
        
        for i, d in enumerate(batch):
            pos = d.pos     # [N, 3]
            z = d.z         # [N] atomic numbers
            N = pos.size(0)
            
            # Fill in actual values (no need to pad explicitly with pre-allocated zeros)
            coords[i, :N] = pos
            atom_types[i, :N] = z
        
        # Apply vocabulary mapping
        atom_types = self.mapping[atom_types]
        
        # Normalize coordinates (as in GeoDiff)
        coords = coords / 2.2
        
        return coords, atom_types


class MoleculeDataModule(pl.LightningDataModule):
    """Lightning DataModule for QM9 with train/val split."""
    
    def __init__(
        self,
        dataset: QM9,
        vocab_atom2enc: torch.Tensor,
        class_weights: torch.Tensor,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            dataset: QM9 dataset
            vocab_atom2enc: Vocabulary mapping tensor
            class_weights: Class weights for atoms
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducible splits
        """
        super().__init__()
        self.dataset = dataset
        self.vocab_atom2enc = vocab_atom2enc
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.atom_weights = class_weights
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.max_N = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        # Compute max number of atoms
        N = [d.num_nodes for d in self.dataset]
        self.max_N = max(N)
        
        # Create train/val split
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Reproducible random split
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        
        print(f"Dataset split: {train_size} train, {val_size} val (max_N={self.max_N})")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=Collate_qm9(self.vocab_atom2enc, self.max_N),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=self.num_workers,
            collate_fn=Collate_qm9(self.vocab_atom2enc, self.max_N),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


def build_qm9_dataloader(
    root: str = 'data/QM9',
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.1,
    cache_vocab: bool = True,
    seed: int = 42
) -> Tuple[MoleculeDataModule, torch.Tensor, torch.Tensor]:
    """
    Build QM9 dataloader with train/val split.
    
    Args:
        root: Path to QM9 dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        val_split: Fraction of data for validation
        cache_vocab: Whether to cache vocabulary computation
        seed: Random seed for reproducible splits
        
    Returns:
        module: Lightning DataModule with train/val dataloaders
        enc2atom: Tensor mapping from encoding index to atomic number [vocab_size]
        atom2enc: Tensor mapping from atomic number to one-hot encoding [max_z+1, vocab_size]
    """
    # Load dataset
    dataset = QM9(root)
    
    # Get or compute vocabulary
    enc2atom, atom2enc, max_N, class_weights = get_qm9_vocabulary(
        dataset, root, cache=cache_vocab
    )
    
    # Create data module
    module = MoleculeDataModule(
        dataset=dataset,
        vocab_atom2enc=atom2enc,
        class_weights=class_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed
    )
    
    return module, enc2atom, atom2enc


def get_qm9_vocabulary(
    dataset: QM9,
    root: str,
    cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Get or compute QM9 vocabulary with caching.
    
    Args:
        dataset: QM9 dataset
        root: Root directory for caching
        cache: Whether to use cached vocabulary
        
    Returns:
        enc2atom: Mapping from encoding to atomic number
        atom2enc: Mapping from atomic number to encoding
        max_N: Maximum number of atoms
        class_weights: Class weights for balancing
    """
    cache_path = os.path.join(root, 'qm9_vocab_cache.pkl')
    
    # Try to load from cache
    if cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                print(f"Loaded vocabulary from cache: {cache_path}")
                return (
                    cached['enc2atom'],
                    cached['atom2enc'],
                    cached['max_N'],
                    cached['class_weights']
                )
        except Exception as e:
            print(f"Failed to load cache ({e}), recomputing...")
    
    # Compute vocabulary
    print("Computing QM9 vocabulary...")
    enc2atom, atom2enc, max_N, class_weights = compute_qm9_vocabulary(dataset)
    
    # Save to cache
    if cache:
        try:
            os.makedirs(root, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'enc2atom': enc2atom,
                    'atom2enc': atom2enc,
                    'max_N': max_N,
                    'class_weights': class_weights
                }, f)
            print(f"Saved vocabulary to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache ({e})")
    
    return enc2atom, atom2enc, max_N, class_weights


def compute_qm9_vocabulary(dataset: QM9) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Compute vocabulary for QM9 dataset.
    
    QM9 contains: H (1), C (6), N (7), O (8), F (9)
    We add 0 for padding.
    
    Returns:
        enc2atom: [vocab_size] tensor mapping encoding index to atomic number
        atom2enc: [max_atomic_num+1, vocab_size] tensor for one-hot encoding
        max_N: Maximum number of atoms in dataset
        class_weights: [vocab_size+1] weights for class balancing
    """
    # QM9 atomic numbers (known apriori, but we can verify)
    atomic_nums = {0, 1, 6, 7, 8, 9}  # 0 for padding
    vocab_size = len(atomic_nums)
    
    # Create one-hot encodings
    one_hot = F.one_hot(torch.arange(vocab_size))
    
    # Create mappings
    vocab_atom2enc = {atom: encoding for atom, encoding in zip(sorted(atomic_nums), one_hot)}
    
    # enc2atom: [vocab_size] - index to atomic number
    enc2atom = torch.tensor(sorted(atomic_nums), dtype=torch.long)
    
    # atom2enc: [max_atomic_num+1, vocab_size] - atomic number to one-hot
    max_atomic_num = max(atomic_nums)
    atom2enc = torch.zeros((max_atomic_num + 1, vocab_size), dtype=torch.float32)
    for atom_num, encoding in vocab_atom2enc.items():
        atom2enc[atom_num] = encoding.float()
    
    # Compute max_N
    max_N = max(d.num_nodes for d in dataset)
    
    # Pre-computed class weights (from your data)
    # These are empirically tuned for QM9
    class_weights = torch.tensor([
        0.1921,  # padding
        0.2093,  # H
        0.2523,  # C
        0.6322,  # N
        0.5376,  # O
        4.1765,  # F
        0.0000   # extra padding
    ], dtype=torch.float32)
    
    print(f"Vocabulary: {sorted(atomic_nums)}")
    print(f"Max atoms: {max_N}")
    print(f"Class weights: {class_weights}")
    
    return enc2atom, atom2enc, max_N, class_weights


def compute_class_weights_from_data(
    dataset: QM9,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    Compute class weights from dataset statistics (alternative to hardcoded).
    
    Args:
        dataset: QM9 dataset
        gamma: Exponent for inverse frequency weighting (0.3-0.7 typical)
        
    Returns:
        class_weights: [vocab_size+1] tensor of weights
    """
    print("Computing class weights from data...")
    
    all_atoms = []
    N_list = []
    
    for g in tqdm(dataset, desc='Analyzing dataset'):
        all_atoms.extend(g.z.flatten().tolist())
        N_list.append(g.num_nodes)
    
    max_N = max(N_list)
    
    # Map to vocabulary indices
    atomic_nums = sorted({0, 1, 6, 7, 8, 9})
    mapping = torch.full((max(atomic_nums) + 1,), -1, dtype=torch.long)
    for idx, atom_num in enumerate(atomic_nums):
        mapping[atom_num] = idx
    
    # Count atoms
    all_atoms_tensor = torch.tensor(all_atoms, dtype=torch.long)
    all_atoms_mapped = mapping[all_atoms_tensor]
    
    # Add padding counts
    total_atoms = len(all_atoms)
    total_slots = sum(N_list) 
    padding_count = len(dataset) * max_N - total_slots
    
    zeros = torch.zeros(padding_count, dtype=torch.long)
    counts = torch.bincount(torch.cat([zeros, all_atoms_mapped]))
    
    # Inverse frequency weighting
    alpha_c = 1.0 / (counts.float() ** gamma)
    class_weights = alpha_c / alpha_c.mean()
    
    # Add extra padding weight
    class_weights = torch.cat([class_weights, torch.tensor([0.0])])
    
    print(f"Computed class weights: {class_weights}")
    
    return class_weights
if __name__ == '__main__':

    from torchani.models import ANI2x 
    m, e2a, a2e = build_qm9_dataloader()
    dl = m.train_dataloader()
    ani = ANI2x(periodic_table_index=True)
    for coords_all, atoms_all in dl:
        for j, molecules in enumerate(zip(atoms_all, coords_all)):
            symbols, coords = molecules
            mask = (symbols[:, 0] != 1.0)
            symbols, coords = symbols[mask], coords[mask]
            #edit_mol = Chem.RWMol()
            #conf = Chem.Conformer(len(symbols))
            symb = torch.zeros(symbols.shape[:1]).long().to(symbols.device)

            atoms = torch.full(atoms_all.shape[:-1], -1, device=atoms_all.device)
            for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                s = e2a[tuple(symbol.tolist())]
                symb[i] = s
            #    atom_idx = edit_mol.AddAtom(Chem.Atom(s))
            #    # Set the coordinates for each atom
            #    conf.SetAtomPosition(atom_idx, coord.tolist())
            atoms[j, mask] = symb
        coords_all.requires_grad = True
        energies = ani((atoms, coords_all)).energies
        print(energies)
        derivative = torch.autograd.grad(energies.sum(), coords_all, create_graph=True, allow_unused=True)[0]
        print(derivative)
        forces = -derivative

    