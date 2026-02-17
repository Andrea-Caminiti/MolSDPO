import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdMolDescriptors import CalcMolFormula # Necessary for robust molecule creation
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import QED # For QED calculation
from rdkit.Chem import RDConfig
import os, sys
import torch
import argparse
import gc
from tqdm import tqdm
from data.dataloader import build_qm9_dataloader
from model.model import TabascoV2
from model.scheduler import precompute_schedule
from RL.SDPO import pipeline_with_logprob, ddim_step_with_logprob, categorical_reverse_step
# Append the path to the SAScore implementation (standard practice for RDKit SAscore)
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # Requires the SA_Score files from RDKit contrib
from diffusers import DDIMScheduler
from config import DDIM_config
# --- I. Molecule Handling Utilities ---

def coords_to_mol(atom_types, coords):
    """
    Creates an RDKit Mol object from atom types and 3D coordinates.
    This is the critical step that relies on your model's output.
    """
    mol = Chem.Mol()
    # 1. Build the atoms
    editable_mol = Chem.EditableMol(mol)
    ptable = Chem.GetPeriodicTable()
    conf = Chem.Conformer(len(atom_types))
    for element_symbol, coord in zip(atom_types, coords):
        atom = Chem.Atom(element_symbol.item())
        atom_idx = editable_mol.AddAtom(atom)
        conf.SetAtomPosition(atom_idx, coord.tolist())

    mol = editable_mol.GetMol()
    mol.AddConformer(conf)

    # 3. Infer bonds from 3D geometry (essential for 3D generation)
    # This function is crucial for converting a point cloud into a chemical graph
    try:
        Chem.rdDetermineBonds.DetermineBonds(mol) # RDKit attempts to infer bonds
        # Final sanitization attempt to fix valency issues
        Chem.SanitizeMol(mol) 
    except Exception as e:
        # print(f"Bond determination or sanitization failed: {e}")
        return None 
    
    return mol

# --- II. Core Metric Functions ---

def calculate_validity_uniqueness_novelty(generated_mols, training_smiles):
    """Calculates Validity, Uniqueness, and Novelty based on canonical SMILES."""
    valid_smiles = set()
    invalid_count = 0
    total_samples = len(generated_mols)
    
    for mol in generated_mols:
        if mol is None:
            invalid_count += 1
            continue
            
        try:
            # Get canonical SMILES (with stereochemistry)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            valid_smiles.add(smiles)
        except:
            invalid_count += 1
    
    validity = len(valid_smiles) / total_samples
    uniqueness = len(valid_smiles) / (total_samples - invalid_count) if (total_samples - invalid_count) > 0 else 0
    
    # Calculate Novelty
    training_smiles_set = set(training_smiles)
    novel_smiles = valid_smiles.difference(training_smiles_set)
    novelty = len(novel_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0
    
    return validity, uniqueness, novelty, list(valid_smiles)

def calculate_diversity(valid_smiles_list):
    """Calculates internal diversity (1 - average Tanimoto similarity)."""
    if len(valid_smiles_list) < 2:
        return 0.0

    # Convert SMILES back to Mol objects (needed for fingerprinting)
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles_list if Chem.MolFromSmiles(s) is not None]
    
    # Use Morgan Fingerprints (ECFP4)
    fingerprints = [AllChem.GetMorganFingerprint(m, 2) for m in mols]
    
    similarity_sum = 0
    count = 0
    
    # Calculate pairwise similarity
    for i in range(len(fingerprints)):
        sims = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        similarity_sum += sum(sims)
        count += len(sims)
        
    avg_similarity = similarity_sum / count if count > 0 else 0
    diversity = 1.0 - avg_similarity # Diversity is 1 - Similarity
    
    return diversity

def calculate_drug_properties(valid_smiles_list):
    """Calculates average QED and SAscore."""
    if not valid_smiles_list:
        return 0.0, 0.0
    
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles_list if Chem.MolFromSmiles(s) is not None]
    
    qed_scores = [QED.qed(m) for m in mols]
    # SAscore requires the RDKit contrib path to be set correctly
    sa_scores = [sascorer.calculateScore(m) for m in mols]
    
    avg_qed = np.mean(qed_scores)
    avg_sascore = np.mean(sa_scores)
    
    return avg_qed, avg_sascore

# --- III. Main Evaluation Function ---

def evaluate_generated_molecules(generated_data, training_smiles, N_samples=5000):
    """
    Main function to run all evaluations.
    
    Args:
        generated_data (list): List of tuples, where each tuple is 
                               (atom_types, coords). atom_types is a list of 
                               element symbols, coords is a numpy array (N, 3).
        training_smiles (list): List of SMILES strings from the training set.
        N_samples (int): Number of generated samples to evaluate.
        
    Returns:
        dict: Dictionary containing all calculated metrics.
    """
    print(f"--- Starting Evaluation on {len(generated_data)} samples ---")
    
    # A. Convert all generated 3D data into RDKit Mol objects
    generated_mols = []
    for i, (coords, atom_types) in enumerate(generated_data[:N_samples]):
        mask = (atom_types[:, :, 0] != 1.0)
        print(mask.sum(), atom_types[0], mask[0])
        symbols, coords = atom_types[mask], coords[mask]
        symb = enc2atom[symbols.argmax(dim=1)]
        mol = coords_to_mol(symb, coords)
        generated_mols.append(mol)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} molecules...")

    # B. Calculate Validity, Uniqueness, Novelty
    validity, uniqueness, novelty, valid_smiles = \
        calculate_validity_uniqueness_novelty(generated_mols, training_smiles)

    # C. Calculate Diversity
    diversity = calculate_diversity(valid_smiles)
    
    # D. Calculate Drug Properties
    avg_qed, avg_sascore = calculate_drug_properties(valid_smiles)

    results = {
        "Total Samples": N_samples,
        "Valid Molecules Count": len(valid_smiles),
        "Validity (%)": validity * 100,
        "Uniqueness (%)": uniqueness * 100,
        "Novelty (%)": novelty * 100,
        "Internal Diversity (1-Tanimoto)": diversity,
        "Avg QED": avg_qed,
        "Avg SAscore": avg_sascore
    }
    
    print("\n--- Evaluation Complete ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}{'%' if '%' in k else ''}")

    return results

# --- IV. Example Usage ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=200_000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--sample-steps', type=int, default=25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    module, enc2atom, _ = build_qm9_dataloader()
    dl = module.train_dataloader()
    mols = []
    for batch in tqdm(dl, desc='Creating dataset smiles list...'):
        for coords, atoms in zip(*batch):
            mask = (atoms[:, 0] != 1.0)
            symbols, coords = atoms[mask], coords[mask]
            symb = enc2atom[symbols.argmax(dim=1)]
            mol = coords_to_mol(symb, coords)
            if mol:
                mol = Chem.MolToSmiles(mol)
                mols.append(mol)
    device = 'cuda'
    ABSORB_IDX = len(enc2atom)
    enc2atom = enc2atom.to(device)
    tabasco = torch.load('logs/TrainingSDPO/ckpts/Finetuned-loss-0.20340534823015333', weights_only=False)
    ##checkpoint = {k[6:]: v for k,v in checkpoint.items()}
    #checkpoint = {k[6:] : v for k,v in checkpoint.items() if 'model' in k}
    #tabasco = TabascoV2(atom_vocab_size=ABSORB_IDX, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, pos_coord_dim=128, pair_rbf_centers=args.d_model//2, dropout=0.1)
    #tabasco.load_state_dict(checkpoint)
    tabasco = tabasco.to(device)
    scheduler = DDIMScheduler.from_config(DDIM_config)
    generated_mols = []
    with torch.no_grad():
        for _ in tqdm(range(1000//32), desc='Generating mols'):
            coord = torch.randn(32, 29, 3, device=device)
            types = torch.randn(32, 29, 6, device=device)
            m, *a = pipeline_with_logprob(tabasco, coord, types, scheduler, device, num_inference_steps=25, eta=0.2)
            m[-1][0] = m[-1][0].cpu()
            m[-1][1] = m[-1][1].cpu()
            generated_mols.append(m[0])
            del m, coord, types
            gc.collect()

                
    results = evaluate_generated_molecules(generated_mols, mols, N_samples=-1)
        
    # Expected results for mock data:
    # Valid = 80/100 = 80.0%
    # Unique = (70+10)/80 = 100.0% (assuming C2H6 and H2O are unique)
    # Novel = 70/80 = 87.5% (C2H6 is novel, H2O is not)