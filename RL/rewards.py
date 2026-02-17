import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem      
import numpy as np


class ValidityReward:
    def __init__(self, device="cuda"):
        self.device = device
        self.periodic_table = Chem.GetPeriodicTable()
        
        # Pre-cache common atomic numbers to avoid RDKit overhead in the loop
        self.atomic_nums = [1, 6, 7, 8, 9]
        self.vdw_radii = {z: self.periodic_table.GetRvdw(z) for z in self.atomic_nums}
        self.cov_radii = {z: self.periodic_table.GetRcovalent(z) for z in self.atomic_nums}
        self.max_valence = {z: self.periodic_table.GetDefaultValence(z) for z in self.atomic_nums}

    def compute_reward(self, pos, atom_types):
        """
        pos: [B, N, 3] coordinates
        atom_types: [B, N] atomic numbers
        """
        M, B, N, _ = pos.shape
        # Compute distance matrix for the whole batch: [B, N, N]
        mask = (atom_types == 0).squeeze(-1)
        dist_matrix = torch.cdist(pos, pos)
        # 1. Map RDKit Constants to Batch Tensors
        # Flatten to map, then reshape back to [B, N]
        flat_types = atom_types.flatten().cpu().tolist()
        radii_vdw = torch.tensor([self.vdw_radii.get(z, 1.5) for z in flat_types], device=self.device).view(M, B, N)
        radii_cov = torch.tensor([self.cov_radii.get(z, 0.7) for z in flat_types], device=self.device).view(M, B, N)
        valences = torch.tensor([self.max_valence.get(z, 4) for z in flat_types], device=self.device).view(M, B, N)

        radii_vdw[mask] = 0.0
        radii_cov[mask] = 0.0
        valences[mask] = 0
        
        # 2. Dynamic Bond Identification (Batch-wise)
        # radii_cov.unsqueeze(-1) + radii_cov.unsqueeze(-2) -> [B, N, N]
        bond_threshold = (radii_cov.unsqueeze(-1) + radii_cov.unsqueeze(-2)) * 1.15
        # Self-interaction mask to avoid atoms bonding with themselves
        eye = torch.eye(N, device=self.device).unsqueeze(0) # [1, N, N]
        is_bonded = (dist_matrix > 0.1) & (dist_matrix < bond_threshold) & (eye == 0)
        is_non_bonded = (dist_matrix > bond_threshold) & (eye == 0)

        # 3. Bond Length Penalty
        # We average per-molecule first to keep rewards balanced across the batch
        target_dists = radii_cov.unsqueeze(-1) + radii_cov.unsqueeze(-2)
        bond_diffs = torch.pow(dist_matrix - target_dists, 2)
        # Mask out non-bonds and sum over N, N
        bond_reward = -(bond_diffs * is_bonded.float()).sum(dim=(2, 3)) / (is_bonded.sum(dim=(2, 3)) + 1e-6)

        # 4. Steric Clash Penalty
        vdw_sum = (radii_vdw.unsqueeze(-1) + radii_vdw.unsqueeze(-2)) * 0.9
        clash_val = torch.clamp(vdw_sum - dist_matrix, min=0.0)
        clash_reward = -torch.pow(clash_val * is_non_bonded.float(), 2).sum(dim=(2, 3)) / N

        # 5. Valence Penalty
        current_valence = is_bonded.sum(dim=-1) # [B, N]
        valence_excess = torch.clamp(current_valence - valences, min=0.0)
        valence_reward = -torch.pow(valence_excess, 2).mean(dim=-1) # Mean over atoms
        # Total reward per molecule in the batch: [B]
        total_reward = bond_reward + (5.0 * clash_reward) + (2.0 * valence_reward)
        return total_reward
    
def get_reward(mols, rewarder, vocab, vdW, alpha = 10.0, energy_scale=300.0, force_scale=50.0, beta=0.01, gamma = 1.0):
    start, anchor, last = mols
    coords_all = torch.stack((start[0], anchor[0], last[0]))
    atoms_all = torch.stack((start[1], anchor[1], last[1]))
    coords_all = coords_all * 2.2
    atoms_all = vocab[atoms_all.argmax(-1, keepdim=True)]
    rewards = rewarder.compute_reward(coords_all, atoms_all)
    return rewards










#from torchani.models import ANI2x 
#import torchani
#from torchani import single_point

#HCNOFSCl
#
#def soft_steric_penalty(coords_batch, Z_batch, vdW_radii_Z, mask_batch=None):
#    """
#    Computes a differentiable, soft-overlap penalty.
#    """
#    B, N, _ = coords_batch.shape
#    vdW_radii_tensor = torch.zeros_like(Z_batch, dtype=coords_batch.dtype)
#    for Z, r in vdW_radii_Z.items():
#        vdW_radii_tensor[Z_batch == Z] = r
#
#    if mask_batch is not None:
#        vdW_radii_tensor = vdW_radii_tensor * mask_batch
#        coords_batch = coords_batch * mask_batch.unsqueeze(-1)
#
#    diff = coords_batch.unsqueeze(2) - coords_batch.unsqueeze(1)
#    dist = torch.linalg.norm(diff, dim=-1)
#    vdw_sum = vdW_radii_tensor.unsqueeze(2) + vdW_radii_tensor.unsqueeze(1)
#
#    # Use upper triangle only
#    tril_mask = torch.triu(torch.ones(N, N, device=coords_batch.device), diagonal=1).unsqueeze(0)
#
#    # REPLACEMENT LOGIC: 
#    # overlap = (sum_of_radii - distance), clamped at 0.
#    overlap = torch.clamp(vdw_sum - dist, min=0.0)
#    
#    # Square the overlap to create a smooth parabolic potential 
#    # (similar to the repulsive part of a Lennard-Jones potential)
#    clash_score = (overlap ** 2) * tril_mask
#    
#    # Average across the atoms to keep scale invariant of N
#    penalty = clash_score.sum(dim=(1, 2)) / N 
#    return penalty
#
#def get_reward(mols, ani, vocab, vdW, alpha = 10.0, energy_scale=300.0, force_scale=50.0, beta=0.01, gamma = 1.0):
#    start, anchor, last = mols
#    coords_all = torch.stack((start[0], anchor[0], last[0]))
#    atoms_all = torch.stack((start[1], anchor[1], last[1]))
#    coords_all = coords_all * 2.2
#    M, B, N, A = atoms_all.shape
#    _, _, _, C = coords_all.shape
#    atoms = torch.full(atoms_all.shape[:-1], -1, device=atoms_all.device).unsqueeze(-1)
#    coords = torch.full_like(coords_all, 0.0, device=atoms_all.device)
#    atoms_all = vocab[atoms_all.argmax(-1, keepdim=True)]
#    mask = (atoms_all[:, :, :] != 0).to(atoms_all.device)
#    num_atoms = mask.sum(2).squeeze()
#    atoms[mask] = atoms_all[mask]
#    coords[mask.repeat(1, 1, 1, 3)] = coords_all[mask.repeat(1, 1, 1, 3)]
#    atoms_all, coords_all = atoms.view(M, B, -1), coords.view(M, B, -1, C)
#    rewards = torch.full((3, atoms_all.shape[1]), -10.0).to(coords_all.device)
#    for i in range(3):
#        c = coords_all[i]
#        mask = atoms_all[i]!=-1
#        m = torch.where(mask.sum(-1, keepdim=True) > 2, torch.ones_like(mask), torch.zeros_like(mask)).bool()
#        c = c[m].reshape(-1, N, 3).clone().detach().requires_grad_()
#        a = atoms_all[i][m].reshape(-1, N)
#        mask_ = mask[m].reshape(-1, N)
#        steric = soft_steric_penalty(c, a, vdW, mask_)
#        clash_penalty = -torch.log1p(steric.float())
#        if m.sum() != 0:
#            result = single_point(ani, a, c, forces=True)
#            
#            energies = result["energies"]
#            forces = result["forces"]
#            constants = ani.energy_shifter.self_energies
#            converter = ani.species_converter
#            atoms = converter(a)
#            atom_energies = constants[atoms]
#            total_self_energy = atom_energies.view_as(a).sum(dim=-1)
#            m = m.sum(-1).bool()
#    # Interaction energy is usually a small negative or positive value
#            interaction_energy = energies - total_self_energy
#            interaction_energy = interaction_energy / num_atoms[i][m]
#            atom_force_magnitudes = torch.norm(forces, p=2, dim=-1).mean(dim=-1)
#            physics_signal = torch.exp(-(interaction_energy + alpha * atom_force_magnitudes))
#                
#            # 2. Average across all atoms: result shape (Batch,)
#                
#            gate = torch.sigmoid(10.0 * (0.1 - steric)) # High when steric is low
#            rewards[i][m] = (1 - gate) * clash_penalty + gate * (1.0 + physics_signal)
#    return rewards
#
#
#
##def get_reward(trajs, ani, vocab, vdW, alpha = 10.0, energy_scale=300.0, force_scale=50.0, beta=0.01, gamma = 1.0):
##    coords_all, atoms_all = trajs
##    flags = []
##    frags = []
##    num_atoms = []
##    atoms = torch.full(atoms_all.shape[:-1], -1, device=atoms_all.device)
##    for j, molecules in enumerate(zip(atoms_all, coords_all)):
##        symbols, coords = molecules
##        mask = (symbols[:, 0] != 1.0)
##        num_atoms.append(mask.sum())
##        symbols, coords = symbols[mask], coords[mask]
###        edit_mol = Chem.RWMol()
###        conf = Chem.Conformer(len(symbols))
##        symb = vocab[symbols.argmax(dim=1)]
###            atom_idx = edit_mol.AddAtom(Chem.Atom(s))
###            # Set the coordinates for each atom
###            conf.SetAtomPosition(atom_idx, coord.tolist())
##        atoms[j, mask] = symb
###    # Co#nvert EditableMol to Mol
###        mol = edit_mol.GetMol() 
###        mol.AddConformer(conf)
###        rdDetermineBonds.DetermineConnectivity(mol)
###        #mol = _make_mol_openbabel(coords, symb)
###        sanitization_flag = True
###        try:
###            Chem.SanitizeMol(mol)
###        except:
###            sanitization_flag = False
###        if mol:
###            frags.append(len(Chem.GetMolFrags(mol)))
###            valid_flag = (not bool(Chem.DetectChemistryProblems(mol))) and len(Chem.GetMolFrags(mol)) == 1 and sanitization_flag
###            #print(not bool(Chem.DetectChemistryProblems(mol)), len(Chem.GetMolFrags(mol)), sanitization_flag, Chem.MolToSmiles(mol))
###        else:
###            frags.append(atoms.shape[1])
###            valid_flag = False
###        
###        flags.append(valid_flag)
###    mask = torch.as_tensor(flags, device=atoms_all.device).bool()
###    #print(atoms.shape, mask.shape, atoms[mask].shape)
###    rewards = torch.zeros_like(mask, dtype=torch.float, device=mask.device)
###    if atoms[mask].shape[0] != 0:
###        atoms = atoms[mask]
##
###        coords_all = coords_all[mask]
##    coords_all.requires_grad = True
##    energies = ani((atoms, coords_all)).energies
##    derivative = torch.autograd.grad(energies.sum(), coords_all)[0]
##    f_per_atom = derivative.norm(dim=1).mean()
##    num_atoms = torch.as_tensor(num_atoms, device=energies.device)
##    e_per_atom = (energies / num_atoms)
##    e_scaled = torch.where(e_per_atom > 0, torch.log1p(e_per_atom / energy_scale), e_per_atom / energy_scale)
##    f_scaled = f_per_atom / force_scale
##    mask = atoms!=-1
##    steric = steric_penalty(coords_all, atoms, vdW, mask)
##    rewards = - (gamma * e_scaled + alpha * f_scaled - beta * steric)
##
##    return rewards, gamma * e_scaled, alpha * f_scaled, beta * steric
#
#if __name__ == '__main__':
#        a = torch.randint(6, (32, 29, 1))
#        print(a.dtype)
#        c = torch.randn((32, 29, 3))
#        total = {0, 1, 6, 7, 8, 9}
#        one_h = F.one_hot(torch.arange(len(total)))
#        vocab_atom2enc = {t:o for t,o in zip(total, one_h)}
#        atom2enc = torch.zeros((max(vocab_atom2enc.keys())+1, 6), dtype=torch.float) # invalid default
#        for k, v in vocab_atom2enc.items():
#            atom2enc[k] = v
#
#        enc2atom = torch.zeros((6, ), dtype=torch.long) # invalid default
#        for i,( k, v) in enumerate(vocab_atom2enc.items()):
#            enc2atom[i] = k
#        random2atom = torch.zeros((10, ), dtype=torch.long) # invalid default
#        for i,( k, v) in enumerate(vocab_atom2enc.items()):
#            random2atom[k] = i
#        a = random2atom[a]
#        a = atom2enc[a].squeeze()
#        mols = [(c, a), (c, a), (c, a)]
#        ani = ANI2x(periodic_table_index=True)
#        ani = ani.eval()
#        ptable = Chem.GetPeriodicTable()
#
#        # Example: list of atomic numbers you care about
#        atomic_numbers = [1, 6, 7, 8, 9,]  # H, C, N, O, F
#
#        # Extract vdW radii
#        vdW_radii = {}
#        for Z in atomic_numbers:
#            radius = ptable.GetRvdw(Z)  # Returns van der Waals radius in Å
#            vdW_radii[Z] = radius
#        print(get_reward(mols, ani, enc2atom, vdW_radii))