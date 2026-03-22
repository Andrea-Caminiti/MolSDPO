import os
import pickle
import rdkit
import posebusters
import pandas as pd

if __name__ == '__main__':
    d = {}
    for a in os.listdir('../generated'):

        path = os.path.join('../generated', a)
        if a.endswith('pb'):
            with open(path, 'r') as f:
                lines = f.readlines()
            e = 0
            for line in lines:
                e += eval(line[line.index('(') + 1:-2]) == 1
            d[a[:-3]] = e/len(lines)
        #if a.endswith('pkl'):
        #    with open(path, 'rb') as f:
        #        mols = pickle.load(f)
        #    mols = [mol for mol in mols if mol]
        #    df = posebusters.PoseBusters('mol').bust(mols)
        #    df.to_csv('tabasco.csv')
            #total = len(df.columns)
            #df = df.sum(axis=0)
            #df['Passed'] = df == total
            #d[a[:-3]] = df['Passed'].sum()
    print(d)
    #path = 'tabasco.csv'
    #df = pd.read_csv(path)
    #df = df[["mol_pred_loaded","sanitization","inchi_convertible","all_atoms_connected","no_radicals","bond_lengths","bond_angles","internal_steric_clash","aromatic_ring_flatness","non-aromatic_ring_non-flatness","double_bond_flatness","internal_energy"]]
    #df = df.sum(axis=1) == 12
    #print(df.sum())

    