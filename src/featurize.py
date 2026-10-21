import warnings
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.datasets import MoleculeNet


data = MoleculeNet(root="../data", name="BACE")

def extract_data(data):
    smiles = []
    y_list = []
    
    for i in range(len(data)):
        smiles.append(data[i].smiles)
        # Taking the value of y tensor
        y_list.append(int(data[i].y.item()))
        
    return smiles, np.array(y_list)

def compute_ecfp(smiles_list, radius=2, n_bits=2048):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(n_bits))
        else:
            # ECFP
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            arr = np.zeros((n_bits,))
            fp.ConvertToNumpyArray(arr)
            features.append(arr)
            
    return np.array(features)


smiles, y = extract_data(data)
X = compute_ecfp(smiles, radius=2, n_bits=2048)