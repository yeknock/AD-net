import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def extract_data(data):
    smiles = []
    y_list = []
    
    for i in range(len(data)):
        smiles.append(data[i].smiles)
        # Taking the value of y tensor
        y_list.append(int(data[i].y.item()))
        
    return smiles, np.array(y_list)



def compute_ecfp(smiles_list, radius=2, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(n_bits, dtype=int))
        else:
            arr = generator.GetFingerprintAsNumPy(mol)
            features.append(arr)
            
    return np.array(features)