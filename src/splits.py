import random
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def get_scaffold_splits(smiles_list, seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """
    Splits indices of smiles_list based on Bemis-Murcko scaffolds.
    Ensures molecules sharing the same core scaffold stay in the same split.
    """
    # 1. Map scaffolds to indices of molecules that share them
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        else:
            scaffold = ""
        scaffold_to_indices[scaffold].append(idx)
        
    # 2. Sort scaffold groups by size (descending)
    # Putting the biggest groups into training first ensures they fit easily
    scaffold_sets = list(scaffold_to_indices.values())
    random.seed(seed)
    random.shuffle(scaffold_sets)
    scaffold_sets.sort(key=lambda x: len(x), reverse=True)
    
    total_size = len(smiles_list)
    target_train = int(train_frac * total_size)
    target_val = int(val_frac * total_size)
    
    train_idx, val_idx, test_idx = [], [], []
    
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= target_train or len(train_idx) == 0:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= target_val or len(val_idx) == 0:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
            
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def get_random_splits(smiles_list, seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """
    Standard random split for comparison control.
    """
    indices = np.arange(len(smiles_list))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    total_size = len(smiles_list)
    target_train = int(train_frac * total_size)
    target_val = int(val_frac * total_size)
    
    train_idx = indices[:target_train]
    val_idx = indices[target_train:target_train + target_val]
    test_idx = indices[target_train + target_val:]
    
    return train_idx, val_idx, test_idx