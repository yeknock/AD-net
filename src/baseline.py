import sys
sys.path.append('../src')

import splits
import featurize
import pandas as pd
import importlib
from torch_geometric.datasets import MoleculeNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Main Graph data
data = MoleculeNet(root="../data", name="BACE")

# Extracted smiles, y and X
smiles, y = featurize.extract_data(data)
X = featurize.compute_ecfp(smiles)


# Splitting the data for baseline model (RF)
train_idx, val_idx, test_idx = splits.get_scaffold_splits(smiles)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]


# Training and prediction
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train.ravel())

# Evaluate on Validation set
val_preds = rf_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val.ravel(), val_preds)

print(f"Scaffold Split Validation ROC-AUC: {val_auc:.4f}")


# Random Splitting, just for comparison
rand_train_idx, rand_val_idx, rand_test_idx = splits.get_random_splits(smiles)

X_train_r, y_train_r = X[rand_train_idx], y[rand_train_idx]
X_val_r, y_val_r     = X[rand_val_idx], y[rand_val_idx]

rf_rand = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_rand.fit(X_train_r, y_train_r.ravel())

val_preds_r = rf_rand.predict_proba(X_val_r)[:, 1]
val_auc_r = roc_auc_score(y_val_r.ravel(), val_preds_r)

print(f"Random Split Validation ROC-AUC:    {val_auc_r:.4f}")


# Evaluate on Training set
train_preds = rf_model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train.ravel(), train_preds)


print(f"Train ROC-AUC:      {train_auc:.4f}")
print(f"Validation ROC-AUC: {val_auc:.4f}")