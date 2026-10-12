import numpy as np
import pandas as pd

SEED = 42
TEST_FRACTION = 0.20

print("Loading dataset...")

df = pd.read_parquet(
    "data/processed/continuous_spectra.parquet"
)

print("Rows:", len(df))

# unique molecules
unique_smiles = df["smiles"].unique()

print("Unique smiles:", len(unique_smiles))

# shuffle molecules
rng = np.random.default_rng(SEED)

rng.shuffle(unique_smiles)

split_idx = int(
    len(unique_smiles) * (1 - TEST_FRACTION)
)

train_smiles = set(
    unique_smiles[:split_idx]
)

test_smiles = set(
    unique_smiles[split_idx:]
)

# convert molecule split -> row indices
train_idx = df.index[
    df["smiles"].isin(train_smiles)
].to_numpy()

test_idx = df.index[
    df["smiles"].isin(test_smiles)
].to_numpy()

print()
print("Train rows:", len(train_idx))
print("Test rows :", len(test_idx))

# sanity checks

overlap = len(
    set(train_idx) &
    set(test_idx)
)

print("Row overlap:", overlap)

train_unique = set(
    df.loc[train_idx, "smiles"]
)

test_unique = set(
    df.loc[test_idx, "smiles"]
)

smiles_overlap = len(
    train_unique &
    test_unique
)

print("SMILES overlap:", smiles_overlap)

assert overlap == 0
assert smiles_overlap == 0

np.save(
    "data/processed/train_indices.npy",
    train_idx
)

np.save(
    "data/processed/test_indices.npy",
    test_idx
)

print()
print("Saved:")
print("data/processed/train_indices.npy")
print("data/processed/test_indices.npy")
