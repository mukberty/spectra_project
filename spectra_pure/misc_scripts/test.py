import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/curated_subset_named.parquet")

train = np.load("checkpoints/train_indices.npy")
test = np.load("checkpoints/test_indices.npy")

print("Train first 5:")
for i in train[:5]:
    print(i, df.iloc[i]["smiles"][:60])

print("\nTest first 5:")
for i in test[:5]:
    print(i, df.iloc[i]["smiles"][:60])
