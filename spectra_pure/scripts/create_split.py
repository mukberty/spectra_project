import random
import numpy as np
import pandas as pd

random.seed(42)

df = pd.read_parquet(
    "processed/curated_subset_named.parquet"
)

indices = list(range(len(df)))

random.shuffle(indices)

split = int(0.8 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

np.save(
    "processed/train_indices.npy",
    train_idx
)

np.save(
    "processed/test_indices.npy",
    test_idx
)

print("saved split")
print("train:", len(train_idx))
print("test:", len(test_idx))
