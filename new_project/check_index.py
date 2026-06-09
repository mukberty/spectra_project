import numpy as np
import pandas as pd
import faiss

# load index
index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy", allow_pickle=True)

# load dataset
df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)

print("FAISS vectors:", index.ntotal)
print("IDs:", len(ids))
print("Dataset rows:", len(df))

print("\nFirst 5 IDs in index:")
for x in ids[:5]:
    print(x)

print("\nFirst 5 molecule names in dataset:")
for x in df["molecule_name"].head():
    print(x)

print("\nUnique IDs in index:", len(set(ids)))
print("Unique molecules in dataset:", df["molecule_name"].nunique())
