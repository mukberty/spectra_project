import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)

print("Molecules:", len(df))

# =====================================================
# ADD GLOBAL ROW ID
# =====================================================

df = df.reset_index(drop=True)

df["row_id"] = np.arange(len(df))

# =====================================================
# SAVE FULL LOOKUP
# =====================================================

df.to_parquet(
    "checkpoints/mol_lookup.parquet",
    index=False
)

# =====================================================
# SAVE LIGHTWEIGHT LOOKUP
# =====================================================

df[
    [
        "row_id",
        "smiles",
        "molecule_name"
    ]
].to_csv(
    "checkpoints/mol_lookup.csv",
    index=False
)

print()
print("Saved:")
print("  checkpoints/mol_lookup.parquet")
print("  checkpoints/mol_lookup.csv")
print()

print("Columns available:")

for col in df.columns:
    print(" -", col)

print()
print("Done.")
