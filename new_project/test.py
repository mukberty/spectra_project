import pandas as pd

df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)

split = int(len(df) * 0.8)

train = set(df.iloc[:split]["smiles"])
test = set(df.iloc[split:]["smiles"])

overlap = train & test

print("Train unique:", len(train))
print("Test unique:", len(test))
print("Overlap:", len(overlap))
