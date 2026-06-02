import pandas as pd

df = pd.read_parquet("continuous_spectra.parquet")

print("unique smiles:", df["smiles"].nunique())
print("rows:", len(df))
