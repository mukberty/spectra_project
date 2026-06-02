import pandas as pd

PARQUET_FILE = "data/parquet/filtered_dataset.parquet"

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_parquet(PARQUET_FILE)

print("\nROWS:", len(df))
print("\nCOLUMNS:")
for c in df.columns:
    print(" -", c)

print("\n")
print("=" * 80)
print("DTYPES")
print("=" * 80)

print(df.dtypes)

print("\n")
print("=" * 80)
print("FIRST ROW")
print("=" * 80)

row = df.iloc[0]

for col in df.columns:
    print(f"\n--- {col} ---")
    print("TYPE:", type(row[col]))
    print(row[col])

print("\n")
print("=" * 80)
print("HEAD")
print("=" * 80)

print(df.head())
