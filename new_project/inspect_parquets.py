import pandas as pd
from pathlib import Path

PARQUET_DIR = Path("data/parquet")

for file in sorted(PARQUET_DIR.glob("*.parquet")):

    print("\n" + "=" * 100)
    print(file.name)
    print("=" * 100)

    df = pd.read_parquet(file)

    print("\nROWS:")
    print(len(df))

    print("\nCOLUMNS:")
    print(list(df.columns))

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nFIRST ROW:")

    row = df.iloc[0]

    for col in df.columns:

        print(f"\n--- {col} ---")

        try:
            print(type(row[col]))

            if hasattr(row[col], "__len__") and not isinstance(row[col], str):
                print("length =", len(row[col]))

            print(row[col])

        except Exception as e:
            print(e)
