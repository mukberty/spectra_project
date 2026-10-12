import pandas as pd
import random
from pathlib import Path

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "processed/filtered_by_classes.parquet"

OUTPUT_FILE = "processed/curated_subset.parquet"

# Max molecules per class
MAX_PER_CLASS = 3000

# =========================================================
# IMPORTANT CLASSES
# =========================================================

IMPORTANT_CLASSES = [

    "5.2.1 Alcohols",
    "5.2.2 Ethers",
    "5.2.5 Aldehydes/Ketones",
    "5.2.6 Carboxylic acids",
    "5.2.9 Esters",

    "5.3.1 Amines",
    "5.3.2 Amides",
    "5.3.4 Nitro compounds",
    "5.3.13 Nitriles",

    "5.4.1 Thiols",
    "5.4.7 Sulfonyl compounds",

    "5.5 Benzene derivatives",
    "5.6 Heterocycles",

    "6.1 Amino acids",
    "6.2 Peptides",
    "6.9 Carbohydrates",
]

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...\n")

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df)} molecules")

# =========================================================
# BUILD SUBSET
# =========================================================

selected_indices = set()

for target_class in IMPORTANT_CLASSES:

    print(f"\nProcessing: {target_class}")

    matching = df[
        df["matched_classes"].apply(
            lambda x: target_class in x
        )
    ]

    print(f"Found {len(matching)} molecules")

    # Random sample
    if len(matching) > MAX_PER_CLASS:

        matching = matching.sample(
            MAX_PER_CLASS,
            random_state=42
        )

    selected_indices.update(matching.index.tolist())

# =========================================================
# FINAL DATAFRAME
# =========================================================

subset_df = df.loc[list(selected_indices)].reset_index(drop=True)

print("\n================================================")
print(f"Final curated subset: {len(subset_df)} molecules")
print("================================================\n")

# =========================================================
# SAVE
# =========================================================

Path("processed").mkdir(exist_ok=True)

subset_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(f"Saved to:\n{OUTPUT_FILE}")
