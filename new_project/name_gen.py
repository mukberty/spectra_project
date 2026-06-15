import pandas as pd
from rdkit import Chem
from tqdm import tqdm


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)

print("Total molecules:", len(df))

# ==========================================
# CLASSIFY NAMES
# ==========================================

valid_smiles = 0
names = 0

results = []

for molecule_name in tqdm(
    df["molecule_name"].astype(str),
    total=len(df)
):

    mol = Chem.MolFromSmiles(
        molecule_name
    )

    if mol is not None:

        classification = "VALID_SMILES"
        valid_smiles += 1

    else:

        classification = "NAME"
        names += 1

    results.append(
        {
            "molecule_name": molecule_name,
            "classification": classification
        }
    )

# ==========================================
# SUMMARY
# ==========================================

print("\n========== SUMMARY ==========")
print("Total rows        :", len(df))
print("Valid SMILES      :", valid_smiles)
print("Names (non-SMILES):", names)

print(
    "\nPercent SMILES:",
    round(
        100 * valid_smiles / len(df),
        2
    ),
    "%"
)

print(
    "Percent Names:",
    round(
        100 * names / len(df),
        2
    ),
    "%"
)

# ==========================================
# SAVE FULL CLASSIFICATION
# ==========================================

out_df = pd.DataFrame(results)

out_df.to_csv(
    "molecule_name_classification.csv",
    index=False
)

print(
    "\nSaved molecule_name_classification.csv"
)

# ==========================================
# SAVE UNIQUE NAMES ONLY
# ==========================================

unique_names = pd.DataFrame(
    {
        "molecule_name":
        sorted(
            df["molecule_name"]
            .astype(str)
            .unique()
        )
    }
)

unique_names.to_csv(
    "unique_molecule_names.csv",
    index=False
)

print(
    "Saved unique_molecule_names.csv"
)

same = (
    df["molecule_name"].astype(str)
    ==
    df["smiles"].astype(str)
)

print(
    "\nRows where molecule_name == smiles:",
    same.sum()
)

print(
    "Percent:",
    round(
        100 * same.sum() / len(df),
        2
    ),
    "%"
)
