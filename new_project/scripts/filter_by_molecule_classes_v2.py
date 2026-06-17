import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm
from pathlib import Path

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "processed/filtered_dataset.parquet"
OUTPUT_FILE = "processed/filtered_by_classes.parquet"

MIN_MW = 50
MAX_MW = 1000
MAX_ATOMS = 80

# =========================================================
# SMARTS PATTERNS
# =========================================================

CLASS_SMARTS = {

    # -----------------------------------------------------
    # 5.1 Aliphatic carbon compounds
    # -----------------------------------------------------

    "5.1.1 Alkanes": "[CX4]",
    "5.1.2 Haloalkanes": "[CX4][F,Cl,Br,I]",
    "5.1.3 Alkenes": "C=C",
    "5.1.4 Alkynes": "C#C",

    # -----------------------------------------------------
    # 5.2 Oxygen-containing compounds
    # -----------------------------------------------------

    "5.2.1 Alcohols": "[OX2H]",
    "5.2.2 Ethers": "[OD2]([#6])[#6]",
    "5.2.4 Peroxides": "[OX2][OX2]",
    "5.2.5 Aldehydes/Ketones": "[CX3]=O",
    "5.2.6 Carboxylic acids": "C(=O)[OX2H1]",
    "5.2.9 Esters": "C(=O)O[#6]",

    # -----------------------------------------------------
    # 5.3 Nitrogen compounds
    # -----------------------------------------------------

    "5.3.1 Amines": "[NX3;H2,H1;!$(NC=O)]",
    "5.3.2 Amides": "[NX3][CX3](=O)",
    "5.3.4 Nitro compounds": "[N+](=O)[O-]",
    "5.3.7 Imines": "C=N",
    "5.3.12 Azo compounds": "N=N",
    "5.3.13 Nitriles": "C#N",
    "5.3.16 Azides": "[N-]=[N+]=N",

    # -----------------------------------------------------
    # 5.4 Sulfur compounds
    # -----------------------------------------------------

    "5.4.1 Thiols": "[SX2H]",
    "5.4.2 Sulfides": "[#16X2]",
    "5.4.3 Disulfides": "S-S",
    "5.4.7 Sulfonyl compounds": "S(=O)(=O)",

    # -----------------------------------------------------
    # 5.5 Aromatic compounds
    # -----------------------------------------------------

    "5.5 Benzene derivatives": "c1ccccc1",
    "5.5.11 Naphthalenes": "c1ccc2ccccc2c1",
    "5.5.12 Anthracenes/Phenanthrenes": "c1ccc2cc3ccccc3cc2c1",

    # -----------------------------------------------------
    # 5.6 Heterocycles
    # -----------------------------------------------------

    "5.6 Heterocycles": "[!#6;R]",

    # -----------------------------------------------------
    # 6 Natural products
    # -----------------------------------------------------

    "6.1 Amino acids": "NCC(=O)O",
    "6.2 Peptides": "NCC(=O)N",
    "6.5 Alkaloids": "[nR]",
    "6.6 Lipids": "CCCCCCCC",
    "6.9 Carbohydrates": "[C@H](O)[C@H](O)",
}

# =========================================================
# COMPILE SMARTS
# =========================================================

compiled_patterns = {}

print("\nCompiling SMARTS patterns...\n")

for class_name, smarts in CLASS_SMARTS.items():

    try:
        patt = Chem.MolFromSmarts(smarts)

        if patt is not None:
            compiled_patterns[class_name] = patt
            print(f"[OK] {class_name}")

        else:
            print(f"[FAILED] {class_name}")

    except Exception as e:
        print(f"[ERROR] {class_name}: {e}")

print(f"\nCompiled {len(compiled_patterns)} SMARTS patterns")

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...\n")

df = pd.read_parquet(INPUT_FILE)

print(f"Total molecules in dataset: {len(df)}")

# =========================================================
# FILTERING
# =========================================================

filtered_rows = []

print("\nFiltering molecules...\n")

for idx, row in tqdm(df.iterrows(), total=len(df)):

    smiles = row.get("smiles")

    if pd.isna(smiles):
        continue

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    # Molecular weight filter
    mw = Descriptors.MolWt(mol)

    if mw < MIN_MW or mw > MAX_MW:
        continue

    # Atom count filter
    if mol.GetNumAtoms() > MAX_ATOMS:
        continue

    matched_classes = []

    for class_name, patt in compiled_patterns.items():

        try:
            if mol.HasSubstructMatch(patt):
                matched_classes.append(class_name)

        except:
            continue

    # Keep molecule if ANY class matched
    if len(matched_classes) > 0:

        new_row = row.to_dict()
        new_row["matched_classes"] = matched_classes

        filtered_rows.append(new_row)

# =========================================================
# SAVE RESULTS
# =========================================================

filtered_df = pd.DataFrame(filtered_rows)

print(f"\nFiltered molecules: {len(filtered_df)}")

Path("processed").mkdir(exist_ok=True)

filtered_df.to_parquet(OUTPUT_FILE, index=False)

print(f"\nSaved filtered dataset to:\n{OUTPUT_FILE}")

# =========================================================
# SUMMARY
# =========================================================

print("\n===================================================")
print("CLASS DISTRIBUTION")
print("===================================================\n")

if len(filtered_df) > 0:

    all_classes = []

    for classes in filtered_df["matched_classes"]:
        all_classes.extend(classes)

    summary = pd.Series(all_classes).value_counts()

    print(summary)

else:
    print("No molecules matched.")

print("\nDone.\n")
