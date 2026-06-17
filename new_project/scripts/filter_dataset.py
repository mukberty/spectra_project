import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_FILE = "processed/filtered_dataset.parquet"

# ---------- FUNCTIONAL GROUP SMARTS ----------

FUNCTIONAL_GROUPS = {
    "alcohol": "[OX2H]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=O)[#6]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2][#6]",
    "nitrile": "C#N",
    "aromatic": "a1aaaaa1",
}

PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUPS.items()
}

filtered_rows = []

# ---------- PROCESS EACH PKL ----------

pkl_files = sorted(DATA_DIR.glob("*.pkl"))

print(f"Found {len(pkl_files)} PKL files")

for pkl_file in pkl_files:

    print(f"\nProcessing {pkl_file.name}")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    for item in tqdm(data):

        try:
            smiles = item["SMILES"]

            mol = Chem.MolFromSmiles(smiles)

            # skip invalid molecules
            if mol is None:
                continue

            # remove salts/disconnected fragments
            if "." in smiles:
                continue

            # molecular weight filter
            mw = Descriptors.MolWt(mol)

            if mw < 50 or mw > 800:
                continue

            # atom count filter
            if mol.GetNumAtoms() > 80:
                continue

            # must contain at least one target functional group
            labels = []

            for name, pattern in PATTERNS.items():
                if mol.HasSubstructMatch(pattern):
                    labels.append(name)

            if len(labels) == 0:
                continue

            # extract spectral data
            row = {
                "smiles": smiles,
                "labels": labels,
                "freq": item["freq"],
                "ir_intensity": item["IR Inten"],
                "raman_activity": item["Raman Activ"],
            }

            filtered_rows.append(row)

        except Exception as e:
            continue

# ---------- SAVE ----------

df = pd.DataFrame(filtered_rows)

print("\nFINAL DATASET SIZE:")
print(df.shape)

df.to_parquet(OUTPUT_FILE)

print(f"\nSaved to {OUTPUT_FILE}")
