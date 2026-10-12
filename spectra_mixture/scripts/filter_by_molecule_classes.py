import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm
from pathlib import Path

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
# CLASS DEFINITIONS (SMARTS patterns)
# =========================================================

CLASS_SMARTS = {

    # -----------------------------------------------------
    # 5.1 Aliphatic carbon compounds
    # -----------------------------------------------------

    "5.1.1 Alkanes": "CCCC",
    "5.1.2 Haloalkanes": "[CX4][Cl,Br,I,F]",
    "5.1.3 Alkenes": "C=C",
    "5.1.4 Alkynes": "C#C",

    # -----------------------------------------------------
    # 5.2 Oxygen-containing aliphatic compounds
    # -----------------------------------------------------

    "5.2.1 Alcohols": "[OX2H]",
    "5.2.2 Ethers": "[OX2][#6]",
    "5.2.3 Acetals": "[CX4](OR)(OR)",
    "5.2.4 Peroxides": "[OX][OX]",
    "5.2.5 Aldehydes and ketones": "[CX3](=O)[#6]",
    "5.2.6 Carboxylic acids": "C(=O)[OX2H1]",
    "5.2.7 Acid halides": "C(=O)[F,Cl,Br,I]",
    "5.2.8 Cyclic anhydrides": "C1OC(=O)OC1=O",
    "5.2.9 Esters": "C(=O)O[#6]",
    "5.2.10 Lactones": "C1OC(=O)[#6]1",

    # -----------------------------------------------------
    # 5.3 Nitrogen-containing aliphatic compounds
    # -----------------------------------------------------

    "5.3.1 Amines": "[NX3;H2,H1;!$(NC=O)]",
    "5.3.2 Amides": "[NX3][CX3](=O)[#6]",
    "5.3.3 Lactams": "O=C1NCCCC1",
    "5.3.4 Nitroalkanes": "[#6][CH2][N+](=O)[O-]",
    "5.3.5 Nitrates": "[O][N](=O)=O",
    "5.3.6 Nitrites": "[O-]N=O",
    "5.3.7 Imines": "C=N",
    "5.3.8 Amidines": "C(=N)N",
    "5.3.9 Oximes": "C=NO",
    "5.3.10 Hydrazones": "C=NN",
    "5.3.11 Azines": "C=N-N=C",
    "5.3.12 Azo compounds": "N=N",
    "5.3.13 Nitriles": "C#N",
    "5.3.14 Cyanides": "[C-]#N",
    "5.3.15 Rhodanides": "SC#N",
    "5.3.16 Azides": "[N-]=[N+]=N",

    # -----------------------------------------------------
    # 5.4 Sulfur-containing aliphatic compounds
    # -----------------------------------------------------

    "5.4.1 Thiols": "[SH]",
    "5.4.2 Sulfides": "[SX2][#6]",
    "5.4.3 Disulfides": "[SX2]-[SX2]",
    "5.4.4 Polysulfides": "[SX2]([#6])[#6][SX2]",
    "5.4.5 Thiocarboxylic acids": "C(=S)[SH]",
    "5.4.6 Thionyl compounds": "S(=O)(=O)[#6]",
    "5.4.7 Sulfonyl compounds": "S(=O)(=O)[#6]",

    # -----------------------------------------------------
    # 5.5 Aromatic compounds
    # -----------------------------------------------------

    "5.5.1 Aromatic halogen compounds": "a1ccccc1X",
    "5.5.2 Monosubstituted benzenes": "a1ccccc1R",
    "5.5.3 Disubstituted benzenes": "a1cc(R)cc(R)c1",
    "5.5.4 ortho-Disubstituted benzenes": "a1c(R)c(R)cccc1",
    "5.5.5 meta-Disubstituted benzenes": "a1cc(R)cc(R)cc1",
    "5.5.6 para-Disubstituted benzenes": "a1cc(R)ccc(R)c1",
    "5.5.7 Trisubstituted benzenes": "a1cc(R)cc(R)c(R)c1",
    "5.5.8 Tetrasubstituted benzenes": "a1c(R)c(R)c(R)c(R)c1R",
    "5.5.9 Pentasubstituted benzenes": "a1c(R)c(R)c(R)c(R)c1R",
    "5.5.10 Hexasubstituted benzenes": "a1c(R)c(R)c(R)c(R)c(R)c1R",
    "5.5.11 Naphthalenes": "c1ccc2ccccc2c1",
    "5.5.12 Anthracenes and phenanthrenes": "c1ccc2c3ccccc3ccc2c1",
    "5.5.13 Polycyclic aromatic hydrocarbons": "c1ccc2ccccc2c1",

    # -----------------------------------------------------
    # 5.6 Heterocycles
    # -----------------------------------------------------

    "5.6.1 Three-membered rings": "C1CC1",
    "5.6.2 Four-membered rings": "C1CCC1",
    "5.6.3 Five-membered rings": "C1CCCC1",
    "5.6.4 Six-membered rings": "C1CCCCC1",

    # -----------------------------------------------------
    # 6 Natural products
    # -----------------------------------------------------

    "6.1 Amino acids": "NCC(=O)O",
    "6.2 Peptides": "NCC(=O)N",
    "6.3 Proteins": "NCC(=O)N",
    "6.4 Aromatic compounds": "a1ccccc1",
    "6.5 Alkaloids": "n1ccccc1",
    "6.6 Lipids": "CCCCCCCC(=O)O",
    "6.7 Steroids, terpenes, terpenoids": "C1CCC2CCCCC2C1",
    "6.8 Dyes": "N=N",
    "6.9 Carbohydrates": "C(O)C(O)C(O)C(O)C",
    "6.10 Nucleic acids": "n1cnc2n(cnc12)N",
    "6.11 Vitamins": "C1=CC2=C(C=C1)C(=O)NC(=O)C2",
    "6.12 Antibiotics": "C1CC2=C(C1)C(=O)N(C2)C",
    "6.13 Miscellaneous": "C1CCNCC1",
    "6.14 Biological samples": "C(C(=O)O)N",
}

# =========================================================
# COMPILE SMARTS
# =========================================================

compiled_patterns = {}

for name, smarts in CLASS_SMARTS.items():
    try:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled_patterns[name] = patt
    except:
        pass

print(f"Compiled {len(compiled_patterns)} SMARTS patterns")

# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

print(f"Total molecules: {len(df)}")

# =========================================================
# FILTERING
# =========================================================

filtered_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    smiles = row.get("SMILES")

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

    if len(matched_classes) > 0:

        new_row = row.to_dict()
        new_row["matched_classes"] = matched_classes

        filtered_rows.append(new_row)

# =========================================================
# SAVE
# =========================================================

filtered_df = pd.DataFrame(filtered_rows)

print(f"\nFiltered molecules: {len(filtered_df)}")

Path("processed").mkdir(exist_ok=True)

filtered_df.to_parquet(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")

# =========================================================
# SUMMARY
# =========================================================

print("\nClass distribution:\n")

all_classes = []

for classes in filtered_df["matched_classes"]:
    all_classes.extend(classes)

summary = pd.Series(all_classes).value_counts()

print(summary)
