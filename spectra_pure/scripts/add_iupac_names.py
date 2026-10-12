import pandas as pd
import requests
import time
from tqdm import tqdm

# =====================================================
# INPUT / OUTPUT
# =====================================================

INPUT_FILE = "processed/curated_subset.parquet"

OUTPUT_FILE = "processed/curated_subset_named.parquet"

# =====================================================
# LOAD
# =====================================================

print("\nLoading dataset...\n")

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df)} molecules")

# =====================================================
# CACHE
# =====================================================

cache = {}

# =====================================================
# PUBCHEM LOOKUP
# =====================================================

def get_name(smiles):

    if smiles in cache:
        return cache[smiles]

    try:

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/smiles/{smiles}/property/"
            "IUPACName,Title/JSON"
        )

        r = requests.get(url, timeout=10)

        if r.status_code == 200:

            data = r.json()

            props = data["PropertyTable"]["Properties"][0]

            # Prefer common title
            if "Title" in props:
                name = props["Title"]

            elif "IUPACName" in props:
                name = props["IUPACName"]

            else:
                name = smiles

            cache[smiles] = name

            time.sleep(0.1)

            return name

    except:
        pass

    cache[smiles] = smiles

    return smiles

# =====================================================
# RESOLVE NAMES
# =====================================================

names = []

print("\nResolving molecule names...\n")

for smiles in tqdm(df["smiles"]):

    name = get_name(smiles)

    names.append(name)

# =====================================================
# SAVE
# =====================================================

df["molecule_name"] = names

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")

print(f"\nSaved:\n{OUTPUT_FILE}")
