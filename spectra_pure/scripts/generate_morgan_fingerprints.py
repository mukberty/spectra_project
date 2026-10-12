import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

# =====================================================
# GENERATE FINGERPRINTS
# =====================================================

fingerprints = []

for smiles in df["smiles"]:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        fingerprints.append(
            np.zeros(2048)
        )

        continue

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    arr = np.array(fp)

    fingerprints.append(arr)

fingerprints = np.array(fingerprints)

print("Fingerprint shape:", fingerprints.shape)

# =====================================================
# SAVE
# =====================================================

np.save(
    "processed/morgan_fingerprints.npy",
    fingerprints
)

print("\nSaved fingerprints.")
