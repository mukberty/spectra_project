import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------- SETTINGS ----------

DATA_FILE = "processed/filtered_dataset.parquet"

OUTPUT_DIR = Path("processed/broadened_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 5

# frequency grid
GRID_MIN = 400
GRID_MAX = 4000
GRID_STEP = 1

# Lorentzian width parameter
GAMMA = 20

# ---------- LOAD DATA ----------

print("Loading dataset...")

df = pd.read_parquet(DATA_FILE)

print(df.shape)

samples = df.sample(NUM_SAMPLES, random_state=42)

# ---------- LORENTZIAN BROADENING ----------

def lorentzian_broadening(freqs, intensities, grid, gamma):

    spectrum = np.zeros_like(grid)

    for f, inten in zip(freqs, intensities):

        peak = inten * (
            gamma**2 /
            ((grid - f)**2 + gamma**2)
        )

        spectrum += peak

    return spectrum

# ---------- COMMON GRID ----------

grid = np.arange(
    GRID_MIN,
    GRID_MAX + GRID_STEP,
    GRID_STEP
)

# ---------- PLOTTING ----------

for idx, row in samples.iterrows():

    smiles = row["smiles"]

    mol = Chem.MolFromSmiles(smiles)

    # molecular formula
    formula = rdMolDescriptors.CalcMolFormula(mol)

    freq = np.array(row["freq"])

    ir = np.array(row["ir_intensity"])

    raman = np.array(row["raman_activity"])

    # ---------- BROADEN SPECTRA ----------

    ir_spectrum = lorentzian_broadening(
        freq,
        ir,
        grid,
        GAMMA
    )

    raman_spectrum = lorentzian_broadening(
        freq,
        raman,
        grid,
        GAMMA
    )

    # ---------- NORMALIZE ----------

    if ir_spectrum.max() > 0:
        ir_spectrum /= ir_spectrum.max()

    if raman_spectrum.max() > 0:
        raman_spectrum /= raman_spectrum.max()

    # ---------- IR PLOT ----------

    plt.figure(figsize=(12, 5))

    plt.plot(grid, ir_spectrum)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity")

    plt.title(f"IR Spectrum\n{formula}")

    plt.xlim(400, 4000)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{idx}_IR.png",
        dpi=300
    )

    plt.close()

    # ---------- RAMAN PLOT ----------

    plt.figure(figsize=(12, 5))

    plt.plot(grid, raman_spectrum)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Activity")

    plt.title(f"Raman Spectrum\n{formula}")

    plt.xlim(400, 4000)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{idx}_Raman.png",
        dpi=300
    )

    plt.close()

print(f"\nSaved broadened spectra to: {OUTPUT_DIR}")
