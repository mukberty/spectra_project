import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = "processed/curated_subset_named.parquet"

OUTPUT_PARQUET = "processed/continuous_spectra.parquet"

PLOT_DIR = "plots/continuous_spectra"

GENERATE_PLOTS = True

# Number of plots to save
MAX_PLOTS = 10000000

# Spectral range
X_MIN = 400
X_MAX = 4000
STEP = 2

# Lorentzian broadening width
GAMMA = 12

# =========================================================
# CREATE WAVENUMBER AXIS
# =========================================================

x_axis = np.arange(X_MIN, X_MAX + STEP, STEP)

# =========================================================
# LORENTZIAN FUNCTION
# =========================================================

def lorentzian(x, x0, intensity, gamma):

    return intensity * (
        (gamma**2) /
        ((x - x0)**2 + gamma**2)
    )

# =========================================================
# GENERATE CONTINUOUS SPECTRUM
# =========================================================

def generate_spectrum(freqs, intensities):

    spectrum = np.zeros_like(x_axis, dtype=np.float32)

    for f, inten in zip(freqs, intensities):

        if f < X_MIN or f > X_MAX:
            continue

        spectrum += lorentzian(
            x_axis,
            f,
            inten,
            GAMMA
        )

    return spectrum

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading filtered dataset...\n")

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df)} molecules")

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

Path("processed").mkdir(exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# =========================================================
# GENERATE SPECTRA
# =========================================================

continuous_ir = []
continuous_raman = []

print("\nGenerating continuous spectra...\n")

for idx, row in tqdm(df.iterrows(), total=len(df)):

    freqs = np.array(row["freq"], dtype=np.float32)

    ir_intensities = np.array(
        row["ir_intensity"],
        dtype=np.float32
    )

    raman_intensities = np.array(
        row["raman_activity"],
        dtype=np.float32
    )

    # =====================================================
    # GENERATE BROADENED SPECTRA
    # =====================================================

    ir_spectrum = generate_spectrum(
        freqs,
        ir_intensities
    )

    raman_spectrum = generate_spectrum(
        freqs,
        raman_intensities
    )

    continuous_ir.append(ir_spectrum)
    continuous_raman.append(raman_spectrum)

    # =====================================================
    # GET MOLECULE NAME
    # =====================================================

    molecule_name = row["molecule_name"]

    # =====================================================
    # SAVE PLOTS
    # =====================================================

    if GENERATE_PLOTS and idx < MAX_PLOTS:

        # ---------------- IR ----------------

        plt.figure(figsize=(12, 5))

        plt.plot(x_axis, ir_spectrum)

        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("IR Intensity")

        plt.title(molecule_name[:80])

        plt.gca().invert_xaxis()

        plt.tight_layout()

        plt.savefig(
            f"{PLOT_DIR}/molecule_{idx}_IR.png",
            dpi=150
        )

        plt.close()

        # ---------------- Raman ----------------

        plt.figure(figsize=(12, 5))

        plt.plot(x_axis, raman_spectrum)

        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Raman Activity")

        plt.title(molecule_name[:80])

        plt.gca().invert_xaxis()

        plt.tight_layout()

        plt.savefig(
            f"{PLOT_DIR}/molecule_{idx}_Raman.png",
            dpi=150
        )

        plt.close()

# =========================================================
# SAVE FINAL DATASET
# =========================================================

output_df = pd.DataFrame({

    "smiles": df["smiles"],

    "molecule_name": df["molecule_name"],

    "labels": df["labels"],

    "matched_classes": df["matched_classes"],

    "x_axis": [x_axis] * len(df),

    "continuous_ir": continuous_ir,

    "continuous_raman": continuous_raman
})

output_df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# =========================================================
# DONE
# =========================================================

print("\n================================================")
print("DONE")
print("================================================\n")

print(f"Saved spectra dataset to:\n{OUTPUT_PARQUET}")

print(f"\nPlots saved to:\n{PLOT_DIR}")
