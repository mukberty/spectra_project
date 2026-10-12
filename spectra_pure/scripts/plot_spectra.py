import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# ---------- SETTINGS ----------

DATA_FILE = "processed/filtered_dataset.parquet"
OUTPUT_DIR = Path("processed/sample_plots")

OUTPUT_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 5

# ---------- LOAD DATA ----------

print("Loading dataset...")

df = pd.read_parquet(DATA_FILE)

print(df.shape)

# ---------- RANDOM SAMPLES ----------

samples = df.sample(NUM_SAMPLES, random_state=42)

# ---------- PLOT ----------

for idx, row in samples.iterrows():

    smiles = row["smiles"]

    freq = np.array(row["freq"])
    ir = np.array(row["ir_intensity"])
    raman = np.array(row["raman_activity"])

    # ---------- IR PLOT ----------

    plt.figure(figsize=(12, 5))

    plt.vlines(freq, 0, ir)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("IR Intensity")
    plt.title(f"IR Spectrum\n{smiles}")

    plt.xlim(400, 4000)

    plt.tight_layout()

    ir_file = OUTPUT_DIR / f"sample_{idx}_IR.png"

    plt.savefig(ir_file, dpi=300)

    plt.close()

    # ---------- RAMAN PLOT ----------

    plt.figure(figsize=(12, 5))

    plt.vlines(freq, 0, raman)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Raman Activity")
    plt.title(f"Raman Spectrum\n{smiles}")

    plt.xlim(400, 4000)

    plt.tight_layout()

    raman_file = OUTPUT_DIR / f"sample_{idx}_Raman.png"

    plt.savefig(raman_file, dpi=300)

    plt.close()

print(f"\nSaved plots to: {OUTPUT_DIR}")
