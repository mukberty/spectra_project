import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

# =====================================================
# PICK MOLECULE INDEX
# =====================================================

IDX = 0

row = df.iloc[IDX]

# =====================================================
# EXTRACT RAMAN SPECTRUM
# =====================================================

wavenumbers = np.array(row["x_axis"])

intensity = np.array(row["continuous_raman"])

# =====================================================
# LIMIT TO RAMANBIOLIB RANGE
# 450–1800 cm^-1
# =====================================================

mask = (
    (wavenumbers >= 450)
    & (wavenumbers <= 1800)
)

wavenumbers = wavenumbers[mask]
intensity = intensity[mask]

# =====================================================
# MIN-MAX NORMALIZATION
# RamanBiolib expects this
# =====================================================

intensity = (
    intensity - intensity.min()
) / (
    intensity.max() - intensity.min()
)

# =====================================================
# SAVE CSV
# =====================================================

out = pd.DataFrame({
    "wavenumber": wavenumbers,
    "intensity": intensity
})

out.to_csv(
    "test_raman.csv",
    index=False
)

print("\nSaved: test_raman.csv\n")

print("Molecule:")
print(row["iupac_name"])
