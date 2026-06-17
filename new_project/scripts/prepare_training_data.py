import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib

# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

print(df.columns)

# =====================================================
# INPUT FEATURES
# Raman only for now
# =====================================================

X = np.stack(df["continuous_raman"].values)

print("X shape:", X.shape)

# =====================================================
# LABELS
# =====================================================

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(
    df["matched_classes"]
)

print("y shape:", y.shape)

# =====================================================
# SAVE LABEL ENCODER
# =====================================================

joblib.dump(
    mlb,
    "processed/label_binarizer.pkl"
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# SAVE
# =====================================================

np.save("processed/X_train.npy", X_train)
np.save("processed/X_test.npy", X_test)

np.save("processed/y_train.npy", y_train)
np.save("processed/y_test.npy", y_test)

print("\nSaved training tensors.")
