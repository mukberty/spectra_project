import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score
)
import joblib

# =====================================================
# LOAD
# =====================================================

X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

X_test = X_test[..., np.newaxis]

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(
    "processed/raman_cnn.keras"
)

# =====================================================
# LOAD LABEL BINARIZER
# =====================================================

mlb = joblib.load(
    "processed/label_binarizer.pkl"
)

# =====================================================
# PREDICT
# =====================================================

y_pred_probs = model.predict(X_test)

# Threshold
y_pred = (y_pred_probs > 0.5).astype(int)

# =====================================================
# METRICS
# =====================================================

print("\n================================================")
print("MICRO F1")
print("================================================\n")

print(
    f1_score(
        y_test,
        y_pred,
        average="micro"
    )
)

print("\n================================================")
print("MACRO F1")
print("================================================\n")

print(
    f1_score(
        y_test,
        y_pred,
        average="macro"
    )
)

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_
    )
)
