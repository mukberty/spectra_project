import numpy as np
import pandas as pd
import tensorflow as tf

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

X = np.stack(
    df["continuous_raman"].values
)

X = X[..., np.newaxis]

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(
    "processed/raman_cnn.keras"
)

# =====================================================
# EXTRACT EMBEDDING LAYER
# =====================================================

embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output
)

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

embeddings = embedding_model.predict(
    X,
    batch_size=64
)

print("Embedding shape:", embeddings.shape)

# =====================================================
# SAVE
# =====================================================

np.save(
    "processed/spectral_embeddings.npy",
    embeddings
)

print("\nSaved embeddings.")
