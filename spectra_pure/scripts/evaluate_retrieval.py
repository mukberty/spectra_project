import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

spectra = np.load(
    "processed/spectral_embeddings.npy"
).astype("float32")

fingerprints = np.load(
    "processed/morgan_fingerprints.npy"
).astype("float32")

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

# =====================================================
# LOAD MODELS
# =====================================================

print("\nLoading encoders...")

spectrum_encoder = tf.keras.models.load_model(
    "processed/spectrum_encoder.keras"
)

molecule_encoder = tf.keras.models.load_model(
    "processed/molecule_encoder.keras"
)

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

print("\nEncoding spectra...")

spec_embeddings = spectrum_encoder.predict(
    spectra,
    batch_size=256
)

print("\nEncoding molecules...")

mol_embeddings = molecule_encoder.predict(
    fingerprints,
    batch_size=256
)

# =====================================================
# NORMALIZE
# =====================================================

spec_embeddings = (
    spec_embeddings /
    np.linalg.norm(spec_embeddings, axis=1, keepdims=True)
)

mol_embeddings = (
    mol_embeddings /
    np.linalg.norm(mol_embeddings, axis=1, keepdims=True)
)

# =====================================================
# SIMILARITY MATRIX
# =====================================================

print("\nComputing similarities...")

similarity = cosine_similarity(
    spec_embeddings,
    mol_embeddings
)

# =====================================================
# RETRIEVAL METRICS
# =====================================================

top1 = 0
top5 = 0
top10 = 0

N = similarity.shape[0]

print("\nEvaluating retrieval...")

for i in range(N):

    ranking = np.argsort(
        similarity[i]
    )[::-1]

    if i == ranking[0]:
        top1 += 1

    if i in ranking[:5]:
        top5 += 1

    if i in ranking[:10]:
        top10 += 1

top1_acc = top1 / N
top5_acc = top5 / N
top10_acc = top10 / N

print("\n====================================")
print("RETRIEVAL RESULTS")
print("====================================\n")

print(f"Top-1 Accuracy : {top1_acc:.4f}")
print(f"Top-5 Accuracy : {top5_acc:.4f}")
print(f"Top-10 Accuracy: {top10_acc:.4f}")

# =====================================================
# SHOW EXAMPLES
# =====================================================

print("\n====================================")
print("EXAMPLE RETRIEVALS")
print("====================================\n")

for idx in range(5):

    ranking = np.argsort(
        similarity[idx]
    )[::-1]

    print(f"\nQUERY MOLECULE:")
    print(df.iloc[idx]["molecule_name"])

    print("\nTop Matches:")

    for rank in range(5):

        match_idx = ranking[rank]

        print(
            f"{rank+1}. "
            f"{df.iloc[match_idx]['molecule_name']}"
        )
