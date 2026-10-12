import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("\nLoading embeddings...")

embeddings = np.load(
    "processed/spectral_embeddings.npy"
)

print("Embeddings shape:", embeddings.shape)

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading dataset...")

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

print("Dataset shape:", df.shape)

# =====================================================
# LABELS
# Use first matched class for coloring
# =====================================================

labels = []

for item in df["matched_classes"]:

    if isinstance(item, list) and len(item) > 0:
        labels.append(item[0])

    else:
        labels.append("Unknown")

print("\nNumber of labels:", len(labels))

# =====================================================
# OPTIONAL SUBSAMPLING
# t-SNE can be very slow on huge datasets
# =====================================================

MAX_POINTS = 10000

if len(embeddings) > MAX_POINTS:

    print(f"\nSubsampling to {MAX_POINTS} points for t-SNE...")

    indices = np.random.choice(
        len(embeddings),
        MAX_POINTS,
        replace=False
    )

    embeddings = embeddings[indices]

    labels = [labels[i] for i in indices]

print("Embeddings used:", embeddings.shape)

# =====================================================
# t-SNE DIMENSION REDUCTION
# =====================================================

print("\nRunning t-SNE...")

reducer = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)

embedding_2d = reducer.fit_transform(
    embeddings
)

print("t-SNE output shape:", embedding_2d.shape)

# =====================================================
# LABEL ENCODING
# =====================================================

label_ids = pd.factorize(labels)[0]

# =====================================================
# PLOT
# =====================================================

print("\nGenerating plot...")

plt.figure(figsize=(12, 10))

scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=label_ids,
    s=5,
    alpha=0.7
)

plt.title("t-SNE of Spectral Embeddings")

plt.xlabel("Component 1")
plt.ylabel("Component 2")

# =====================================================
# SAVE
# =====================================================

output_path = "plots/spectral_embedding_tsne.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved plot to: {output_path}")

print("\nDone.")
