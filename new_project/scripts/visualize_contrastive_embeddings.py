import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_parquet(
    "processed/continuous_spectra.parquet"
)

spectra = np.load(
    "processed/spectral_embeddings.npy"
).astype("float32")

# =====================================================
# LOAD ENCODER
# =====================================================

encoder = tf.keras.models.load_model(
    "processed/spectrum_encoder.keras"
)

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

embeddings = encoder.predict(
    spectra,
    batch_size=256
)

# =====================================================
# t-SNE
# =====================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

coords = tsne.fit_transform(
    embeddings[:5000]
)

# =====================================================
# COLOR BY CLASS
# =====================================================

labels = []

for x in df.iloc[:5000]["labels"]:

    if len(x) > 0:
        labels.append(x[0])
    else:
        labels.append("Unknown")

unique = list(set(labels))

mapping = {
    u:i for i,u in enumerate(unique)
}

colors = [
    mapping[x]
    for x in labels
]

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(12,10))

scatter = plt.scatter(
    coords[:,0],
    coords[:,1],
    c=colors,
    s=5
)

plt.title(
    "Contrastive Spectrum Embeddings"
)

plt.savefig(
    "processed/contrastive_tsne.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nSaved:",
    "processed/contrastive_tsne.png"
)
