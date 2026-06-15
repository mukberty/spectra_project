import faiss
import numpy as np
import umap
import matplotlib.pyplot as plt

print("Loading FAISS index...")

index = faiss.read_index(
    "checkpoints/molclip.index"
)

embeddings = index.reconstruct_n(
    0,
    index.ntotal
)

print("Embeddings:", embeddings.shape)

# use subset for speed
N = min(5000, len(embeddings))

embeddings = embeddings[:N]

print("Running UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

emb2d = reducer.fit_transform(
    embeddings
)

plt.figure(figsize=(10,8))

plt.scatter(
    emb2d[:,0],
    emb2d[:,1],
    s=8,
    alpha=0.7
)

plt.title(
    "MolCLIP Molecule Embedding Space (UMAP)"
)

plt.tight_layout()

plt.savefig(
    "molclip_umap.png",
    dpi=300
)

plt.show()
