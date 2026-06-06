import numpy as np
import faiss

# load saved embeddings
embeddings = np.load("checkpoints/molecule_embeddings.npy").astype("float32")
ids = np.load("checkpoints/molecule_ids.npy")

# normalize (VERY IMPORTANT for cosine similarity)
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)  # Inner Product = cosine (after normalization)
index.add(embeddings)

faiss.write_index(index, "checkpoints/molecule.index")
np.save("checkpoints/molecule_ids.npy", ids)

print("FAISS index built:", index.ntotal)
