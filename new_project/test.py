import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -------------------
# dataset
# -------------------

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Dataset size:", len(dataset))

# -------------------
# load checkpoint (SAFE MODE)
# -------------------

ckpt = torch.load("checkpoints/molclip.pt", map_location=device)

print("\nCheckpoint keys sample:")
print(list(ckpt.keys())[:10])

# detect vocab safely
vocab_size = None
for k in ckpt.keys():
    if "token_emb.weight" in k:
        vocab_size = ckpt[k].shape[0]
        break

print("Detected vocab_size:", vocab_size)

model = MolCLIP(vocab_size=vocab_size).to(device)

# IMPORTANT: strict=False for debugging ONLY
model.load_state_dict(ckpt, strict=False)

model.eval()

# -------------------
# embedding
# -------------------

embeddings = []
ids = []

print("\nBuilding embeddings...")

with torch.no_grad():
    for i, (ir, ram, mol_id) in enumerate(loader):

        ir = ir.to(device)
        ram = ram.to(device)

        emb = model.encode_image(ir, ram)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        ids.extend(list(mol_id))

        if i % 50 == 0:
            print(f"Processed batch {i}")

embeddings = np.vstack(embeddings).astype("float32")
ids = np.array(ids)

print("\nFinal shapes:")
print("Embeddings:", embeddings.shape)
print("IDs:", len(ids))

# -------------------
# FAISS
# -------------------

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print("\nFAISS index built:", index.ntotal)

# -------------------
# save
# -------------------

np.save("checkpoints/molclip_ids.npy", ids)
faiss.write_index(index, "checkpoints/molclip.index")

print("\nDONE")
