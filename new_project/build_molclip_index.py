import os
import torch
import numpy as np
import faiss
import pandas as pd
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -----------------------
# setup
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------
# load dataset
# -----------------------

df = pd.read_parquet("data/processed/curated_subset_named.parquet")

train_idx = np.load("data/splits/train_indices.npy")

train_df = df.iloc[train_idx].reset_index(drop=True)

print("Train size:", len(train_df))

# IMPORTANT: override dataset with TRAIN ONLY dataframe
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

dataset.df = train_df

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# -----------------------
# load MolCLIP model
# -----------------------

ckpt = torch.load("checkpoints/molclip.pt", map_location=device)

vocab_size = ckpt["smiles_encoder.token_emb.weight"].shape[0]
print("Detected vocab_size:", vocab_size)

model = MolCLIP(vocab_size=vocab_size).to(device)
model.load_state_dict(ckpt, strict=True)
model.eval()

# -----------------------
# build embeddings
# -----------------------

all_embeddings = []
all_ids = []

with torch.no_grad():
    for ir_img, raman_img, mol_id in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model.encode_image(ir_img, raman_img)

        all_embeddings.append(emb.cpu().numpy())
        all_ids.extend(list(mol_id))

all_embeddings = np.vstack(all_embeddings).astype("float32")

# cosine similarity normalization
faiss.normalize_L2(all_embeddings)

print("Embedding shape:", all_embeddings.shape)

# -----------------------
# build FAISS index
# -----------------------

dim = all_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(all_embeddings)

# -----------------------
# save artifacts
# -----------------------

os.makedirs("checkpoints", exist_ok=True)

faiss.write_index(index, "checkpoints/molclip.index")
np.save("checkpoints/molclip_ids.npy", np.array(all_ids))

print("\n✅ INDEX BUILT SUCCESSFULLY")
print("Index size:", len(all_ids))
print("Embedding dim:", dim)
