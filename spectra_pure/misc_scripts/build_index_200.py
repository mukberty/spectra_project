import os
import torch
import numpy as np
import faiss
import pandas as pd
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# ============================================================
# SETUP
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# ============================================================
# LOAD DATASET
# ============================================================

PARQUET = "data/processed/curated_subset_named.parquet"
IMAGE_DIR = "data/plots/continuous_spectra"

df = pd.read_parquet(PARQUET)

train_idx = np.load(
    "checkpoints/train_indices.npy"
)

train_df = df.iloc[train_idx].reset_index(drop=True)

print("Full dataset :", len(df))
print("Train size   :", len(train_df))


# ============================================================
# DATASET — TRAIN ONLY
# ============================================================

dataset = ImageRetrievalDataset(
    parquet_path=PARQUET,
    image_dir=IMAGE_DIR
)

dataset.df = train_df

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

ckpt = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

print("\nCheckpoint:")
print("  Epoch     :", ckpt["epoch"])
print("  Loss      :", ckpt["loss"])
print("  Vocab size:", ckpt["vocab_size"])


model = MolCLIP(
    vocab_size=ckpt["vocab_size"]
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True
)

model.eval()


# ============================================================
# BUILD IMAGE EMBEDDINGS
# ============================================================

all_embeddings = []
all_ids = []

print("\nBuilding embeddings...")

with torch.no_grad():

    for batch_num, (ir_img, raman_img, mol_id) in enumerate(loader):

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model.encode_image(
            ir_img,
            raman_img
        )

        emb = emb.cpu().numpy().astype("float32")

        all_embeddings.append(emb)
        all_ids.extend(list(mol_id))

        if batch_num % 50 == 0:
            print(
                f"Processed {batch_num * 32}/{len(dataset)}"
            )


all_embeddings = np.vstack(
    all_embeddings
).astype("float32")


# ============================================================
# NORMALIZE FOR COSINE SIMILARITY
# ============================================================

faiss.normalize_L2(
    all_embeddings
)

print("\nEmbedding shape:", all_embeddings.shape)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

dim = all_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(
    all_embeddings
)


# ============================================================
# SAVE
# ============================================================

faiss.write_index(
    index,
    "checkpoints/molclip.index"
)

np.save(
    "checkpoints/molclip_ids.npy",
    np.array(all_ids)
)


print("\n========================================")
print("INDEX BUILT SUCCESSFULLY")
print("========================================")
print("Index size :", index.ntotal)
print("Dimension  :", index.d)
print("IDs        :", len(all_ids))
print("Checkpoint :", ckpt["epoch"])
print("========================================")
