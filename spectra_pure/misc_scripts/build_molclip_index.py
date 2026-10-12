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
# PATHS
# ============================================================

PARQUET_PATH = (
    "data/processed/curated_subset_named.parquet"
)

IMAGE_DIR = (
    "data/plots/continuous_spectra"
)

CHECKPOINT_PATH = (
    "checkpoints/best_model.pt"
)

TRAIN_INDICES_PATH = (
    "checkpoints/train_indices.npy"
)

INDEX_PATH = (
    "checkpoints/molclip.index"
)

IDS_PATH = (
    "checkpoints/molclip_ids.npy"
)


# ============================================================
# LOAD FULL DATAFRAME
# ============================================================

df = pd.read_parquet(
    PARQUET_PATH
)

print("\nFull dataset:", len(df))


# ============================================================
# LOAD ORIGINAL TRAIN SPLIT
# ============================================================

train_idx = np.load(
    TRAIN_INDICES_PATH
)

print("Train indices:", len(train_idx))

# Explicitly select ONLY the original training molecules
train_df = (
    df.iloc[train_idx]
    .reset_index(drop=True)
)

train_df["global_idx"] = train_idx

print("Train dataframe:", len(train_df))


# ============================================================
# CREATE DATASET
# ============================================================

dataset = ImageRetrievalDataset(
    parquet_path=PARQUET_PATH,
    image_dir=IMAGE_DIR
)

# IMPORTANT:
# Override the dataframe with TRAIN ONLY.
dataset.df = train_df

print("Dataset used for index:", len(dataset))


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# LOAD EPOCH-193 CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

ckpt = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

print("Checkpoint epoch:", ckpt["epoch"])
print("Checkpoint loss :", ckpt["loss"])
print("Vocab size      :", ckpt["vocab_size"])


# ============================================================
# BUILD MODEL
# ============================================================

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

print("\nBuilding image embeddings...")

with torch.no_grad():

    for batch_num, (
        ir_img,
        raman_img,
        mol_id
    ) in enumerate(loader):

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model.encode_image(
            ir_img,
            raman_img
        )

        emb = emb.cpu().numpy().astype(
            "float32"
        )

        all_embeddings.append(
            emb
        )

        all_ids.extend(
            list(mol_id)
        )

        if batch_num % 50 == 0:
            print(
                f"Processed batch {batch_num} "
                f"| molecules: "
                f"{min((batch_num + 1) * 32, len(dataset))}"
                f"/{len(dataset)}"
            )


# ============================================================
# STACK EMBEDDINGS
# ============================================================

all_embeddings = np.vstack(
    all_embeddings
).astype("float32")

print(
    "\nEmbedding shape:",
    all_embeddings.shape
)

print(
    "Number of IDs:",
    len(all_ids)
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

# The original pipeline uses cosine similarity
# implemented through normalized vectors + inner product.

faiss.normalize_L2(
    all_embeddings
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

dim = all_embeddings.shape[1]

index = faiss.IndexFlatIP(
    dim
)

index.add(
    all_embeddings
)


# ============================================================
# SAVE INDEX
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

faiss.write_index(
    index,
    INDEX_PATH
)

np.save(
    IDS_PATH,
    np.array(all_ids)
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("MOLCLIP INDEX BUILT")
print("=" * 70)

print("Checkpoint epoch :", ckpt["epoch"])
print("Checkpoint loss  :", ckpt["loss"])

print("Full dataset     :", len(df))
print("Train samples    :", len(train_idx))
print("Index vectors    :", index.ntotal)
print("Index dimension  :", index.d)
print("IDs              :", len(all_ids))

print("=" * 70)

if index.ntotal == len(train_idx):
    print("✅ INDEX CONTAINS TRAINING SET ONLY")
else:
    print("❌ ERROR: INDEX SIZE DOES NOT MATCH TRAINING SET")
    raise RuntimeError(
        "Index contains the wrong number of molecules."
    )
