import os
import pickle
import numpy as np
import pandas as pd
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T

from models.image_cnn import ImageCNN
from models.smiles_encoder import SmilesEncoder


# ==========================================================
# CONFIG
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARQUET = "data/processed/curated_subset_named.parquet"
IMAGE_DIR = "data/plots/continuous_spectra"

TRAIN_INDICES = "checkpoints/train_indices.npy"

CHECKPOINT = "checkpoints/best_model.pt"
TOKENIZER = "checkpoints/tokenizer.pkl"

INDEX_PATH = "checkpoints/molclip.index"
IDS_PATH = "checkpoints/molclip_ids.npy"

BATCH_SIZE = 32


# ==========================================================
# MODEL
# ==========================================================

class MolCLIP(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.image_encoder = ImageCNN(embed_dim=256)

        self.smiles_encoder = SmilesEncoder(
            vocab_size=vocab_size,
            embed_dim=256
        )

    def encode_image(self, ir, ram):
        z = self.image_encoder(ir, ram)
        return F.normalize(z, dim=1)


# ==========================================================
# DATASET
# ==========================================================

class TrainRetrievalDataset(Dataset):

    def __init__(self):

        self.df = pd.read_parquet(PARQUET)

        self.indices = np.load(TRAIN_INDICES)

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])

        print("Training molecules:", len(self.indices))

    def load_img(self, name):

        img = Image.open(
            os.path.join(IMAGE_DIR, name)
        ).convert("RGB")

        return self.transform(img)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = int(self.indices[idx])

        row = self.df.iloc[original_idx]

        ir = self.load_img(
            f"molecule_{original_idx}_IR.png"
        )

        ram = self.load_img(
            f"molecule_{original_idx}_Raman.png"
        )

        return ir, ram, str(row["smiles"])


# ==========================================================
# TOKENIZER
# ==========================================================

with open(TOKENIZER, "rb") as f:
    tokenizer = pickle.load(f)

VOCAB = tokenizer.vocab_size

print("Vocab:", VOCAB)


# ==========================================================
# LOAD MODEL
# ==========================================================

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

model = MolCLIP(VOCAB).to(DEVICE)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True
)

model.eval()

print("Checkpoint epoch:", ckpt["epoch"])
print("Checkpoint loss :", ckpt["loss"])


# ==========================================================
# LOAD INDEX
# ==========================================================

index = faiss.read_index(INDEX_PATH)

ids = np.load(
    IDS_PATH,
    allow_pickle=True
)

ids = [str(x) for x in ids]

print("Index vectors:", index.ntotal)


# ==========================================================
# DATALOADER
# ==========================================================

loader = DataLoader(
    TrainRetrievalDataset(),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ==========================================================
# EVALUATION
# ==========================================================

top1 = 0
top5 = 0
top10 = 0
total = 0

print("\nStarting TRAIN retrieval...\n")

with torch.no_grad():

    for batch_num, (ir, ram, smiles) in enumerate(loader):

        ir = ir.to(DEVICE)
        ram = ram.to(DEVICE)

        emb = model.encode_image(ir, ram)

        emb = emb.cpu().numpy().astype("float32")

        _, retrieved = index.search(emb, 10)

        for i in range(len(smiles)):

            truth = str(smiles[i])

            preds = [
                ids[j]
                for j in retrieved[i]
            ]

            if truth == preds[0]:
                top1 += 1

            if truth in preds[:5]:
                top5 += 1

            if truth in preds[:10]:
                top10 += 1

            total += 1

        if batch_num % 50 == 0:

            print(
                f"Batch {batch_num:3d} | "
                f"{total:5d}/{len(loader.dataset)} | "
                f"T1={top1/total:.4%}"
            )


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "="*60)
print("TRAIN RETRIEVAL RESULTS")
print("="*60)

print(f"Training samples : {total}")
print(f"Top-1 accuracy   : {top1/total:.4%}")
print(f"Top-5 accuracy   : {top5/total:.4%}")
print(f"Top-10 accuracy  : {top10/total:.4%}")

print("="*60)
