import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from models.image_cnn import ImageCNN
from models.smiles_encoder import SmilesEncoder


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARQUET = "data/processed/curated_subset_named.parquet"
IMAGE_DIR = "data/plots/continuous_spectra"

TRAIN_IDX = "checkpoints/train_indices.npy"

CHECKPOINT = "checkpoints/best_model.pt"
TOKENIZER = "checkpoints/tokenizer.pkl"

INDEX = "checkpoints/molclip.index"
IDS = "checkpoints/molclip_ids.npy"

BATCH_SIZE = 32


# ============================================================
# MODEL
# ============================================================

class MolCLIP(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.image_encoder = ImageCNN(embed_dim=256)

        self.smiles_encoder = SmilesEncoder(
            vocab_size=vocab_size,
            embed_dim=256
        )

    def encode_image(self, ir, ram):
        return F.normalize(
            self.image_encoder(ir, ram),
            dim=1
        )


# ============================================================
# DATASET
# ============================================================

class TrainDataset(Dataset):

    def __init__(self):

        self.df = pd.read_parquet(PARQUET)

        self.indices = np.load(TRAIN_IDX)

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def load(self, name):
        img = Image.open(
            os.path.join(IMAGE_DIR, name)
        ).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):

        original = int(self.indices[idx])

        row = self.df.iloc[original]

        ir = self.load(f"molecule_{original}_IR.png")
        ram = self.load(f"molecule_{original}_Raman.png")

        return ir, ram, str(row["smiles"])


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(TOKENIZER, "rb") as f:
    tok = pickle.load(f)

vocab = tok.vocab_size

print("Vocab:", vocab)


# ============================================================
# LOAD MODEL
# ============================================================

ckpt = torch.load(CHECKPOINT, map_location="cpu")

model = MolCLIP(vocab).to(DEVICE)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True
)

model.eval()

print("Epoch:", ckpt["epoch"])


# ============================================================
# LOAD INDEX
# ============================================================

index = faiss.read_index(INDEX)

ids = np.load(IDS, allow_pickle=True)

ids = [str(x) for x in ids]

print("Index vectors:", index.ntotal)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    TrainDataset(),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# EVALUATION
# ============================================================

top1 = top5 = top10 = total = 0

with torch.no_grad():

    for b, (ir, ram, smiles) in enumerate(loader):

        ir = ir.to(DEVICE)
        ram = ram.to(DEVICE)

        emb = model.encode_image(ir, ram)

        emb = emb.cpu().numpy().astype("float32")

        _, retrieved = index.search(emb, 10)

        for i in range(len(smiles)):

            truth = str(smiles[i])

            preds = [ids[j] for j in retrieved[i]]

            if truth == preds[0]:
                top1 += 1

            if truth in preds[:5]:
                top5 += 1

            if truth in preds[:10]:
                top10 += 1

            total += 1

        if b % 50 == 0:
            print(
                f"Batch {b:3d} | "
                f"{total}/{len(loader.dataset)} | "
                f"T1={top1/total:.4%}"
            )

print("\n==============================")
print("TRAIN RETRIEVAL RESULTS")
print("==============================")
print(f"Samples : {total}")
print(f"Top-1   : {top1/total:.4%}")
print(f"Top-5   : {top5/total:.4%}")
print(f"Top-10  : {top10/total:.4%}")
