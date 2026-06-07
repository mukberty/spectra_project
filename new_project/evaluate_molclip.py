import torch
import numpy as np
import faiss
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from models.spectra_cnn import SpectraCNN


# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_parquet("data/processed/curated_subset_named.parquet")

train_idx = np.load("data/splits/train_indices.npy")
test_idx = np.load("data/splits/test_indices.npy")

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


# ----------------------------
# DATASET (SPECTRA ONLY)
# ----------------------------
class SpectraDataset(Dataset):
    def __init__(self, df):
        self.x = np.stack(df["continuous_ir"].values).astype(np.float32)
        self.y = np.stack(df["continuous_raman"].values).astype(np.float32)
        self.smiles = df["smiles"].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ir = torch.tensor(self.x[idx])
        raman = torch.tensor(self.y[idx])
        smiles = self.smiles[idx]
        return ir, raman, smiles


train_loader = DataLoader(SpectraDataset(train_df), batch_size=64, shuffle=False)
test_loader = DataLoader(SpectraDataset(test_df), batch_size=64, shuffle=False)


# ----------------------------
# MODEL (SPECTRA CNN)
# ----------------------------
model = SpectraCNN(embed_dim=256).to(device)

ckpt = torch.load("checkpoints/molclip.pt", map_location=device)

# NOTE: if checkpoint contains full MolCLIP, extract only spectra encoder if needed
if "spectra_encoder" in ckpt:
    model.load_state_dict(ckpt["spectra_encoder"])
else:
    model.load_state_dict(ckpt)

model.eval()


# ----------------------------
# BUILD FAISS INDEX (TRAIN)
# ----------------------------
train_embeddings = []
train_ids = []

with torch.no_grad():
    for ir, raman, smiles in train_loader:
        ir = ir.to(device)
        raman = raman.to(device)

        emb = model(ir)

        train_embeddings.append(emb.cpu().numpy())
        train_ids.extend(smiles)


train_embeddings = np.vstack(train_embeddings).astype("float32")

faiss.normalize_L2(train_embeddings)

index = faiss.IndexFlatIP(train_embeddings.shape[1])
index.add(train_embeddings)


# ----------------------------
# EVALUATION
# ----------------------------
top1 = 0
top5 = 0
total = 0

with torch.no_grad():
    for ir, raman, smiles in test_loader:

        ir = ir.to(device)
        raman = raman.to(device)

        query = model(ir)
        query = query.cpu().numpy().astype("float32")

        faiss.normalize_L2(query)

        scores, idx = index.search(query, 5)

        for i in range(len(smiles)):
            true = smiles[i]
            retrieved = [train_ids[j] for j in idx[i]]

            if true == retrieved[0]:
                top1 += 1

            if true in retrieved:
                top5 += 1

            total += 1


print("\n===== FINAL RESULTS =====")
print(f"Top-1 Accuracy: {top1/total:.4f}")
print(f"Top-5 Accuracy: {top5/total:.4f}")
print(f"Total samples: {total}")
