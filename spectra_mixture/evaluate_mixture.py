import random
import pickle

import faiss
import numpy as np
import pandas as pd
import torch

#from models.molclip_model import MolCLIP
from models.molclip_model_mixture import MolCLIP
from utils.mixture_dataset import MixtureDataset


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# =====================================================
# TOKENIZER
# =====================================================

with open(
    "checkpoints/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)


# =====================================================
# MODEL
# =====================================================

ckpt = torch.load(
    "checkpoints/best_mixture_embedding_maxloss.pt",
    map_location=device
)

model = MolCLIP(
    vocab_size=ckpt["vocab_size"]
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()


# =====================================================
# FAISS
# =====================================================

index = faiss.read_index(
    "checkpoints/molclip_mixture_maxloss.index"
)

ids = np.load(
    "checkpoints/molclip_mixture_maxloss_ids.npy",
    allow_pickle=True
)


# =====================================================
# DATASET
# =====================================================

dataset = MixtureDataset(
    parquet_path="data/processed/continuous_spectra.parquet"
)


# =====================================================
# RANDOM MIXTURE
# =====================================================

sample_idx = random.randint(
    0,
    len(dataset) - 1
)

(
    ir_img,
    ram_img,
    smiles1,
    smiles2,
    w
) = dataset[sample_idx]

print("\nTRUE MOLECULES:\n")
print("A:", smiles1)
print("B:", smiles2)
print("Weight:", w)


# =====================================================
# EMBED MIXTURE
# =====================================================

ir_img = ir_img.unsqueeze(0).to(device)
ram_img = ram_img.unsqueeze(0).to(device)

with torch.no_grad():

    emb1, emb2 = model.encode_image(
        ir_img,
        ram_img
    )

emb1 = emb1.cpu().numpy().astype("float32")
emb2 = emb2.cpu().numpy().astype("float32")


# =====================================================
# SEARCH
# =====================================================

scores1, idxs1 = index.search(
    emb1,
    20
)

scores2, idxs2 = index.search(
    emb2,
    20
)

print("\n==============================")
print("HEAD 1")
print("==============================\n")

for rank, idx in enumerate(idxs1[0], start=1):

    hit = ""

    if ids[idx] == smiles1:
        hit = "<=== MOLECULE A"

    elif ids[idx] == smiles2:
        hit = "<=== MOLECULE B"

    print(f"{rank:2d} | {ids[idx]} {hit}")


print("\n==============================")
print("HEAD 2")
print("==============================\n")

for rank, idx in enumerate(idxs2[0], start=1):

    hit = ""

    if ids[idx] == smiles1:
        hit = "<=== MOLECULE A"

    elif ids[idx] == smiles2:
        hit = "<=== MOLECULE B"

    print(f"{rank:2d} | {ids[idx]} {hit}")
