import pickle
import numpy as np
import faiss
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# =====================================================
# DATASET
# =====================================================

full_dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="all"
)

test_idx = np.load(
    "checkpoints/test_indices.npy"
)

dataset = Subset(
    full_dataset,
    test_idx
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

print("Test molecules:", len(dataset))


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

checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

model = MolCLIP(
    vocab_size=checkpoint["vocab_size"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded model from epoch:",
    checkpoint["epoch"]
)


# =====================================================
# FAISS
# =====================================================

index = faiss.read_index(
    "checkpoints/molclip.index"
)

ids = np.load(
    "checkpoints/molclip_ids.npy",
    allow_pickle=True
)

print(
    "FAISS database size:",
    index.ntotal
)


# =====================================================
# EMBED
# =====================================================

def embed(ir, ram):

    ir = ir.to(device)
    ram = ram.to(device)

    with torch.no_grad():

        z = model.encode_image(
            ir,
            ram
        )

    return (
        z.cpu()
         .numpy()
         .astype("float32")
    )


# =====================================================
# EVALUATION
# =====================================================

top1 = 0
top5 = 0
top10 = 0
total = 0

print("\nStarting evaluation...\n")

for batch in loader:

    ir_img, raman_img, smiles = batch

    emb = embed(
        ir_img,
        raman_img
    )

    _, retrieved_idx = index.search(
        emb,
        10
    )

    for i in range(len(smiles)):

        total += 1

        true_smiles = str(
            smiles[i]
        )

        retrieved = [
            str(ids[j])
            for j in retrieved_idx[i]
        ]

        if retrieved[0] == true_smiles:
            top1 += 1

        if true_smiles in retrieved[:5]:
            top5 += 1

        if true_smiles in retrieved[:10]:
            top10 += 1


# =====================================================
# RESULTS
# =====================================================

print("\n===== TEST RESULTS =====")

print(
    f"Top-1 Accuracy : {top1/total:.4f}"
)

print(
    f"Top-5 Accuracy : {top5/total:.4f}"
)

print(
    f"Top-10 Accuracy: {top10/total:.4f}"
)

print()

print(
    "Queries:",
    total
)
