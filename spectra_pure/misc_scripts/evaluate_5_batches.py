import os
import pickle
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "checkpoints/best_model.pt"
INDEX_PATH = "checkpoints/molclip.index"
IDS_PATH = "checkpoints/molclip_ids.npy"
TEST_INDICES_PATH = "checkpoints/test_indices.npy"

PARQUET_PATH = "data/processed/curated_subset_named.parquet"
IMAGE_DIR = "data/plots/continuous_spectra"

NUM_BATCHES = 5
BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

ckpt = torch.load(
    CHECKPOINT,
    map_location=device
)

model = MolCLIP(
    vocab_size=ckpt["vocab_size"]
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print("Checkpoint epoch:", ckpt["epoch"])
print("Checkpoint loss :", ckpt["loss"])
print("Vocab size      :", ckpt["vocab_size"])


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

with open(
    "checkpoints/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

print("Tokenizer vocab size:", tokenizer.vocab_size)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

full_dataset = ImageRetrievalDataset(
    parquet_path=PARQUET_PATH,
    image_dir=IMAGE_DIR,
    mode="all"
)

test_indices = np.load(
    TEST_INDICES_PATH
)

print("Full dataset :", len(full_dataset))
print("Test samples :", len(test_indices))


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

ids = np.load(
    IDS_PATH,
    allow_pickle=True
)

print("Index vectors :", index.ntotal)
print("Index dimension:", index.d)
print("IDs shape     :", ids.shape)


# ============================================================
# SPLIT TEST SET INTO 5 BATCHES
# ============================================================

test_batches = np.array_split(
    test_indices,
    NUM_BATCHES
)

print("\nTest batches:")

for i, batch in enumerate(test_batches):
    print(
        f"Batch {i+1}: {len(batch)} samples"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_batch(batch_indices):

    dataset = Subset(
        full_dataset,
        batch_indices
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    top1 = 0
    top5 = 0
    top10 = 0
    total = 0

    with torch.no_grad():

        for ir_img, raman_img, smiles in loader:

            ir_img = ir_img.to(device)
            raman_img = raman_img.to(device)

            # ------------------------------------------------
            # Encode spectra
            # ------------------------------------------------

            z_img = model.encode_image(
                ir_img,
                raman_img
            )

            z_img = z_img.cpu().numpy().astype(
                "float32"
            )

            # ------------------------------------------------
            # Search molecule index
            # ------------------------------------------------

            _, retrieved = index.search(
                z_img,
                10
            )

            # ------------------------------------------------
            # Ground-truth SMILES
            # ------------------------------------------------

            for i in range(len(smiles)):

                true_smiles = smiles[i]

                retrieved_ids = ids[
                    retrieved[i]
                ]

                # Convert to strings to avoid
                # numpy/string type mismatches
                retrieved_ids = [
                    str(x)
                    for x in retrieved_ids
                ]

                true_smiles = str(
                    true_smiles
                )

                # -------------------------------
                # Top-1
                # -------------------------------

                if true_smiles == retrieved_ids[0]:
                    top1 += 1

                # -------------------------------
                # Top-5
                # -------------------------------

                if true_smiles in retrieved_ids[:5]:
                    top5 += 1

                # -------------------------------
                # Top-10
                # -------------------------------

                if true_smiles in retrieved_ids[:10]:
                    top10 += 1

                total += 1

    return {
        "n": total,
        "top1": top1 / total,
        "top5": top5 / total,
        "top10": top10 / total
    }


# ============================================================
# RUN ALL 5 BATCHES
# ============================================================

results = []

print("\n")
print("=" * 70)
print("5-BATCH TEST EVALUATION")
print("=" * 70)

for batch_number, batch_indices in enumerate(
    test_batches,
    start=1
):

    print(
        f"\nEvaluating Batch {batch_number}/{NUM_BATCHES}..."
    )

    result = evaluate_batch(
        batch_indices
    )

    results.append(
        result
    )

    print(
        f"Batch {batch_number}: "
        f"N={result['n']} | "
        f"Top-1={result['top1']:.4f} | "
        f"Top-5={result['top5']:.4f} | "
        f"Top-10={result['top10']:.4f}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

total_n = sum(
    r["n"]
    for r in results
)

total_top1 = sum(
    r["top1"] * r["n"]
    for r in results
) / total_n

total_top5 = sum(
    r["top5"] * r["n"]
    for r in results
) / total_n

total_top10 = sum(
    r["top10"] * r["n"]
    for r in results
) / total_n


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"{'Batch':<10}"
    f"{'N':<10}"
    f"{'Top-1':<12}"
    f"{'Top-5':<12}"
    f"{'Top-10':<12}"
)

print("-" * 56)

for i, r in enumerate(results, start=1):

    print(
        f"{i:<10}"
        f"{r['n']:<10}"
        f"{r['top1']*100:>8.2f}%   "
        f"{r['top5']*100:>8.2f}%   "
        f"{r['top10']*100:>8.2f}%"
    )

print("-" * 56)

print(
    f"{'Overall':<10}"
    f"{total_n:<10}"
    f"{total_top1*100:>8.2f}%   "
    f"{total_top5*100:>8.2f}%   "
    f"{total_top10*100:>8.2f}%"
)

print("=" * 70)