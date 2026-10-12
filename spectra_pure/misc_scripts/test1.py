import os
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

PARQUET_PATH = "data/processed/curated_subset_named.parquet"

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "checkpoints/tokenizer.pkl"

INDEX_PATH = "checkpoints/molclip.index"
IDS_PATH = "checkpoints/molclip_ids.npy"

TEST_INDICES_PATH = "checkpoints/test_indices.npy"

IMAGE_DIR = "data/plots/continuous_spectra"

BATCH_SIZE = 32
NUM_WORKERS = 2

TOP_K = 10


# ============================================================
# MOLCLIP
# ============================================================

class MolCLIP(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.image_encoder = ImageCNN(
            embed_dim=256
        )

        self.smiles_encoder = SmilesEncoder(
            vocab_size=vocab_size,
            embed_dim=256
        )

    def encode_image(self, ir_img, raman_img):

        z = self.image_encoder(
            ir_img,
            raman_img
        )

        return F.normalize(
            z,
            dim=1
        )

    def encode_smiles(self, smiles):

        z = self.smiles_encoder(
            smiles
        )

        return F.normalize(
            z,
            dim=1
        )


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(
        self,
        parquet_path,
        image_dir,
        test_indices
    ):

        self.df = pd.read_parquet(
            parquet_path
        )

        self.image_dir = image_dir

        # IMPORTANT:
        # test_indices refer to rows in the ORIGINAL dataframe.
        self.test_indices = np.asarray(
            test_indices
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        print(
            f"Test dataset size: {len(self.test_indices)}"
        )

    def __len__(self):

        return len(self.test_indices)

    def load_img(self, filename):

        path = os.path.join(
            self.image_dir,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Image not found:\n{path}"
            )

        img = Image.open(
            path
        ).convert("RGB")

        return self.transform(img)

    def __getitem__(self, idx):

        # Position inside test_indices
        original_idx = int(
            self.test_indices[idx]
        )

        row = self.df.iloc[
            original_idx
        ]

        # The image filenames correspond to the
        # ORIGINAL dataframe row number.
        ir_name = (
            f"molecule_{original_idx}_IR.png"
        )

        raman_name = (
            f"molecule_{original_idx}_Raman.png"
        )

        ir_img = self.load_img(
            ir_name
        )

        raman_img = self.load_img(
            raman_name
        )

        smiles = str(
            row["smiles"]
        )

        return (
            ir_img,
            raman_img,
            smiles,
            original_idx
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MOLCLIP TEST SET EVALUATION")
print("=" * 70)

print()
print("Using device:", DEVICE)


# ============================================================
# LOAD DATAFRAME
# ============================================================

print()
print("Loading dataframe...")

df = pd.read_parquet(
    PARQUET_PATH
)

print(
    "Full dataset:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# LOAD TEST INDICES
# ============================================================

print()
print("Loading test indices...")

test_indices = np.load(
    TEST_INDICES_PATH
)

test_indices = np.asarray(
    test_indices,
    dtype=np.int64
)

print(
    "Number of test indices:",
    len(test_indices)
)

print(
    "First 10 test indices:",
    test_indices[:10]
)

print(
    "Min test index:",
    test_indices.min()
)

print(
    "Max test index:",
    test_indices.max()
)


# ============================================================
# SANITY CHECK TEST INDICES
# ============================================================

if test_indices.min() < 0:

    raise RuntimeError(
        "Test indices contain negative values."
    )

if test_indices.max() >= len(df):

    raise RuntimeError(
        "Test indices contain values outside the dataframe."
    )


# ============================================================
# CHECK TRAIN / TEST OVERLAP
# ============================================================

TRAIN_INDICES_PATH = "checkpoints/train_indices.npy"

if os.path.exists(
    TRAIN_INDICES_PATH
):

    train_indices = np.load(
        TRAIN_INDICES_PATH
    )

    train_set = set(
        map(
            int,
            train_indices
        )
    )

    test_set = set(
        map(
            int,
            test_indices
        )
    )

    overlap = train_set.intersection(
        test_set
    )

    print()
    print(
        "Train/test overlap:",
        len(overlap)
    )

    if len(overlap) > 0:

        raise RuntimeError(
            "TRAIN/TEST LEAKAGE DETECTED: "
            f"{len(overlap)} indices overlap."
        )

    print(
        "✓ No train/test index overlap."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

import pickle

with open(
    TOKENIZER_PATH,
    "rb"
) as f:

    tokenizer = pickle.load(
        f
    )


# Try to determine vocabulary size
vocab_size = None


if hasattr(
    tokenizer,
    "vocab_size"
):

    vocab_size = tokenizer.vocab_size


elif hasattr(
    tokenizer,
    "vocab"
):

    vocab_size = len(
        tokenizer.vocab
    )


elif isinstance(
    tokenizer,
    dict
):

    if "vocab" in tokenizer:

        vocab_size = len(
            tokenizer["vocab"]
        )

    elif "stoi" in tokenizer:

        vocab_size = len(
            tokenizer["stoi"]
        )

    elif "token_to_idx" in tokenizer:

        vocab_size = len(
            tokenizer["token_to_idx"]
        )


if vocab_size is None:

    raise RuntimeError(
        "Could not determine tokenizer vocabulary size."
    )


print(
    "Vocab size:",
    vocab_size
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading checkpoint...")

ckpt = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print(
    "Checkpoint epoch:",
    ckpt.get(
        "epoch",
        "unknown"
    )
)

print(
    "Checkpoint loss:",
    ckpt.get(
        "loss",
        "unknown"
    )
)


model = MolCLIP(
    vocab_size=vocab_size
)


model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True
)

print()
print("=" * 70)
print("STRICT MODEL LOAD SUCCESS")
print("=" * 70)


model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print()
print("Loading FAISS training index...")

index = faiss.read_index(
    INDEX_PATH
)

print(
    "Index vectors  :",
    index.ntotal
)

print(
    "Index dimension:",
    index.d
)


# ============================================================
# LOAD INDEX IDS
# ============================================================

ids = np.load(
    IDS_PATH,
    allow_pickle=True
)

print(
    "Number of IDs  :",
    len(ids)
)

if index.ntotal != len(ids):

    raise RuntimeError(
        "FAISS index / ID count mismatch:\n"
        f"Index vectors = {index.ntotal}\n"
        f"IDs           = {len(ids)}"
    )


# ============================================================
# IMPORTANT ID SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("INDEX ID SANITY CHECK")
print("=" * 70)

print(
    "ID type:",
    type(ids[0])
)

print()
print("First 5 FAISS IDs:")

for i in range(
    min(5, len(ids))
):

    print(
        f"FAISS position {i}: "
        f"{repr(ids[i])}"
    )


# ============================================================
# THE IDS ARE SMILES
# ============================================================
#
# IMPORTANT:
#
# FAISS search returns POSITIONS:
#
#     0, 1, 2, ...
#
# Those positions are used to access:
#
#     ids[position]
#
# The entries in ids themselves are SMILES strings.
#
# ============================================================

index_smiles = [
    str(x)
    for x in ids
]


# ============================================================
# CHECK INDEX SMILES
# ============================================================

print()
print(
    "Checking index SMILES..."
)

dataset_smiles = set(
    df["smiles"].astype(str)
)

index_smiles_set = set(
    index_smiles
)

print(
    "Unique SMILES in dataframe:",
    len(dataset_smiles)
)

print(
    "Unique SMILES in FAISS:",
    len(index_smiles_set)
)


# ============================================================
# CHECK WHETHER TEST MOLECULES ARE IN THE INDEX
# ============================================================

test_smiles = [
    str(
        df.iloc[int(idx)]["smiles"]
    )
    for idx in test_indices
]

test_smiles_set = set(
    test_smiles
)

test_in_index = (
    test_smiles_set
    .intersection(
        index_smiles_set
    )
)

print()
print("=" * 70)
print("CLOSED-SET EVALUATION CHECK")
print("=" * 70)

print(
    "Test molecules:",
    len(test_smiles_set)
)

print(
    "Test molecules present in FAISS:",
    len(test_in_index)
)

print(
    "Test molecules NOT in FAISS:",
    len(test_smiles_set - index_smiles_set)
)


if len(test_in_index) > 0:

    print()
    print(
        "NOTE:"
    )

    print(
        "Some test molecules are present in the "
        "training FAISS library."
    )

    print(
        "This is expected ONLY if your split is "
        "by spectrum rather than by molecule."
    )


# ============================================================
# DATASET / DATALOADER
# ============================================================

dataset = TestDataset(
    parquet_path=PARQUET_PATH,
    image_dir=IMAGE_DIR,
    test_indices=test_indices
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("STARTING TEST EVALUATION")
print("=" * 70)

top1_correct = 0
top5_correct = 0
top10_correct = 0

total = 0

# Diagnostic counters
queries_not_found = 0
queries_with_correct_top1 = 0


with torch.no_grad():

    for batch_num, batch in enumerate(
        loader
    ):

        (
            ir_img,
            raman_img,
            smiles,
            original_indices
        ) = batch

        ir_img = ir_img.to(
            DEVICE,
            non_blocking=True
        )

        raman_img = raman_img.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Encode spectra
        # ----------------------------------------------------

        image_embeddings = model.encode_image(
            ir_img,
            raman_img
        )

        image_embeddings = (
            image_embeddings
            .cpu()
            .numpy()
            .astype(
                "float32"
            )
        )


        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        distances, retrieved_positions = (
            index.search(
                image_embeddings,
                TOP_K
            )
        )


        # ----------------------------------------------------
        # Evaluate each query
        # ----------------------------------------------------

        for i in range(
            len(smiles)
        ):

            true_smiles = str(
                smiles[i]
            )

            retrieved_smiles = []

            for position in retrieved_positions[i]:

                position = int(
                    position
                )

                if position < 0:

                    continue

                if position >= len(index_smiles):

                    raise RuntimeError(
                        "FAISS returned invalid position: "
                        f"{position}"
                    )

                retrieved_smiles.append(
                    index_smiles[position]
                )


            # ------------------------------------------------
            # Correctness
            # ------------------------------------------------

            if len(retrieved_smiles) == 0:

                queries_not_found += 1

                continue


            if (
                true_smiles
                == retrieved_smiles[0]
            ):

                top1_correct += 1

                queries_with_correct_top1 += 1


            if (
                true_smiles
                in retrieved_smiles[:5]
            ):

                top5_correct += 1


            if (
                true_smiles
                in retrieved_smiles[:10]
            ):

                top10_correct += 1


            total += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_num % 50 == 0:

            current_top1 = (
                top1_correct / total
                if total > 0
                else 0
            )

            current_top5 = (
                top5_correct / total
                if total > 0
                else 0
            )

            current_top10 = (
                top10_correct / total
                if total > 0
                else 0
            )

            print(
                f"Batch {batch_num:4d} | "
                f"Queries: {total:5d} | "
                f"Top-1: {current_top1:.4%} | "
                f"Top-5: {current_top5:.4%} | "
                f"Top-10: {current_top10:.4%}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

if total == 0:

    raise RuntimeError(
        "No valid queries were evaluated."
    )


top1_accuracy = (
    top1_correct / total
)

top5_accuracy = (
    top5_correct / total
)

top10_accuracy = (
    top10_correct / total
)


# ============================================================
# RESULTS
# ============================================================

print()
print()
print("=" * 70)
print("MOLCLIP TEST SET RESULTS")
print("=" * 70)

print(
    f"Test samples evaluated : {total}"
)

print(
    f"Top-1 accuracy          : {top1_accuracy:.4%}"
)

print(
    f"Top-5 accuracy          : {top5_accuracy:.4%}"
)

print(
    f"Top-10 accuracy         : {top10_accuracy:.4%}"
)

print()
print(
    f"Queries with empty result : "
    f"{queries_not_found}"
)

print("=" * 70)


# ============================================================
# EXPECTED RANDOM BASELINE
# ============================================================

print()
print("=" * 70)
print("RANDOM BASELINE")
print("=" * 70)

N = index.ntotal

print(
    f"Candidate molecules: {N}"
)

print(
    f"Random Top-1  ≈ {1/N:.6%}"
)

print(
    f"Random Top-5  ≈ {5/N:.6%}"
)

print(
    f"Random Top-10 ≈ {10/N:.6%}"
)

print("=" * 70)
