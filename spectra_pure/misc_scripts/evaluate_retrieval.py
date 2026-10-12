import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import faiss


# ============================================================
# CONFIG
# ============================================================

ROOT = "/mnt/hdd1/mehul/spectra_pure"

PARQUET_PATH = os.path.join(
    ROOT,
    "data/processed/curated_subset_named.parquet"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "data/plots/continuous_spectra"
)

CHECKPOINT_PATH = os.path.join(
    ROOT,
    "checkpoints/best_model.pt"
)

TRAIN_INDICES_PATH = os.path.join(
    ROOT,
    "checkpoints/train_indices.npy"
)

TOKENIZER_PATH = os.path.join(
    ROOT,
    "checkpoints/tokenizer.pkl"
)

BATCH_SIZE = 32
NUM_WORKERS = 2

NUM_SPLITS = 5

EMBED_DIM = 256
VOCAB_SIZE = 39


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MOLCLIP RETRIEVAL EVALUATION")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# IMPORTANT:
# FORCE MolCLIP TO USE image_cnn2.py
#
# Your epoch-193 checkpoint EXACTLY matches image_cnn2.py,
# not the current models/image_cnn.py.
# ============================================================

sys.path.insert(0, ROOT)

import models.image_cnn2 as image_cnn2

# Replace models.image_cnn with image_cnn2 before importing MolCLIP
sys.modules["models.image_cnn"] = image_cnn2

from models.molclip_model import MolCLIP


# ============================================================
# DATASET
# ============================================================

class RetrievalDataset(Dataset):

    def __init__(self, df, global_indices, image_dir):

        self.df = df
        self.global_indices = np.asarray(
            global_indices,
            dtype=np.int64
        )

        self.image_dir = image_dir

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.global_indices)

    def load_image(self, filename):

        path = os.path.join(
            self.image_dir,
            filename
        )

        img = Image.open(path).convert("RGB")

        return self.transform(img)

    def __getitem__(self, idx):

        global_idx = int(
            self.global_indices[idx]
        )

        row = self.df.iloc[global_idx]

        ir_name = (
            f"molecule_{global_idx}_IR.png"
        )

        raman_name = (
            f"molecule_{global_idx}_Raman.png"
        )

        ir_img = self.load_image(ir_name)

        raman_img = self.load_image(raman_name)

        return (
            ir_img,
            raman_img,
            global_idx
        )


# ============================================================
# FIND FAISS INDEX
# ============================================================

def find_faiss_index():

    print("\nSearching for FAISS index...")

    candidates = []

    patterns = [
        os.path.join(ROOT, "**/*.index"),
        os.path.join(ROOT, "**/*.faiss"),
        os.path.join(ROOT, "**/*index*.bin"),
    ]

    for pattern in patterns:
        candidates.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    candidates = sorted(
        set(candidates)
    )

    valid = []

    for path in candidates:

        try:

            index = faiss.read_index(path)

            print(
                f"  {path} -> "
                f"vectors={index.ntotal}, "
                f"dim={index.d}"
            )

            if (
                index.ntotal == 31644
                and index.d == EMBED_DIM
            ):
                valid.append(path)

        except Exception:
            pass

    if len(valid) == 0:

        print(
            "\nERROR: Could not automatically find "
            "the 31,644-vector FAISS index."
        )

        print("\nFAISS candidates found:")

        for path in candidates:
            print(" ", path)

        sys.exit(1)

    if len(valid) > 1:

        print(
            "\nWARNING: Multiple matching FAISS indexes found."
        )

        for path in valid:
            print(" ", path)

        print(
            "\nUsing the first matching index."
        )

    index_path = valid[0]

    print(
        "\nUsing FAISS index:",
        index_path
    )

    return index_path


# ============================================================
# FIND INDEX IDS
# ============================================================

def find_index_ids():

    print("\nSearching for index ID file...")

    candidates = []

    patterns = [
        os.path.join(ROOT, "**/*ids*.npy"),
        os.path.join(ROOT, "**/*index*.npy"),
    ]

    for pattern in patterns:
        candidates.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    candidates = sorted(
        set(candidates)
    )

    valid = []

    for path in candidates:

        try:

            arr = np.load(path)

            if len(arr) == 31644:

                print(
                    f"  {path} -> "
                    f"{len(arr)} IDs"
                )

                valid.append(path)

        except Exception:
            pass

    if len(valid) == 0:

        print(
            "\nERROR: Could not find the "
            "31,644-element index ID file."
        )

        print("\nCandidates found:")

        for path in candidates:
            print(" ", path)

        sys.exit(1)

    if len(valid) > 1:

        print(
            "\nWARNING: Multiple possible ID files found."
        )

        for path in valid:
            print(" ", path)

        print(
            "\nUsing the first matching file."
        )

    ids_path = valid[0]

    print(
        "\nUsing index IDs:",
        ids_path
    )

    return ids_path


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch", "UNKNOWN")
)

print(
    "Checkpoint loss :",
    checkpoint.get("loss", "UNKNOWN")
)

print(
    "Checkpoint vocab:",
    checkpoint.get("vocab_size", VOCAB_SIZE)
)

model = MolCLIP(
    vocab_size=VOCAB_SIZE
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("\nSTRICT MODEL LOAD SUCCESS")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_parquet(
    PARQUET_PATH
)

train_indices = np.load(
    TRAIN_INDICES_PATH
)

train_indices = np.asarray(
    train_indices,
    dtype=np.int64
)

print(
    "Full dataset:",
    len(df)
)

print(
    "Training set:",
    len(train_indices)
)

assert len(train_indices) == 31644

print(
    "Training indices loaded successfully."
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

INDEX_PATH = find_faiss_index()

INDEX_IDS_PATH = find_index_ids()

index = faiss.read_index(
    INDEX_PATH
)

index_ids = np.load(
    INDEX_IDS_PATH
).astype(np.int64)

print("\n" + "=" * 70)
print("INDEX")
print("=" * 70)

print(
    "Vectors:",
    index.ntotal
)

print(
    "Dimension:",
    index.d
)

print(
    "IDs:",
    len(index_ids)
)

assert index.ntotal == 31644
assert index.d == 256
assert len(index_ids) == 31644


# ============================================================
# VERIFY INDEX IDS MATCH TRAINING SET
# ============================================================

train_id_set = set(
    train_indices.tolist()
)

index_id_set = set(
    index_ids.tolist()
)

if train_id_set != index_id_set:

    print(
        "\nWARNING:"
        "\nFAISS index IDs do NOT exactly match "
        "train_indices.npy."
    )

    print(
        "Training IDs:",
        len(train_id_set)
    )

    print(
        "Index IDs:",
        len(index_id_set)
    )

else:

    print(
        "\nINDEX IDS MATCH TRAINING SET EXACTLY."
    )


# ============================================================
# SPLIT TRAINING SET INTO 5 SUBSETS
# ============================================================

splits = np.array_split(
    train_indices,
    NUM_SPLITS
)

print("\n" + "=" * 70)
print("5-WAY TRAINING SPLIT")
print("=" * 70)

for i, split in enumerate(splits):

    print(
        f"Set {i + 1}: "
        f"{len(split)} molecules"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_split(
    split_indices,
    split_number
):

    dataset = RetrievalDataset(
        df=df,
        global_indices=split_indices,
        image_dir=IMAGE_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    total = 0

    top1 = 0
    top5 = 0
    top10 = 0

    print("\n" + "-" * 70)
    print(
        f"SET {split_number} "
        f"({len(split_indices)} molecules)"
    )
    print("-" * 70)

    with torch.no_grad():

        for batch_num, (
            ir_img,
            raman_img,
            global_ids
        ) in enumerate(loader):

            ir_img = ir_img.to(
                DEVICE,
                non_blocking=True
            )

            raman_img = raman_img.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # IMAGE EMBEDDINGS
            # ------------------------------------------------

            image_embeddings = model.encode_image(
                ir_img,
                raman_img
            )

            image_embeddings = (
                image_embeddings
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )

            # ------------------------------------------------
            # SEARCH
            #
            # Search one extra result because the query
            # molecule itself is in the training index.
            # ------------------------------------------------

            distances, positions = index.search(
                image_embeddings,
                11
            )

            global_ids = (
                global_ids
                .numpy()
                .astype(np.int64)
            )

            # ------------------------------------------------
            # EVALUATE EACH QUERY
            # ------------------------------------------------

            for i in range(
                len(global_ids)
            ):

                query_id = int(
                    global_ids[i]
                )

                retrieved_ids = []

                for pos in positions[i]:

                    if pos < 0:
                        continue

                    retrieved_id = int(
                        index_ids[pos]
                    )

                    # Remove exact self-match
                    if retrieved_id == query_id:
                        continue

                    retrieved_ids.append(
                        retrieved_id
                    )

                    if len(retrieved_ids) >= 10:
                        break

                # --------------------------------------------
                # TOP-K
                # --------------------------------------------

                if len(retrieved_ids) >= 1:

                    if (
                        retrieved_ids[0]
                        == query_id
                    ):
                        top1 += 1

                if query_id in retrieved_ids[:5]:
                    top5 += 1

                if query_id in retrieved_ids[:10]:
                    top10 += 1

                total += 1

            if batch_num % 50 == 0:

                print(
                    f"Processed batch "
                    f"{batch_num} | "
                    f"molecules: {total}/{len(dataset)}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    acc1 = 100.0 * top1 / total
    acc5 = 100.0 * top5 / total
    acc10 = 100.0 * top10 / total

    print("\nRESULTS")
    print(
        f"Set {split_number}:"
    )

    print(
        f"  Samples : {total}"
    )

    print(
        f"  Top-1   : {acc1:.2f}%"
    )

    print(
        f"  Top-5   : {acc5:.2f}%"
    )

    print(
        f"  Top-10  : {acc10:.2f}%"
    )

    return {
        "set": split_number,
        "samples": total,
        "top1": acc1,
        "top5": acc5,
        "top10": acc10
    }


# ============================================================
# RUN ALL 5 SETS
# ============================================================

results = []

for split_number, split_indices in enumerate(
    splits,
    start=1
):

    result = evaluate_split(
        split_indices,
        split_number
    )

    results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL 5-SET RETRIEVAL RESULTS")
print("=" * 70)

print(
    f"{'Set':<8}"
    f"{'Samples':<12}"
    f"{'Top-1':<12}"
    f"{'Top-5':<12}"
    f"{'Top-10':<12}"
)

print("-" * 70)

for r in results:

    print(
        f"{r['set']:<8}"
        f"{r['samples']:<12}"
        f"{r['top1']:<12.2f}"
        f"{r['top5']:<12.2f}"
        f"{r['top10']:<12.2f}"
    )


# ============================================================
# OVERALL
# ============================================================

total_samples = sum(
    r["samples"]
    for r in results
)

overall_top1 = sum(
    r["top1"] * r["samples"]
    for r in results
) / total_samples

overall_top5 = sum(
    r["top5"] * r["samples"]
    for r in results
) / total_samples

overall_top10 = sum(
    r["top10"] * r["samples"]
    for r in results
) / total_samples

print("-" * 70)

print(
    f"{'OVERALL':<8}"
    f"{total_samples:<12}"
    f"{overall_top1:<12.2f}"
    f"{overall_top5:<12.2f}"
    f"{overall_top10:<12.2f}"
)

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
