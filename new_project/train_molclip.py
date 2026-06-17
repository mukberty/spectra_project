import os
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset
from utils.smiles_tokenizer import SmilesTokenizer


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# =====================================================
# DATASET
# =====================================================

full_dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="all"
)

indices = np.arange(
    len(full_dataset)
)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    shuffle=True,
    random_state=42
)

np.save(
    "checkpoints/train_indices.npy",
    train_idx
)

np.save(
    "checkpoints/test_indices.npy",
    test_idx
)

print("Train:", len(train_idx))
print("Test :", len(test_idx))

dataset = Subset(
    full_dataset,
    train_idx
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)


# =====================================================
# TOKENIZER
# =====================================================

all_smiles = (
    full_dataset.df["smiles"]
    .astype(str)
    .tolist()
)

tokenizer = SmilesTokenizer(
    all_smiles,
    max_len=128
)

with open(
    "checkpoints/tokenizer.pkl",
    "wb"
) as f:

    pickle.dump(
        tokenizer,
        f
    )

print("[INFO] tokenizer saved")

vocab_size = tokenizer.vocab_size


# =====================================================
# MODEL
# =====================================================

model = MolCLIP(
    vocab_size=vocab_size
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# =====================================================
# LOSS
# =====================================================

def contrastive_loss(
    z_img,
    z_txt,
    temperature=0.05
):

    z_img = F.normalize(
        z_img,
        dim=1
    )

    z_txt = F.normalize(
        z_txt,
        dim=1
    )

    logits = (
        z_img @ z_txt.T
    ) / temperature

    labels = torch.arange(
        z_img.size(0),
        device=z_img.device
    )

    loss_img = F.cross_entropy(
        logits,
        labels
    )

    loss_txt = F.cross_entropy(
        logits.T,
        labels
    )

    return (
        loss_img + loss_txt
    ) / 2


# =====================================================
# TRAINING
# =====================================================

epochs = 100

best_loss = float("inf")

try:

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for ir_img, raman_img, smiles in loader:

            ir_img = ir_img.to(device)
            raman_img = raman_img.to(device)

            # =====================================
            # MODALITY DROPOUT
            # =====================================

            modality_rand = torch.rand(
                1
            ).item()

            # 15% IR only
            if modality_rand < 0.15:

                raman_img = torch.zeros_like(
                    raman_img
                )

            # 15% Raman only
            elif modality_rand < 0.30:

                ir_img = torch.zeros_like(
                    ir_img
                )

            # Remaining 70%
            # use both modalities

            smiles = tokenizer.batch_encode(
                smiles
            ).to(device)

            z_img = model.encode_image(
                ir_img,
                raman_img
            )

            z_txt = model.encode_smiles(
                smiles
            )

            loss = contrastive_loss(
                z_img,
                z_txt
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(loader)
        )

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}"
        )

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": vocab_size
                },
                "checkpoints/best_model.pt"
            )

            print(
                f"[BEST] Saved epoch {epoch+1}"
            )

except KeyboardInterrupt:

    print(
        "\nTraining interrupted by user."
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size
        },
        "checkpoints/interrupted_model.pt"
    )

    print(
        "Saved interrupted_model.pt"
    )


# =====================================================
# FINAL SAVE
# =====================================================

torch.save(
    {
        "epoch": epoch + 1,
        "loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size
    },
    "checkpoints/last_model.pt"
)

print()
print("Saved:")
print("checkpoints/best_model.pt")
print("checkpoints/last_model.pt")
print("checkpoints/train_indices.npy")
print("checkpoints/test_indices.npy")
