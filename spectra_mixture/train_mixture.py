import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from models.molclip_model_mixture import MolCLIP
from utils.mixture_dataset import MixtureDataset
temperature = 0.07

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

train_idx = np.load(
    "checkpoints_backup_old/train_indices.npy"
)

dataset = MixtureDataset(
    parquet_path="data/processed/continuous_spectra.parquet",
    indices=train_idx
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

print("Training molecules:", len(dataset))


# =====================================================
# TOKENIZER
# =====================================================

with open(
    "checkpoints_backup_old/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

vocab_size = tokenizer.vocab_size

print("Tokenizer loaded")
print("Vocab size:", vocab_size)


# =====================================================
# MODEL
# =====================================================

#ckpt = torch.load(
#    "checkpoints/best_mixture_cliploss.pt",
#    map_location=device
#)

#######################

model = MolCLIP(
    vocab_size=vocab_size
).to(device)

resume_file = "checkpoints/last_mixture_testing.pt"

start_epoch = 0

if os.path.exists(resume_file):

    print("======================================")
    print("Resuming previous training...")
    print("======================================")

    ckpt = torch.load(
        resume_file,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    start_epoch = ckpt["epoch"]

    print(f"Resuming from epoch {start_epoch}")

else:

    print("======================================")
    print("Loading pretrained pure-spectrum model")
    print("======================================")

    pure_ckpt = torch.load(
        "checkpoints_backup_old/best_model.pt",
        map_location=device
    )

    state = pure_ckpt["model_state_dict"]

    model.image_encoder.backbone.load_state_dict(
        {
            k.replace("image_encoder.backbone.", ""): v
            for k, v in state.items()
            if k.startswith("image_encoder.backbone.")
        }
    )

    proj_w = state["image_encoder.proj.weight"]
    proj_b = state["image_encoder.proj.bias"]

    model.image_encoder.head1.weight.data.copy_(proj_w)
    model.image_encoder.head1.bias.data.copy_(proj_b)

    model.image_encoder.head2.weight.data.copy_(proj_w)
    model.image_encoder.head2.bias.data.copy_(proj_b)

    model.smiles_encoder.load_state_dict(
        {
            k.replace("smiles_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("smiles_encoder.")
        }
    )

    print("Loaded pretrained pure-spectrum model.")
##################
#model.load_state_dict(
#    ckpt["model_state_dict"]
#)

#print(
#    f"Continuing training from epoch {ckpt['epoch']}"
#)


# =====================================================
# TRAIN BOTH ENCODERS
# =====================================================

for p in model.smiles_encoder.parameters():
    p.requires_grad = False

print("SMILES encoder frozen")


# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-6,
    weight_decay=1e-4
)

if os.path.exists(resume_file):

    optimizer.load_state_dict(
        ckpt["optimizer_state_dict"]
    )

    print("Optimizer state restored.")

# =====================================================
# LOSS
# =====================================================

def dual_head_clip_loss(
    z_img1,
    z_img2,
    z1,
    z2,
    temperature=0.07,
    diversity_weight=0.2
):

    z_img1 = F.normalize(z_img1, dim=1)
    z_img2 = F.normalize(z_img2, dim=1)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)

    labels = torch.arange(B, device=z1.device)

    # ---------------------------------
    # SAME assignment
    # ---------------------------------

    logits11 = z_img1 @ z1.T / temperature
    logits22 = z_img2 @ z2.T / temperature

    loss_same = (
        F.cross_entropy(logits11, labels)
        +
        F.cross_entropy(logits22, labels)
    )

    # ---------------------------------
    # SWAPPED assignment
    # ---------------------------------

    logits12 = z_img1 @ z2.T / temperature
    logits21 = z_img2 @ z1.T / temperature

    loss_swap = (
        F.cross_entropy(logits12, labels)
        +
        F.cross_entropy(logits21, labels)
    )

    clip_loss = torch.minimum(
        loss_same,
        loss_swap
    )

    # ---------------------------------
    # Encourage the two heads to
    # represent different molecules
    # ---------------------------------

    diversity_loss = (
        z_img1 * z_img2
    ).sum(dim=1).mean()

    loss = clip_loss + diversity_weight * diversity_loss

    return loss

# =====================================================
# TRAINING
# =====================================================

epochs = 100

if os.path.exists(resume_file):
    best_loss = ckpt["loss"]
else:
    best_loss = float("inf")

for epoch in range(start_epoch, epochs):

    model.train()

    total_loss = 0.0

    for (
        ir_img,
        ram_img,
        smiles1,
        smiles2,
        w
    ) in loader:

        ir_img = ir_img.to(device)
        ram_img = ram_img.to(device)

        w = w.float().to(device)

        z_img1, z_img2 = model.encode_image(
            ir_img,
            ram_img
        )

        z1 = model.encode_smiles(
            tokenizer.batch_encode(
                list(smiles1)
            ).to(device)
        )

        z2 = model.encode_smiles(
            tokenizer.batch_encode(
                list(smiles2)
            ).to(device)
        )

        loss = dual_head_clip_loss(
            z_img1,
            z_img2,
            z1,
            z2
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
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
            "checkpoints/best_mixture_testing.pt"
        )

        print(
            f"[BEST] Saved epoch {epoch+1}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size
            },
            "checkpoints/last_mixture_testing.pt"
    )

# =====================================================
# FINAL SAVE
# =====================================================


