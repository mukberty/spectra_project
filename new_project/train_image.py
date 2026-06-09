import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -----------------------
# setup
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

torch.cuda.empty_cache()


dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)


# 🔥 REDUCED BATCH SIZE (CRITICAL FIX)
loader = DataLoader(
    dataset,
    batch_size=32,   # was 128
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)


model = ImageCNN(embed_dim=256).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)


# -----------------------
# CLIP LOSS (STABLE)
# -----------------------

def clip_loss(z_ir, z_raman, temp=0.07):

    z_ir = F.normalize(z_ir, dim=1)
    z_raman = F.normalize(z_raman, dim=1)

    logits = (z_ir @ z_raman.T) / temp

    labels = torch.arange(z_ir.size(0), device=z_ir.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_r = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_r) / 2


# -----------------------
# TRAIN LOOP (MEMORY SAFE)
# -----------------------

for epoch in range(30):

    model.train()
    total_loss = 0

    for ir_img, raman_img, _ in loader:

        ir_img = ir_img.to(device, non_blocking=True)
        raman_img = raman_img.to(device, non_blocking=True)

        # forward
        ir_feat = model.ir_backbone(ir_img)
        raman_feat = model.raman_backbone(raman_img)

        z_ir = model.ir_proj(ir_feat)
        z_raman = model.raman_proj(raman_feat)

        loss = clip_loss(z_ir, z_raman)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1}: {total_loss/len(loader):.4f}")


# -----------------------
# SAVE
# -----------------------

torch.save(model.state_dict(), "checkpoints/image_retrieval.pt")
print("saved model")
