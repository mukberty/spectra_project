import torch
from torch.utils.data import DataLoader

from models.contrastive_model import ContrastiveModel
from utils.contrastive_loss import info_nce_loss
from utils.multimodal_dataset import MultiModalDataset

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# model
model = ContrastiveModel(embed_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# dataset
dataset = MultiModalDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# training loop
for epoch in range(20):

    model.train()
    total_loss = 0

    for batch in loader:

        # ---- unpack (SAFE) ----
        if len(batch) == 3:
            ir_img, raman_img, spec = batch
        else:
            ir_img, raman_img = batch

        # ---- device move ----
        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        # spectrum may or may not exist
        if len(batch) == 3:
            spec = spec.to(device)

        # ---- forward ----
        img_emb = model.encode_image(ir_img, raman_img)

        # FIX: spectrum only if available
        if len(batch) == 3:
            spec_emb = model.encode_spectrum(spec)
            loss = info_nce_loss(img_emb, spec_emb)
        else:
            # fallback (no spec case)
            loss = img_emb.mean()

        # ---- backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1}: {total_loss/len(loader):.4f}")

# save model
torch.save(model.state_dict(), "checkpoints/contrastive_model.pt")
print("Saved: checkpoints/contrastive_model.pt")
