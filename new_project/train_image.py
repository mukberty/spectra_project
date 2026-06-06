import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


model = ImageCNN(embed_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def contrastive_loss(ir_emb, raman_emb, temp=0.1):

    ir_emb = F.normalize(ir_emb, dim=1)
    raman_emb = F.normalize(raman_emb, dim=1)

    logits = torch.matmul(ir_emb, raman_emb.T) / temp
    targets = torch.arange(ir_emb.size(0)).to(ir_emb.device)

    loss_ir_to_ram = F.cross_entropy(logits, targets)
    loss_ram_to_ir = F.cross_entropy(logits.T, targets)

    return (loss_ir_to_ram + loss_ram_to_ir) / 2


for epoch in range(20):

    model.train()
    total_loss = 0

    for ir_img, raman_img, _ in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        # IMPORTANT FIX: get embeddings separately (correct signal)
        ir_feat = model.backbone(ir_img)
        raman_feat = model.backbone(raman_img)

        ir_emb = model.ir_proj(ir_feat)
        raman_emb = model.raman_proj(raman_feat)

        loss = contrastive_loss(ir_emb, raman_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1}: {total_loss/len(loader):.4f}")


torch.save(model.state_dict(), "checkpoints/image_retrieval.pt")
print("saved model")
