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

loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = ImageCNN(embed_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def contrastive_loss(z1, z2, temp=0.1):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1)).to(device)

    return F.cross_entropy(logits, labels)


for epoch in range(20):

    model.train()
    total_loss = 0

    for ir_img, raman_img, _ in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model(ir_img, raman_img)

        loss = contrastive_loss(emb, emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1}: {total_loss/len(loader):.4f}")


torch.save(model.state_dict(), "checkpoints/image_retrieval.pt")
print("saved model")
