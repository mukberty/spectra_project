import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset
from utils.smiles_tokenizer import SmilesTokenizer


# -----------------------
# device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -----------------------
# dataset
# -----------------------
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)


# -----------------------
# tokenizer (IMPORTANT FIX)
# -----------------------
all_smiles = dataset.df["smiles"].astype(str).tolist()
tokenizer = SmilesTokenizer(all_smiles, max_len=128)

vocab_size = tokenizer.vocab_size


# -----------------------
# model
# -----------------------
model = MolCLIP(vocab_size=vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# -----------------------
# contrastive loss (INFO NCE)
# -----------------------
def contrastive_loss(z1, z2, temperature=0.07):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.T) / temperature

    labels = torch.arange(z1.size(0), device=z1.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


# -----------------------
# training loop
# -----------------------
epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    for ir_img, raman_img, smiles in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        # FIXED: proper tokenization pipeline
        smiles = tokenizer.batch_encode(smiles).to(device)

        # forward
        z_img = model.encode_image(ir_img, raman_img)
        z_txt = model.encode_smiles(smiles)

        loss = contrastive_loss(z_img, z_txt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


# -----------------------
# save model
# -----------------------
torch.save(model.state_dict(), "checkpoints/molclip_aligned.pt")
print("\nSaved molclip_aligned.pt")
