import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle

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
    image_dir="data/plots/continuous_spectra",
    mode="all"
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
# tokenizer (TRAIN ONLY SOURCE OF TRUTH)
# -----------------------
all_smiles = dataset.df["smiles"].astype(str).tolist()
tokenizer = SmilesTokenizer(all_smiles, max_len=128)

vocab_size = tokenizer.vocab_size


# SAVE TOKENIZER (CRITICAL FIX)
with open("checkpoints/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("[INFO] tokenizer saved")


# -----------------------
# model
# -----------------------
model = MolCLIP(vocab_size=vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


def contrastive_loss(z_img, z_txt, temperature=0.05):

    z_img = F.normalize(z_img, dim=1)
    z_txt = F.normalize(z_txt, dim=1)

    logits = (z_img @ z_txt.T) / temperature
    labels = torch.arange(z_img.size(0), device=z_img.device)

    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.T, labels)

    return (loss_img + loss_txt) / 2


# -----------------------
# training
# -----------------------
epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for ir_img, raman_img, smiles in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        smiles = tokenizer.batch_encode(smiles).to(device)

        z_img = model.encode_image(ir_img, raman_img)
        z_txt = model.encode_smiles(smiles)

        loss = contrastive_loss(z_img, z_txt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")


torch.save(model.state_dict(), "checkpoints/molclip_aligned.pt")
print("Saved molclip_aligned.pt")
