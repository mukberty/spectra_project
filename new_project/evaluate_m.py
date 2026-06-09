import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 🔥 TEST ONLY
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    split="test"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Test size:", len(dataset))

ckpt = torch.load("checkpoints/molclip.pt", map_location=device)
vocab_size = ckpt["smiles_encoder.token_emb.weight"].shape[0]

model = MolCLIP(vocab_size=vocab_size).to(device)
model.load_state_dict(ckpt, strict=True)
model.eval()

index = faiss.read_index("checkpoints/molclip.index")
ids = np.load("checkpoints/molclip_ids.npy", allow_pickle=True)

def embed(ir, ram):
    ir = ir.to(device)
    ram = ram.to(device)
    with torch.no_grad():
        return model.encode_image(ir, ram).cpu().numpy()

top1 = 0
top5 = 0
total = 0

for ir, ram, mol_id in loader:

    emb = embed(ir, ram)
    _, topk = index.search(emb, 5)

    for i in range(len(mol_id)):

        total += 1
        true = mol_id[i]
        retrieved = [ids[j] for j in topk[i]]

        if retrieved[0] == true:
            top1 += 1

        if true in retrieved:
            top5 += 1

print("\n===== RESULTS =====")
print("Top-1:", top1 / total)
print("Top-5:", top5 / total)
print("Total:", total)
