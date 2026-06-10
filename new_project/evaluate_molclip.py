import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -----------------------
# DATASET (TEST ONLY)
# -----------------------
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="all"
)

loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

print("[Dataset] Size:", len(dataset))


# -----------------------
# LOAD CHECKPOINT
# -----------------------
ckpt = torch.load("checkpoints/molclip_aligned.pt", map_location=device)

vocab_size = ckpt["smiles_encoder.token_emb.weight"].shape[0]

model = MolCLIP(vocab_size=vocab_size).to(device)
model.load_state_dict(ckpt, strict=False)
model.eval()

print("Loaded checkpoint keys:", len(ckpt.keys()))


# -----------------------
# FAISS INDEX
# -----------------------
index = faiss.read_index("checkpoints/molclip.index")
ids = np.load("checkpoints/molclip_ids.npy", allow_pickle=True)
print(ids[:10])

print("Index size:", index.ntotal)


# -----------------------
# ENCODER
# -----------------------
def embed(ir, ram):
    ir = ir.to(device)
    ram = ram.to(device)
    with torch.no_grad():
        z = model.encode_image(ir, ram)
    return z.cpu().numpy().astype("float32")


# -----------------------
# EVAL METRICS
# -----------------------
top1 = 0
top5 = 0
total = 0

print("\nSTARTING EVAL...\n")


for i, (ir, ram, mol_id) in enumerate(loader):

    emb = embed(ir, ram)
    _, topk = index.search(emb, 5)

    for j in range(len(mol_id)):

        total += 1

        true = str(mol_id[j])
        retrieved = [str(ids[x]) for x in topk[j]]

        if i == 0 and j < 3:
            print("\nDEBUG SAMPLE")
            print("TRUE:", repr(true))
            print("TOP5:", [repr(x) for x in retrieved])
            print("MATCH:", true in retrieved)

        if retrieved[0] == true:
            top1 += 1

        if true in retrieved:
            top5 += 1


# -----------------------
# FINAL RESULTS
# -----------------------
print("\n===== FINAL =====")
print("Top-1:", top1)
print("Top-5:", top5)
print("Total:", total)

print("Top-1 Acc:", top1 / total if total else 0)
print("Top-5 Acc:", top5 / total if total else 0)

print(dataset[0][2])
print(dataset[1][2])
print(dataset[2][2])

