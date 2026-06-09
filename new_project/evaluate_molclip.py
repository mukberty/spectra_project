import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset
from utils.smiles_tokenizer import SmilesTokenizer


# -------------------
# device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -------------------
# dataset (TEST SPLIT ONLY)
# -------------------
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="test"
)

loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

print("Dataset size:", len(dataset))


# -------------------
# load model (IMPORTANT FIX)
# -------------------
ckpt = torch.load("checkpoints/molclip_aligned.pt", map_location=device)

# build vocab from dataset (must match training)
all_smiles = dataset.df["smiles"].astype(str).tolist()
tokenizer = SmilesTokenizer(all_smiles, max_len=128)

model = MolCLIP(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(ckpt, strict=True)
model.eval()


# -------------------
# load FAISS index
# -------------------
index = faiss.read_index("checkpoints/molclip.index")
ids = np.load("checkpoints/molclip_ids.npy", allow_pickle=True)

print("Index size:", index.ntotal)


# -------------------
# embedding function
# -------------------
def embed(ir, ram):
    ir = ir.to(device)
    ram = ram.to(device)
    with torch.no_grad():
        z = model.encode_image(ir, ram)
    return z.cpu().numpy().astype("float32")


# -------------------
# evaluation loop
# -------------------
top1 = 0
top5 = 0
total = 0

print("\nSTARTING EVALUATION...\n")

for i, (ir, ram, mol_id) in enumerate(loader):

    emb = embed(ir, ram)

    _, topk = index.search(emb, 5)

    for j in range(len(mol_id)):

        total += 1

        true = mol_id[j]
        retrieved = [ids[x] for x in topk[j]]

        if i == 0 and j < 3:
            print("\nDEBUG SAMPLE")
            print("TRUE:", true)
            print("TOP5:", retrieved)

        if retrieved[0] == true:
            top1 += 1

        if true in retrieved:
            top5 += 1


# -------------------
# results
# -------------------
print("\n===== FINAL =====")
print("Top-1:", top1)
print("Top-5:", top5)
print("Total:", total)

print("\nTop-1 Acc:", top1 / total if total else 0)
print("Top-5 Acc:", top5 / total if total else 0)
