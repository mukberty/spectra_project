import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle

from models.molclip_model_mixture import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -----------------------
# dataset (TRAIN ONLY)
# -----------------------
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="all"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

print("Train size:", len(dataset))


# -----------------------
# LOAD TOKENIZER (FIX)
# -----------------------
with open("checkpoints/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------
# model
# -----------------------
ckpt = torch.load(
    "checkpoints/best_mixture_testing.pt",
    map_location=device
)

model = MolCLIP(
    vocab_size=ckpt["vocab_size"]
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print("Loaded epoch:", ckpt["epoch"])
print("Saved loss:", ckpt["loss"])


# -----------------------
# embeddings
# -----------------------
embeddings = []
ids = []

with torch.no_grad():
    for ir, ram, smiles in loader:

        ir = ir.to(device)
        ram = ram.to(device)

        emb = model.encode_smiles(
            tokenizer.batch_encode(smiles).to(device)
        )

        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        ids.extend(list(smiles))


embeddings = np.vstack(embeddings).astype("float32")
ids = np.array(ids)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

#faiss.write_index(index, "checkpoints/molclip.index")
#np.save("checkpoints/molclip_ids.npy", ids)

faiss.write_index(
    index,
    "checkpoints/molclip_mixture_testing3.index"
)

np.save(
    "checkpoints/molclip_mixture_testing3_ids.npy",
    ids
)

print("INDEX BUILT:", index.ntotal)
print("MODEL: best_mixture_testing.pt")
print("Vocab size:", tokenizer.vocab_size)
print("Index size:", len(dataset))
