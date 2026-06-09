import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset
from utils.smiles_tokenizer import SmilesTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -----------------------
# dataset (TRAIN ONLY)
# -----------------------
dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra",
    mode="train"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

print("Train size:", len(dataset))


# -----------------------
# build SMILES lookup (FIX)
# -----------------------
df = dataset.df.copy()

# IMPORTANT: map molecule_id -> smiles correctly
if "molecule_name" in df.columns:
    smiles_lookup = dict(zip(df["molecule_name"], df["smiles"]))
else:
    smiles_lookup = dict(zip(df.index, df["smiles"]))


# -----------------------
# tokenizer
# -----------------------
all_smiles = df["smiles"].astype(str).tolist()
tokenizer = SmilesTokenizer(all_smiles, max_len=128)


# -----------------------
# model
# -----------------------
ckpt = torch.load("checkpoints/molclip_aligned.pt", map_location=device)

model = MolCLIP(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(ckpt, strict=True)
model.eval()


# -----------------------
# build embeddings
# -----------------------
embeddings = []
ids = []

with torch.no_grad():

    for ir, ram, mol_id in loader:

        ir = ir.to(device)
        ram = ram.to(device)

        # FIXED: safe lookup instead of .loc
        smiles_batch = [smiles_lookup[mid] for mid in mol_id]
        smiles_batch = tokenizer.batch_encode(smiles_batch).to(device)

        emb = model.encode_smiles(smiles_batch)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        ids.extend(list(mol_id))


embeddings = np.vstack(embeddings).astype("float32")
ids = np.array(ids)


# -----------------------
# FAISS
# -----------------------
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "checkpoints/molclip.index")
np.save("checkpoints/molclip_ids.npy", ids)

print("INDEX BUILT:", index.ntotal)
