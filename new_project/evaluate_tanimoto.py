import torch
import numpy as np
import pandas as pd
import faiss

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -----------------------------
# LOAD MODEL
# -----------------------------

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(torch.load(
    "checkpoints/image_retrieval.pt",
    map_location=device
))
model.eval()


# -----------------------------
# DATASET
# IMPORTANT: USE CLEAN PARQUET
# -----------------------------

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/clean_smiles.parquet",   # 👈 CHANGE ONLY THIS FILE
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)


# -----------------------------
# LOAD FAISS INDEX
# -----------------------------

index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy", allow_pickle=True)


# -----------------------------
# SAFE TANIMOTO FUNCTION
# -----------------------------

def tanimoto(smi1, smi2):

    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)

    if m1 is None or m2 is None:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def get_embedding(ir, raman):

    ir = ir.to(device)
    raman = raman.to(device)

    with torch.no_grad():
        emb = model(ir, raman)

    emb = emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)

    return emb


# -----------------------------
# EVALUATION
# -----------------------------

all_scores_top1 = []
all_scores_top5 = []

total = 0

for ir_img, raman_img, mol_id in loader:

    query_emb = get_embedding(ir_img, raman_img)

    scores, topk = index.search(query_emb, 5)

    for i in range(len(mol_id)):

        true_id = mol_id[i]

        retrieved_ids = [ids[j] for j in topk[i]]

        # skip invalid cases safely
        sims = []

        for rid in retrieved_ids:
            sim = tanimoto(true_id, rid)
            if sim is not None:
                sims.append(sim)

        if len(sims) == 0:
            continue

        # top-1
        if retrieved_ids[0] == true_id:
            all_scores_top1.append(1)
        else:
            all_scores_top1.append(0)

        # top-5 chemical similarity
        all_scores_top5.append(max(sims))

        total += 1


# -----------------------------
# RESULTS
# -----------------------------

print("\n===== CHEMISTRY RETRIEVAL =====")
print(f"Mean Top-1 Match Rate: {np.mean(all_scores_top1):.4f}")
print(f"Mean Top-5 Tanimoto: {np.mean(all_scores_top5):.4f}")
print(f"Evaluated samples: {total}")
