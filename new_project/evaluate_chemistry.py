import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# --------------------
# setup
# --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(torch.load("checkpoints/image_retrieval.pt", map_location=device))
model.eval()


# --------------------
# build embeddings
# --------------------

embeddings = []
smiles_list = []

with torch.no_grad():
    for ir_img, raman_img, mol_name in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model(ir_img, raman_img)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        smiles_list.extend(list(mol_name))

embeddings = np.vstack(embeddings).astype("float32")

# FAISS index
index = faiss.IndexFlatIP(256)
index.add(embeddings)


# --------------------
# chemistry metric
# --------------------

def tanimoto(sm1, sm2):
    mol1 = Chem.MolFromSmiles(sm1)
    mol2 = Chem.MolFromSmiles(sm2)

    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


# --------------------
# evaluation
# --------------------

top1_sim = []
top5_sim = []

batch_size = 512

for start in range(0, len(embeddings), batch_size):

    end = min(start + batch_size, len(embeddings))
    query = embeddings[start:end]

    scores, idxs = index.search(query, 6)

    for i in range(len(query)):

        q_idx = start + i
        q_smiles = smiles_list[q_idx]

        retrieved = []

        for j in idxs[i]:
            if j == q_idx:
                continue
            retrieved.append(smiles_list[j])
            if len(retrieved) == 5:
                break

        sims = [tanimoto(q_smiles, r) for r in retrieved]

        if len(sims) > 0:
            top1_sim.append(sims[0])
            top5_sim.extend(sims)


# --------------------
# results
# --------------------

print("\n===== CHEMISTRY RETRIEVAL =====")
print(f"Mean Top-1 Tanimoto: {np.mean(top1_sim):.4f}")
print(f"Mean Top-5 Tanimoto: {np.mean(top5_sim):.4f}")
print(f"Samples: {len(top1_sim)}")
