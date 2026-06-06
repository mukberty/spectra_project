import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -------------------
# setup
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)


model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(torch.load("checkpoints/image_retrieval.pt", map_location=device))
model.eval()


# -------------------
# build embeddings
# -------------------

embeddings = []
ids = []

with torch.no_grad():

    for ir_img, raman_img, mol_id in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        # IMPORTANT: use proper forward pass
        emb = model(ir_img, raman_img)

        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        ids.extend(list(mol_id))


embeddings = np.vstack(embeddings).astype("float32")
ids = np.array(ids)


# -------------------
# build FAISS index
# -------------------

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)


# -------------------
# save everything
# -------------------

import os
os.makedirs("checkpoints", exist_ok=True)

faiss.write_index(index, "checkpoints/molecule.index")
np.save("checkpoints/molecule_ids.npy", ids)

print("index built + saved successfully")
print("MODEL CHECKSUM:", sum(p.sum().item() for p in model.parameters()))
