import os
import torch
import faiss
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

model = ImageCNN(embed_dim=256).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/image_retrieval.pt",
        map_location=device
    )
)

model.eval()

embeddings = []
names = []

with torch.no_grad():

    for ir_img, raman_img, mol_name in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model(ir_img, raman_img)
        emb = F.normalize(emb, dim=1)

        embeddings.append(
            emb.cpu().numpy()
        )

        names.extend(list(mol_name))

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatIP(256)
index.add(embeddings)

os.makedirs("checkpoints", exist_ok=True)

faiss.write_index(
    index,
    "checkpoints/full_molecule.index"
)

np.save(
    "checkpoints/full_molecule_names.npy",
    np.array(names)
)

print("\nSaved full index")
print("Vectors:", index.ntotal)
