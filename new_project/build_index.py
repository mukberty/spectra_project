import os
import torch
import faiss
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

train_idx = np.load("data/splits/train_indices.npy")

train_dataset = Subset(dataset, train_idx)

loader = DataLoader(
    train_dataset,
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
ids = []


with torch.no_grad():

    for ir_img, raman_img, mol_id in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model(ir_img, raman_img)

        emb = F.normalize(emb, dim=1)

        embeddings.append(
            emb.cpu().numpy()
        )

        ids.extend(list(mol_id))


embeddings = np.vstack(embeddings).astype("float32")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(embeddings)


os.makedirs("checkpoints", exist_ok=True)

faiss.write_index(
    index,
    "checkpoints/molecule.index"
)

np.save(
    "checkpoints/molecule_ids.npy",
    np.array(ids)
)

print("\n===== INDEX BUILT =====")
print("Train molecules:", len(ids))
print("Index size:", index.ntotal)
