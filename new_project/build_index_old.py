import torch
import numpy as np
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(torch.load("checkpoints/image_retrieval.pt"))
model.eval()


embeddings = []
ids = []

with torch.no_grad():

    for ir_img, raman_img, mol_id in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        ir_feat = model.backbone(ir_img)
        raman_feat = model.backbone(raman_img)

        ir_emb = model.ir_proj(ir_feat)
        raman_emb = model.raman_proj(raman_feat)

        emb = (ir_emb + raman_emb) / 2
        emb = torch.nn.functional.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        ids.extend(mol_id)


embeddings = np.vstack(embeddings)

np.save("checkpoints/molecule_embeddings.npy", embeddings)
np.save("checkpoints/molecule_ids.npy", np.array(ids))

print("index built")
