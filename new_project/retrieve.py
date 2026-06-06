import numpy as np
import torch
import torch.nn.functional as F

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


model = ImageCNN(embed_dim=256)
model.load_state_dict(torch.load("checkpoints/image_retrieval.pt"))
model.eval()


db_emb = np.load("checkpoints/molecule_embeddings.npy")
db_ids = np.load("checkpoints/molecule_ids.npy")


def search(ir_img, raman_img, k=5):

    with torch.no_grad():

        ir_feat = model.backbone(ir_img)
        raman_feat = model.backbone(raman_img)

        emb = (model.ir_proj(ir_feat) + model.raman_proj(raman_feat)) / 2
        emb = F.normalize(emb, dim=1)

    emb = emb.cpu().numpy()

    sims = db_emb @ emb.T
    topk = np.argsort(-sims[:, 0])[:k]

    return db_ids[topk]


# output → molecule IDs → lookup IUPAC in parquet
