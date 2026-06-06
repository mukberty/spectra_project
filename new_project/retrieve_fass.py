import numpy as np
import faiss
import torch
import torch.nn.functional as F

from models.image_cnn import ImageCNN


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(torch.load("checkpoints/image_retrieval.pt"))
model.eval()

index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy")


def embed(ir_img, raman_img):

    with torch.no_grad():

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        ir_feat = model.backbone(ir_img)
        raman_feat = model.backbone(raman_img)

        ir_emb = model.ir_proj(ir_feat)
        raman_emb = model.raman_proj(raman_feat)

        emb = (ir_emb + raman_emb) / 2
        emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


def search(ir_img, raman_img, k=5):

    q = embed(ir_img, raman_img)
    faiss.normalize_L2(q)

    scores, idx = index.search(q, k)

    return ids[idx[0]], scores[0]
