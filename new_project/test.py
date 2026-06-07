import torch

ckpt = torch.load("checkpoints/molclip.pt", map_location="cpu")

for k, v in ckpt.items():
    if "token_emb" in k:
        print(k, v.shape)
