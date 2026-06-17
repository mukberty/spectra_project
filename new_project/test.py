import torch

ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")

for k in ckpt["model_state_dict"].keys():
    if "raman_backbone" in k:
        print(k)
        break
