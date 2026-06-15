import torch

ckpt = torch.load(
    "checkpoints/best_model.pt",
    map_location="cpu"
)

print(ckpt.keys())
