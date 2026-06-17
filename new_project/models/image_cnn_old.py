import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        base = models.resnet18(weights=None)

        base.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        base.fc = nn.Identity()

        self.backbone = base

        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x_ir, x_raman=None):

        # ignore raman OR treat as same modality
        x = x_ir

        feat = self.backbone(x)
        emb = self.proj(feat)

        return F.normalize(emb, dim=1)
