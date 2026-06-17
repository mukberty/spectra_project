import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        # separate backbones (IMPORTANT FIX)
        self.ir_backbone = models.resnet18(weights=None)
        self.raman_backbone = models.resnet18(weights=None)

        # modify input conv
        self.ir_backbone.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.raman_backbone.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        # remove classifiers
        self.ir_backbone.fc = nn.Identity()
        self.raman_backbone.fc = nn.Identity()

        # projection heads (slightly deeper = IMPORTANT)
        self.ir_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        self.raman_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def encode_image(self, x_ir, x_raman):

        assert x_ir.dim() == 4
        assert x_raman.dim() == 4

        ir_feat = self.ir_backbone(x_ir)
        raman_feat = self.raman_backbone(x_raman)

        ir_emb = self.ir_proj(ir_feat)
        raman_emb = self.raman_proj(raman_feat)

        # fuse AFTER projection (not before)
        emb = ir_emb + raman_emb

        return F.normalize(emb, dim=1)

    def forward(self, x_ir, x_raman):
        return self.encode_image(x_ir, x_raman)
