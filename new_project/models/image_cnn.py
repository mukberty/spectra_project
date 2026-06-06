import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        # shared backbone (IMPORTANT FIX)
        base_model = models.resnet18(weights=None)

        # modify input conv (RGB images)
        base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # remove classifier
        base_model.fc = nn.Identity()

        # use SAME backbone for IR + Raman (critical fix)
        self.backbone = base_model

        # projection heads
        self.ir_proj = nn.Linear(512, embed_dim)
        self.raman_proj = nn.Linear(512, embed_dim)

    def encode_image(self, x_ir, x_raman):

        # safety checks
        assert x_ir.dim() == 4, f"IR must be [B,3,224,224], got {x_ir.shape}"
        assert x_raman.dim() == 4, f"Raman must be [B,3,224,224], got {x_raman.shape}"

        ir_feat = self.backbone(x_ir)
        raman_feat = self.backbone(x_raman)

        ir_emb = self.ir_proj(ir_feat)
        raman_emb = self.raman_proj(raman_feat)

        emb = (ir_emb + raman_emb) / 2.0

        return F.normalize(emb, dim=1)

    def forward(self, x_ir, x_raman):
        return self.encode_image(x_ir, x_raman)
