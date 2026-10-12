import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        # =====================================================
        # SHARED RESNET BACKBONE
        # =====================================================

        base = models.resnet18(weights=None)

        base.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        base.fc = nn.Identity()

        self.backbone = base

        # =====================================================
        # PROJECTION HEAD 1
        # =====================================================

        self.head1 = nn.Linear(
            512,
            embed_dim
        )

        self.head2 = nn.Linear(
            512,
            embed_dim
        )


    def forward(
        self,
        x_ir,
        x_raman=None
    ):

        # -------------------------------------------------
        # IR only
        # -------------------------------------------------

        feat = self.backbone(x_ir)

        emb1 = self.head1(feat)
        emb2 = self.head2(feat)

        emb1 = F.normalize(
            emb1,
            dim=1
        )

        emb2 = F.normalize(
            emb2,
            dim=1
        )

        return emb1, emb2
