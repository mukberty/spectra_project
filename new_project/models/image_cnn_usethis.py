import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):

        super().__init__()

        # -------------------------
        # IR backbone
        # -------------------------

        ir_backbone = models.resnet18(
            weights=None
        )

        ir_backbone.fc = nn.Identity()

        self.ir_backbone = ir_backbone

        # -------------------------
        # Raman backbone
        # -------------------------

        raman_backbone = models.resnet18(
            weights=None
        )

        raman_backbone.fc = nn.Identity()

        self.raman_backbone = raman_backbone

        # -------------------------
        # Fusion layer
        # -------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                512 + 512,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                embed_dim
            )
        )

    def forward(
        self,
        x_ir,
        x_raman
    ):

        ir_feat = self.ir_backbone(
            x_ir
        )

        raman_feat = self.raman_backbone(
            x_raman
        )

        combined = torch.cat(
            [
                ir_feat,
                raman_feat
            ],
            dim=1
        )

        emb = self.fusion(
            combined
        )

        return F.normalize(
            emb,
            dim=1
        )
