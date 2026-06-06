import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

class ImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = efficientnet_b0(
            weights=
            EfficientNet_B0_Weights.DEFAULT
        )

        in_features = (
            self.backbone.classifier[1]
            .in_features
        )

        self.backbone.classifier = (
            nn.Identity()
        )

        self.embedding = nn.Linear(
            in_features,
            512
        )

        self.head = nn.Linear(
            512,
            20
        )

    def forward(self,x):

        f = self.backbone(x)

        emb = self.embedding(f)

        logits = self.head(emb)

        return logits, emb
