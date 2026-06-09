import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # 🔥 MUST MATCH CHECKPOINT
        self.ir_proj = nn.Linear(512, embed_dim)
        self.raman_proj = nn.Linear(512, embed_dim)

    def forward(self, ir, raman):

        ir_feat = self.backbone(ir)
        rm_feat = self.backbone(raman)

        ir_emb = self.ir_proj(ir_feat)
        rm_emb = self.raman_proj(rm_feat)

        return F.normalize((ir_emb + rm_emb) / 2, dim=1)


class MolCLIP(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.image_encoder = ImageCNN(embed_dim=256)

    def encode_image(self, ir_img, raman_img):
        return self.image_encoder(ir_img, raman_img)

    def forward(self, ir_img, raman_img):
        return self.encode_image(ir_img, raman_img)
