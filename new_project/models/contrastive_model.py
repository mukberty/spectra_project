import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectra_cnn import SpectraCNN
from models.image_cnn import ImageCNN


class ContrastiveModel(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        self.image_encoder = ImageCNN(embed_dim)
        self.spectrum_encoder = SpectraCNN(embed_dim)

    def encode_image(self, ir_img, raman_img):

        # safety checks (IMPORTANT for debugging)
        assert ir_img.dim() == 4, f"IR must be [B,3,224,224], got {ir_img.shape}"
        assert raman_img.dim() == 4, f"Raman must be [B,3,224,224], got {raman_img.shape}"

        ir_emb = self.image_encoder.ir_backbone(ir_img)
        raman_emb = self.image_encoder.raman_backbone(raman_img)

        ir_emb = self.image_encoder.ir_proj(ir_emb)
        raman_emb = self.image_encoder.raman_proj(raman_emb)

        emb = (ir_emb + raman_emb) / 2.0
        return F.normalize(emb, dim=1)

    def encode_spectrum(self, x):

        # safety check
        assert x.dim() == 2, f"Spectrum must be [B,1801], got {x.shape}"

        z = self.spectrum_encoder(x)
        return F.normalize(z, dim=1)
