import torch.nn as nn
import torch.nn.functional as F

from models.image_cnn_mixture import ImageCNN
from models.smiles_encoder import SmilesEncoder


class MolCLIP(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.image_encoder = ImageCNN(
            embed_dim=256
        )

        self.smiles_encoder = SmilesEncoder(
            vocab_size=vocab_size,
            embed_dim=256
        )

    # =====================================================
    # IMAGE
    # =====================================================

    def encode_image(
        self,
        ir_img,
        raman_img=None
    ):

        z1, z2 = self.image_encoder(
            ir_img,
            raman_img
        )

        return (
            F.normalize(z1, dim=1),
            F.normalize(z2, dim=1)
        )

    # =====================================================
    # SMILES
    # =====================================================

    def encode_smiles(
        self,
        smiles
    ):

        z = self.smiles_encoder(
            smiles
        )

        return F.normalize(
            z,
            dim=1
        )
