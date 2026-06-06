import torch
import torch.nn.functional as F

def info_nce_loss(img_emb, spec_emb, temperature=0.07):
    logits = img_emb @ spec_emb.T / temperature

    labels = torch.arange(img_emb.size(0)).to(img_emb.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2
