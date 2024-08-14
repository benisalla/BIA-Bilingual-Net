import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LSRCrossEntropy(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    def __init__(self, eps: float=0.1, device: str = "cpu"):
        super(LSRCrossEntropy, self).__init__()
        self.eps = eps
        self.device = device

    def forward(self, x, y, lens):
        lens = lens.cpu()
        x = pack_padded_sequence(input=x, lengths=lens, batch_first=True, enforce_sorted=False).data.to(self.device)
        y = pack_padded_sequence(input=y, lengths=lens, batch_first=True, enforce_sorted=False).data.to(self.device)

        tv = torch.zeros_like(x).scatter(dim=1, index=y.unsqueeze(1), value=1.).to(self.device)
        tv = tv * (1. - self.eps) + self.eps / tv.size(1)

        loss = (-1 * tv * F.log_softmax(x, dim=1)).sum(dim=1)
        loss = torch.mean(loss)

        return loss