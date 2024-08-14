import torch
from torch import nn

class LSRCrossEntropy(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    def __init__(self, eps=0.1):
        super(LSRCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, x, y, lens):
        lens = lens.cpu()
        x = pack_padded_sequence(input=x, lengths=lens, batch_first=True, enforce_sorted=False).data.to(device)
        y = pack_padded_sequence(input=y, lengths=lens, batch_first=True, enforce_sorted=False).data.to(device)

        tv = torch.zeros_like(x).scatter(dim=1, index=y.unsqueeze(1), value=1.).to(device)
        tv = tv * (1. - self.eps) + self.eps / tv.size(1)

        loss = (-1 * tv * F.log_softmax(x, dim=1)).sum(dim=1)
        loss = torch.mean(loss)

        return loss