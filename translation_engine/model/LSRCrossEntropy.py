import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LSRCrossEntropy(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss (LSRCrossEntropy).

    This loss function applies label smoothing to the standard cross-entropy loss. Label smoothing can help to
    prevent the model from becoming too confident in its predictions, which can improve generalization.

    Attributes:
        eps (float): The smoothing parameter, which controls the amount of label smoothing applied.
        device (str): The device on which the computations will be performed ('cpu' or 'cuda').
    """

    def __init__(self, eps: float = 0.1, device: str = "cpu"):
        super(LSRCrossEntropy, self).__init__()
        self.eps = eps
        self.device = device if torch.cuda.is_available() else 'cpu'

    def forward(self, x, y, lens):
        """
        Forward pass of the LSRCrossEntropy module.

        Args:
            x (torch.Tensor): The predicted logits tensor of shape (B, T, C), where B is the batch size,
                            T is the sequence length, and C is the number of classes.
            y (torch.Tensor): The target labels tensor of shape (B, T).
            lens (torch.Tensor): The lengths of each sequence in the batch, used to handle padding.

        Returns:
            torch.Tensor: The computed label smoothing cross-entropy loss.
        """

        lens = lens.cpu()
        x = pack_padded_sequence(
            input=x, lengths=lens, batch_first=True, enforce_sorted=False
        ).data.to(self.device)
        y = pack_padded_sequence(
            input=y, lengths=lens, batch_first=True, enforce_sorted=False
        ).data.to(self.device)

        tv = (
            torch.zeros_like(x)
            .scatter(dim=1, index=y.unsqueeze(1), value=1.0)
            .to(self.device)
        )
        tv = tv * (1.0 - self.eps) + self.eps / tv.size(1)

        loss = (-1 * tv * F.log_softmax(x, dim=1)).sum(dim=1)
        loss = torch.mean(loss)

        return loss
