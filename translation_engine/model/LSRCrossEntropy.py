import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LSRCrossEntropy(torch.nn.Module):
    """
    Label Smoothing Cross Entropy Loss (LSRCrossEntropy).

    This loss function applies label smoothing to the standard cross-entropy loss. Label smoothing can help to
    prevent the model from becoming too confident in its predictions, which can improve generalization.

    Attributes:
        eps (float): The smoothing parameter, which controls the amount of label smoothing applied.
        device (str): Kept for backward compatibility; the loss runs on the device of its inputs.
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

        lens = lens.cpu()  # pack_padded_sequence wants lengths on the CPU
        x = pack_padded_sequence(
            input=x, lengths=lens, batch_first=True, enforce_sorted=False
        ).data  # [n_tokens, C]: only the real (non-padding) positions
        y = pack_padded_sequence(
            input=y, lengths=lens, batch_first=True, enforce_sorted=False
        ).data  # [n_tokens]

        # smoothed target: (1 - eps) on the true class + eps spread over all C classes.
        # loss = -sum_c tv_c * log p_c = -(1 - eps) * log p_y - (eps / C) * sum_c log p_c,
        # computed without building the dense [n_tokens, C] target matrix (C = 50k here)
        log_p = F.log_softmax(x, dim=1)
        nll = -log_p.gather(dim=1, index=y.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=1)
        loss = (1.0 - self.eps) * nll + self.eps * smooth
        loss = torch.mean(loss)  # mean over real tokens

        return loss
