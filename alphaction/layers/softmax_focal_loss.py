import torch
from torch import nn


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, reduction="mean"):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = probs[range(targets.shape[0]), targets]
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * (targets == 1).float() + (1 - self.alpha) * (targets == 0).float()
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
