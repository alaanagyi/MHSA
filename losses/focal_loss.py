import torch
from torch import nn, Tensor
from typing import Optional, Any


class FocalLoss(nn.Module):
    """
    Implement the focal loss proposed in Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002).
    """
    def __init__(
        self,
        weight: Optional[Tensor] = None, # An optional tensor for weighting classes in cross-entropy loss. Useful for addressing class imbalance.
        gamma: float = 2, #: A float indicating the focusing parameter in Focal Loss, typically set to 2.
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.gamma = gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        ce_loss = self.ce_loss_fn(y_pred, y_true) #loss for each one

        prob = torch.softmax(y_pred, dim=1) #to get the class prob
        prob = torch.gather(prob, dim=1, index=y_true.view(-1, 1)).view(-1) #to get the class probs using y_true

        focal_loss = ((1 - prob) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)
