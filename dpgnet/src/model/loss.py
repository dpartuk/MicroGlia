import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Weighted Binary Cross-Entropy + Weighted IoU structural loss.
    Adopted from F3Net (Wei et al., AAAI 2020) and DPGNet (Wang et al., IEEE JBHI 2025/2026).

    Args:
        pred: Predicted raw logits of shape [B, 1, H, W]
        mask: Ground-truth binary mask of shape [B, 1, H, W] in {0, 1}

    Returns:
        Scalar loss tensor.
    """
    # Calculate boundary-aware weight map
    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    # Weighted Binary Cross Entropy
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted Intersection over Union (IoU)
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (wbce + wiou).mean()


class DPGNetLoss(nn.Module):
    """
    Multi-stage deep supervision loss for DPGNet.
    Computes total loss across all 5 prediction branches (TMF + 4 DUC stages).
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, mask):
        """
        Args:
            preds: Tuple/List of 5 prediction tensors [P1, P2, P3, P4, P5],
                   each with shape [B, 1, H, W]
            mask: Ground truth mask of shape [B, 1, H, W]
        """
        total_loss = 0.0
        losses = []
        for p in preds:
            # Ensure spatial dimension matches ground truth
            if p.shape[2:] != mask.shape[2:]:
                p = F.interpolate(p, size=mask.shape[2:], mode="bilinear", align_corners=False)
            loss_i = structure_loss(p, mask)
            total_loss = total_loss + loss_i
            losses.append(loss_i)

        return total_loss, losses
