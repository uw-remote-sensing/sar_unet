from typing import Tuple

import torch

EPS = 1e-10  # constant to ensure numerical stability


def nanmean(x: torch.Tensor) -> torch.Tensor:
    """Calculate mean of non-nan values."""
    return torch.mean(x[x == x])


def fast_hist(true: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Calculate confusion matrix."""
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist: torch.Tensor) -> torch.Tensor:
    """Calculate overall pixel accuracy."""
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-class pixel accuracy."""
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc, per_class_acc


def jaccard_index(hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate intersect over union (IoU)."""
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc, jaccard


def fw_miou(hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate frequency weighted IoU."""
    hist = hist.T
    classes = len(hist)
    class_scores = torch.zeros(classes)
    for i in range(classes):
        class_scores[i] = torch.sum(hist[i, :]) * hist[i, i] / (max(1, torch.sum(hist[i, :])))
    fmiou = torch.sum(class_scores) / torch.sum(hist)
    return fmiou, class_scores
