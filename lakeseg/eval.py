import torch
import wandb
from lakeseg.metrics import fast_hist, jaccard_index, per_class_pixel_accuracy
import torch.nn.functional as F
from torch.utils.data import DataLoader


def eval_net(net: torch.nn.Module, test_loader: DataLoader, device: torch.device, n_classes: int):
    """
    Evaluate pixel accuracy, IoU and cross-entropy loss for data in a DataLoader and log to wandb.

    :param net: Network used to evaluate test set.
    :param test_loader: DataLoader for test set.
    :param device: torch device, either cpu or cuda.
    :param n_classes: Number of classes in the dataset.
    """
    net.eval()
    tot_loss, tot_iou, tot_acc = 0, 0, 0
    n_val = len(test_loader)
    for test_batch in test_loader:
        imgs = test_batch['image']
        true_masks = test_batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.no_grad():
            mask_pred = net(imgs)

        probs = F.softmax(mask_pred, dim=1)
        argmx = torch.argmax(probs, dim=1)

        hist = fast_hist(true_masks.squeeze(0).squeeze(0), argmx.squeeze(0).to(dtype=torch.long), n_classes)

        tot_iou += jaccard_index(hist)[0]
        tot_acc += per_class_pixel_accuracy(hist)[0]
        tot_loss += F.cross_entropy(mask_pred, true_masks.squeeze(1), ignore_index=-1).item()
    avg_loss, avg_iou, avg_acc = tot_loss / n_val, tot_iou / n_val, tot_acc / n_val
    wandb.log({"Validation Loss": avg_loss})
    wandb.log({"Validation IoU": avg_iou})
    wandb.log({"Validation Accuracy": avg_acc})
    net.train()