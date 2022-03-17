import os
import torch.nn as nn
import torch
import wandb
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from lakeseg.datasets import LakesSeg
from lakeseg.models import UNet
from lakeseg.eval import eval_net
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Make numpy images and masks from SAR geotiffs and ice charts.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', metavar='D', type=str, default='../data/', help='Directory where images '
                                                                                            'and masks are located.',
                        dest='data_dir')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, default=1, dest='batch_size')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200, dest='epochs')
    parser.add_argument('-f', '--eval_frequency', metavar='F', type=int, default=3, help='How often (image frequency) '
                                                                                         'to evaluate the test set '
                                                                                         'while training.',
                        dest='eval_frequency')
    parser.add_argument('-ch', '--channels', metavar='CH', type=int, default=2, dest='n_channels')
    parser.add_argument('-cl', '--classes', metavar='CL', type=int, default=2, dest='n_classes')
    parser.add_argument('-lr', '--learning_rate', metavar='LR', type=float, default=0.0001, dest='lr')

    return parser.parse_args()


def train_unet(data_dir: str, epochs: int, batch_size: int, eval_frequency: int, n_channels: int, n_classes: int,
               lr: float):
    """
    Train UNet for lake ice segmentation. Saves epoch weights to a temporary directory 'checkpoints' and plots
    training predictions and metrics to eights and Biases (wandb). Training ignores pixels with -1 in the ground truth.

    :param data_dir: Path to where imgs/ and masks/ directories are located.
    :param epochs: Training epochs. One epoch = one round through all training images.
    :param batch_size: Number of images to feed to the UNet at a time. If image sizes vary, batch size should equal one.
    :param eval_frequency: How often to evaluate test data during an epoch.
    :param n_channels: Number of channels in images fed into UNet.
    :param n_classes:  Number of classes in the dataset. Background is considered its own class.
    :param lr: Learning rate during backpropagation.
    """
    wandb.init()

    # default device is GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}.")

    # load data
    train_set = LakesSeg(data_dir=data_dir, split='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = LakesSeg(data_dir=data_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # prepare networks and optimizer
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    net = net.to(device=device)
    wandb.watch(net)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(epochs):
        print(f"Epoch {epoch}.")
        for i, batch in enumerate(train_loader):
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            target = true_masks.to(device=device, dtype=torch.long)

            # make predictions
            masks_pred = net(imgs)
            probs = F.softmax(masks_pred, dim=1)  # softmax probabilities
            argmx = torch.argmax(probs, dim=1).to(dtype=torch.float32)  # selects class with highest probability

            # log images and preds to wandb
            example_images = [wandb.Image(imgs[0], caption='Image'),
                              wandb.Image(target.to(dtype=torch.float)[0],
                                          caption='True Mask'),
                              wandb.Image(argmx[0],
                                          caption='Predicted Mask')]
            wandb.log({"Examples": example_images})

            # backpropagation and log loss to wandb
            loss = criterion(probs, target.squeeze(1))
            wandb.log({"Training Loss": loss})
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            # evaluate test set
            if i % eval_frequency == eval_frequency-1:
                print("Testing.")
                eval_net(net, test_loader, device, n_classes)

        # save model weights to checkpoints directory, which is created if it does not exist
        # warning that checkpoints can get overwritten if they are not stored in another location after training
        try:
            os.mkdir('../checkpoints/')
        except OSError:
            pass
        torch.save(net.state_dict(), '../checkpoints/' + f'epoch{epoch + 1}.pth')


if __name__ == '__main__':
    args = get_args()
    train_unet(args.data_dir, args.epochs, args.batch_size, args.eval_frequency, args.n_channels, args.n_classes,
               args.lr)
