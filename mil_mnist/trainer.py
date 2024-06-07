from __future__ import print_function

import torch
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm

from metrics import MetricsManager


def train(
    model: nn.Module,
    data_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    use_cuda: bool,
    metrics: MetricsManager,
) -> float:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0.0

    for batch_idx, (bag, label) in enumerate(tqdm(data_loader)):
        bag_label = label[0]
        bag_label = bag_label.type(torch.FloatTensor)
        bag, bag_label = bag.to(device), bag_label.to(device)

        optimizer.zero_grad()
        Y_pred, Y_hat = model(bag)
        loss = loss_func(Y_pred, bag_label)
        loss.backward()
        optimizer.step()
        metrics.update(Y_hat, bag_label)

        train_loss += loss.item()

    train_loss /= len(data_loader)
    return train_loss


def test(
    model: nn.Module,
    data_loader: data.DataLoader,
    loss_func: nn.Module,
    use_cuda: bool,
    metrics: MetricsManager,
) -> float:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (bag, label) in enumerate(data_loader):
            bag_label, instance_labels = label

            bag_label = bag_label.type(torch.FloatTensor)
            bag, bag_label = bag.to(device), bag_label.to(device)
            Y_pred, Y_hat = model(bag)
            loss = loss_func(Y_pred, bag_label)
            test_loss += loss.item()
            metrics.update(Y_hat, bag_label)

    test_loss /= len(data_loader)
    return test_loss


def get_optimizer(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.reg
        )

    elif args.opt == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg
        )
    else:
        raise NotImplementedError
    return optimizer
