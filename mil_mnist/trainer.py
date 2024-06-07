from __future__ import print_function

from typing import Iterable

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm


def train(
    num_epochs: int,
    model: nn.Module,
    data_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    use_cuda: bool,
) -> None:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    for epoch in range(1, num_epochs + 1):

        model.train()
        train_loss = 0.0

        for batch_idx, (data, label) in enumerate(tqdm(data_loader)):
            bag_label = label[0]
            bag_label = bag_label.type(torch.LongTensor)
            data, bag_label = data.to(device), bag_label.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(data)  # Y_prob, Y_hat
            loss = loss_func(logits, bag_label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(data_loader)

        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}')


def test(model: nn.Module, data_loader: data.DataLoader, loss_func: nn.Module, use_cuda: bool):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            bag_label, instance_labels = label
            bag_label = bag_label.type(torch.LongTensor)
            data, bag_label = data.to(device), bag_label.to(device)
            logits, Y_prob, Y_hat = model(data)
            loss = loss_func(logits, bag_label)
            test_loss += loss.item()

            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (
                    bag_label.cpu().data.numpy()[0], int(Y_hat.cpu().data.numpy()[0][0])
                )
                """instance_level = list(
                    zip(
                        instance_labels.numpy()[0].tolist(),
                        np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()
                    )
                )"""

                print(
                    f'\nTrue Bag Label, Predicted Bag Label: {bag_level}'
                    # '\nTrue Instance Labels, Attention Weights: {}'.format(bag_level, instance_level)
                )

    test_loss /= len(data_loader)
    print(f'\nTest Set, Loss: {test_loss:.4f}')


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
