import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from data_loader import MnistBags
from trainer import get_optimizer, test, train
from model import MultiHeadAttentionNetwork

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(
        description='"Contains at least one 7": MNIST Multiple Instance Learning'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)'
    )
    parser.add_argument(
        '--reg', type=float, default=10e-5, metavar='R', help='weight decay'
    )
    parser.add_argument(
        '--target_number',
        type=int,
        default=9,
        metavar='T',
        help='bags have a positive labels if they contain at least one such number',
    )
    parser.add_argument(
        '--mean_bag_length', type=int, default=10, metavar='ML', help='average bag length'
    )
    parser.add_argument(
        '--std_bag_length',
        type=int,
        default=2,
        metavar='VL',
        help='standard deviation of bag length'
    )
    parser.add_argument(
        '--min_bag_length', type=int, default=5, metavar='MINL', help='minimal bag length'
    )
    parser.add_argument(
        '--max_bag_length', type=int, default=250000000, metavar='MAXL', help='maximal bag length'
    )
    parser.add_argument(
        '--num_bags_train',
        type=int,
        default=200,
        metavar='NTrain',
        help='number of bags in training set',
    )
    parser.add_argument(
        '--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set'
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False, help='disables CUDA training'
    )
    parser.add_argument('--gated', type=bool, default=False, help='turns on gated attentions')
    parser.add_argument('--size', type=str, default='small', help='network size')
    parser.add_argument('--use_dropout', type=bool, default=False, help='whether to use dropout')
    parser.add_argument(
        '--temperature',
        type=float,
        nargs='+',
        default=[1, 1],
        help='list of temperatures',
    )
    parser.add_argument('--head_size', type=str, default='small', help='head size')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = data.DataLoader(
        MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            std_bag_length=args.std_bag_length,
            min_bag_length=args.min_bag_length,
            max_bag_length=args.max_bag_length,
            num_bag=args.num_bags_train,
            seed=args.seed,
            train=True,
        ),
        batch_size=1,
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = data.DataLoader(
        MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            std_bag_length=args.std_bag_length,
            min_bag_length=args.min_bag_length,
            max_bag_length=args.max_bag_length,
            num_bag=args.num_bags_test,
            seed=args.seed,
            train=False,
        ),
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print(
        'Number positive train bags: {}/{}\n'
        'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
            mnist_bags_train, len(train_loader),
            np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)
        )
    )

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print(
        'Number positive test bags: {}/{}\n'
        'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
            mnist_bags_test, len(test_loader),
            np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)
        )
    )

    print('Init Model')
    model = MultiHeadAttentionNetwork(
        args.gated,
        args.size,
        args.use_dropout,
        args.temperature,
        args.head_size
    )
    if args.cuda and torch.cuda.is_available():
        model.cuda()

    optimizer = get_optimizer(model, args)
    loss_func = nn.CrossEntropyLoss()
    print('Start Training')
    train(args.epochs, model, train_loader, optimizer, loss_func, args.cuda)
    print('Start Testing')
    test(model, test_loader, loss_func, args.cuda)
