import argparse

import torch
from torch import nn
from torch.utils import data

from constants import RESULTS_DIR
from data import MnistBags
from data_loader import describe_bags
from metrics import MetricsManager, combine_metrics
from model import MultiHeadAttentionNetwork
from trainer import EarlyStopping, get_optimizer, eval, train
from visualization import plot_loss, plot_metrics

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(
        description='"Contains at least one 7": MNIST Multiple Instance Learning'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        metavar='N',
        help='number of epochs to train (default: 20)'
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
        default=7,
        metavar='T',
        help='bags have positive labels if they contain at least one such number',
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
        '--num_bags_val',
        type=int,
        default=100,
        metavar='NVal',
        help='number of bags in validation set',
    )
    parser.add_argument(
        '--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set'
    )
    parser.add_argument(
        '--seed', type=int, default=64210987, metavar='S', help='random seed (default: 123423)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False, help='disables CUDA training'
    )
    parser.add_argument('--gated', type=bool, default=False, help='turns on gated attentions')
    parser.add_argument('--use_dropout', type=bool, default=False, help='whether to use dropout')
    parser.add_argument(
        '--temperature',
        type=float,
        nargs='+',
        default=[1, 1],
        help='list of temperatures',
    )
    parser.add_argument(
        '--head_size',
        type=str,
        default='small',
        choices=['tiny', 'small', 'same', 'None'],
        help='head size',
    )
    parser.add_argument(
        '--opt',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='optimizer'
    )
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_val_dataset = MnistBags(
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        std_bag_length=args.std_bag_length,
        min_bag_length=args.min_bag_length,
        max_bag_length=args.max_bag_length,
        num_bag=args.num_bags_train + args.num_bags_val,
        seed=args.seed,
        train=True,
    )
    train_dataset, val_dataset = data.random_split(
        train_val_dataset,
        [args.num_bags_train, args.num_bags_val],
    )

    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

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

    describe_bags(train_loader, 'train')
    describe_bags(val_loader, 'validation')
    describe_bags(test_loader, 'test')

    print('Initialize Model')
    model = MultiHeadAttentionNetwork(
        args.gated,
        args.use_dropout,
        args.temperature,
        args.head_size
    )
    if args.cuda:
        model.cuda()

    optimizer = get_optimizer(model, args)
    criterion = nn.BCELoss()
    metrics = MetricsManager()
    early_stopping = EarlyStopping(2)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    print('Start Training')
    for epoch in range(args.epochs):
        metrics.reset()
        train_losses.append(train(model, train_loader, optimizer, criterion, args.cuda, metrics))
        train_metrics.append(metrics.compute())
        metrics_str = ', '.join([f'{key}: {value}' for key, value in train_metrics[-1].items()])
        print(
            f'Epoch: {epoch + 1}/{args.epochs}, Train loss: {train_losses[-1]:.4f}, {metrics_str}'
        )

        metrics.reset()
        val_losses.append(eval(model, val_loader, criterion, args.cuda, metrics))
        val_metrics.append(metrics.compute())
        metrics_str = ', '.join([f'{key}: {value}' for key, value in val_metrics[-1].items()])
        print(f'Validation Set, Loss: {val_losses[-1]:.4f}, {metrics_str}')

        if early_stopping.should_stop(model, val_losses[-1]):
            break

    train_metrics_combined = combine_metrics(train_metrics)
    val_metrics_combined = combine_metrics(val_metrics)

    model.load_state_dict(torch.load(RESULTS_DIR.joinpath('checkpoint.pt')))

    print('\nStart Testing')
    metrics.reset()
    test_loss = eval(model, test_loader, criterion, args.cuda, metrics)
    test_metrics = metrics.compute()
    metrics_str = ', '.join([f'{key}: {value}' for key, value in test_metrics.items()])
    print(f'Test Set, Loss: {test_loss:.4f}, {metrics_str}')

    plot_loss(train_losses, val_losses).write_image(RESULTS_DIR.joinpath('losses.png'), scale=3)
    plot_metrics(
        train_metrics_combined,
        val_metrics_combined,
    ).write_image(RESULTS_DIR.joinpath('metrics.png'), scale=3)
