import numpy as np
from torch.utils import data


def describe_bags(data_loader: data.DataLoader, phase: str) -> None:
    bag_lengths = []
    positive_bags = 0
    for batch_idx, (bag, label) in enumerate(data_loader):
        bag_lengths.append(int(bag.squeeze(0).size()[0]))
        positive_bags += label[0].numpy()[0]
    print(
        f'Number positive {phase} bags: {positive_bags}/{len(data_loader)}\n'
        f'Number of instances per bag, mean: {np.mean(bag_lengths)}, '
        f'min: {np.min(bag_lengths)}, max {np.max(bag_lengths)}\n'
    )
