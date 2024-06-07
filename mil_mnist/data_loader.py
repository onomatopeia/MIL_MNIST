import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

from constants import DATASETS_DIR


class MnistBags(data.Dataset):
    def __init__(
        self,
        target_number=7,
        mean_bag_length=10,
        std_bag_length=2,
        min_bag_length=5,
        max_bag_length=250000000,
        num_bag=250,
        seed=1,
        train=True
    ) -> None:
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.std_bag_length = std_bag_length
        self.min_bag_length = min_bag_length
        self.max_bag_length = max_bag_length
        self.num_bag = num_bag
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.num_samples = 60000 if self.train else 10000
        self.bags, self.labels = self._create_bags()

    def _create_bags(self):
        loader = data.DataLoader(
            datasets.MNIST(
                DATASETS_DIR,
                train=self.train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=self.num_samples,
            shuffle=False
        )

        all_images, all_labels = next(iter(loader))

        bags_list = []
        labels_list = []

        bag_sizes = self.rng.normal(
            loc=self.mean_bag_length,
            scale=self.std_bag_length,
            size=self.num_bag,
        )
        bag_sizes = np.round(bag_sizes).astype(int)
        bag_sizes = np.clip(bag_sizes, self.min_bag_length, self.max_bag_length)
        for bag_size in bag_sizes:
            indices = torch.LongTensor(
                self.rng.integers(0, self.num_samples, bag_size, endpoint=True)
            )
            bags_list.append(all_images[indices])
            labels_list.append((all_labels[indices] == self.target_number).int())

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.bags[index], [max(self.labels[index]), self.labels[index]]
