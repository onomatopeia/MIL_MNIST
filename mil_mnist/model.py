import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionNetwork(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(AttentionNetwork, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x: Tensor):
        return self.module(x), x  # N x n_classes


class GatedAttentionNetwork(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(GatedAttentionNetwork, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: Tensor) -> tuple:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class MultiHeadAttentionNetwork(nn.Module):
    def __init__(
        self,
        gated: bool = True,
        size_arg: str = "small",
        dropout: bool = False,
        temperature: Sequence[float] = (1, 1),
        head_size: str = "small"
    ) -> None:
        """
        Multihead ABMIL model with separate attention modules.

        Args:
            gated (bool): whether to use gated attention network
            size_arg (str): config for network size
            dropout (bool): whether to use dropout
            temperature (sequence): temperature scaling values for each head
            head_size (str): size of each head
        """
        super().__init__()
        self.n_heads = len(temperature)
        self.size_dict = {"small": [784, 128, 256]}
        self.size = self.size_dict[size_arg]
        self.temperature = temperature

        if self.size[1] % self.n_heads != 0:
            print(
                "The feature dim should be divisible by num_heads!! Don't worry, we will fix it for you."
            )
            self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads

        self.step = self.size[1] // self.n_heads

        if head_size == "tiny":
            self.dim = self.step // 4
        elif head_size == "small":
            self.dim = self.step // 2
        elif head_size == "same":
            self.dim = self.size[2]
        else:
            self.dim = self.step

        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gated:
            att_net = [
                GatedAttentionNetwork(L=self.step, D=self.dim, dropout=dropout, n_classes=1)
                for _ in range(self.n_heads)
            ]
        else:
            att_net = [
                AttentionNetwork(L=self.step, D=self.dim, dropout=dropout, n_classes=1)
                for _ in range(self.n_heads)
            ]

        self.net_general = nn.Sequential(*fc)
        self.attention_net = nn.ModuleList(att_net)
        self.classifiers = nn.Linear(self.size[1], 1)
        initialize_weights(self)

    def forward(self, h):
        """
        Forward pass of the model.

        Args:
            h (torch.Tensor): Input tensor

        Returns:
            tuple: Tuple containing logits, predicted probabilities, predicted labels
        """
        device = h.device
        h = h.view(-1, 28 * 28)

        h = self.net_general(h)
        N, C = h.shape

        # Multihead Input
        h = h.reshape(N, self.n_heads, C // self.n_heads)

        A = torch.empty(N, self.n_heads, 1).float().to(device)
        for head in range(self.n_heads):
            a, _ = self.attention_net[head](h[:, head, :])
            A[:, head, :] = a

        A = torch.transpose(A, 2, 0)  # K x heads x N

        # Temperature scaling
        for head_idx, head_temp in enumerate(self.temperature):
            A[:, head_idx, :] = A[:, head_idx, :] / head_temp

        A = F.softmax(A, dim=-1)  # softmax over N

        # Multihead Output
        M = torch.empty(1, self.size[1]).float().to(device)
        for head in range(self.n_heads):
            m = torch.mm(A[:, head, :], h[:, head, :])
            M[:, self.step * head: self.step * head + self.step] = m

        # Singlehead Classification
        logits = torch.squeeze(self.classifiers(M), dim=1)
        Y_prob = F.sigmoid(logits)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat


def initialize_weights(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_normal_(submodule.weight)
            submodule.bias.data.zero_()
        elif isinstance(submodule, nn.BatchNorm1d):
            nn.init.constant_(submodule.weight, 1)
            nn.init.constant_(submodule.bias, 0)
