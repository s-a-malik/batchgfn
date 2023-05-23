from typing import Optional, Literal

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    A simple MLP encoder. (inspired by https://github.com/ae-foster/dad/blob/4b1008174e1531d1f14601d83cef481c0f586f36/death_process.py#L37)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoding_dim: int,
        n_hidden_layers: int,
        activation_fn: Literal["relu", "tanh"],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        assert activation_fn is not None, "activation_fn must be provided"
        activation = get_activation(activation_fn)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation_layer = activation()
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


def get_activation(act_string):
    if act_string == "relu":
        return nn.ReLU
    elif act_string == "tanh":
        return nn.Tanh
    elif act_string == "sigmoid":
        return nn.Sigmoid
    elif act_string == "gelu":
        return nn.GELU
    else:
        raise ValueError("Unknown activation function: {}".format(act_string))
