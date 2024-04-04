import numpy as np
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, variance_scaling=False):
    if variance_scaling:
        std = torch.sqrt(std / torch.tensor(layer.weight.shape[1]))
        distribution_stddev = torch.tensor(0.87962566103423978)
        std /= distribution_stddev
    torch.nn.init.trunc_normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def uniform_scaling_layer_init(layer, bias_const=0.0, scale=0.333):
    max_val = (
        torch.sqrt(torch.tensor(3.0) / torch.tensor(layer.weight.shape[1]))
        * scale
    )
    torch.nn.init.uniform_(layer.weight, a=-max_val, b=max_val)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
