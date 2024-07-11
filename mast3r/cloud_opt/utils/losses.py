# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# losses for sparse ga
# --------------------------------------------------------
import torch
import numpy as np


def l05_loss(x, y):
    return torch.linalg.norm(x - y, dim=-1).sqrt()


def l1_loss(x, y):
    return torch.linalg.norm(x - y, dim=-1)


def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
    if offset is None:
        if gamma == 1:
            return l1_loss
        # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
        offset = (1 / gamma)**(1 / (gamma - 1))

    def loss_func(x, y):
        return (mul * l1_loss(x, y).clip(max=clip) + offset) ** gamma - offset ** gamma
    return loss_func


def meta_gamma_loss():
    return lambda alpha: gamma_loss(alpha)
