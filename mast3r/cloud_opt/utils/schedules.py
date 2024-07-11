# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# lr schedules for sparse ga
# --------------------------------------------------------
import numpy as np


def linear_schedule(alpha, lr_base, lr_end=0):
    lr = (1 - alpha) * lr_base + alpha * lr_end
    return lr


def cosine_schedule(alpha, lr_base, lr_end=0):
    lr = lr_end + (lr_base - lr_end) * (1 + np.cos(alpha * np.pi)) / 2
    return lr
