# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Collate extensions
# --------------------------------------------------------

import torch
import collections
from torch.utils.data._utils.collate import default_collate_fn_map, default_collate_err_msg_format
from typing import Callable, Dict, Optional, Tuple, Type, Union, List


def cat_collate_tensor_fn(batch, *, collate_fn_map):
    return torch.cat(batch, dim=0)


def cat_collate_list_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return [item for bb in batch for item in bb]  # concatenate all lists


cat_collate_fn_map = default_collate_fn_map.copy()
cat_collate_fn_map[torch.Tensor] = cat_collate_tensor_fn
cat_collate_fn_map[List] = cat_collate_list_fn
cat_collate_fn_map[type(None)] = lambda _, **kw: None  # When some Nones, simply return a single None


def cat_collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""Custom collate function that concatenates stuff instead of stacking them, and handles NoneTypes """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: cat_collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: cat_collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(cat_collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            # Backwards compatibility.
            return [cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]
        else:
            try:
                return elem_type([cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
