# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from .base.mast3r_base_stereo_view_dataset import MASt3RBaseStereoViewDataset

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.arkitscenes import ARKitScenes as DUSt3R_ARKitScenes  # noqa
from dust3r.datasets.blendedmvs import BlendedMVS as DUSt3R_BlendedMVS  # noqa
from dust3r.datasets.co3d import Co3d as DUSt3R_Co3d  # noqa
from dust3r.datasets.megadepth import MegaDepth as DUSt3R_MegaDepth  # noqa
from dust3r.datasets.scannetpp import ScanNetpp as DUSt3R_ScanNetpp  # noqa
from dust3r.datasets.staticthings3d import StaticThings3D as DUSt3R_StaticThings3D  # noqa
from dust3r.datasets.waymo import Waymo as DUSt3R_Waymo  # noqa
from dust3r.datasets.wildrgbd import WildRGBD as DUSt3R_WildRGBD  # noqa


class ARKitScenes(DUSt3R_ARKitScenes, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class BlendedMVS(DUSt3R_BlendedMVS, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, split=None, **kwargs):
        super().__init__(*args, ROOT=ROOT, split=split, **kwargs)
        self.is_metric_scale = False


class Co3d(DUSt3R_Co3d, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class MegaDepth(DUSt3R_MegaDepth, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ScanNetpp(DUSt3R_ScanNetpp, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class StaticThings3D(DUSt3R_StaticThings3D, MASt3RBaseStereoViewDataset):
    def __init__(self, ROOT, *args, mask_bg='rand', **kwargs):
        super().__init__(ROOT, *args, mask_bg=mask_bg, **kwargs)
        self.is_metric_scale = False


class Waymo(DUSt3R_Waymo, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class WildRGBD(DUSt3R_WildRGBD, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True
