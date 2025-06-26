# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape, is_symmetrized  # noqa

inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


def load_dune_mast3r_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    net = AsymmetricMASt3RWithDUNEBackbone(ckpt['dune_backbone_name'], ckpt['mast3r_model_str'], landscape_only=False)
    missing_keys, unexpected_keys = net.load_state_dict(ckpt['model'], strict=False)
    assert all(k.startswith('dune_backbone') or k.startswith('imagenet') for k in missing_keys), missing_keys
    assert len(unexpected_keys) == 0, unexpected_keys
    return net.to(device)


class AsymmetricMASt3RWithDUNEBackbone(torch.nn.Module):

    def __init__(self, dune_backbone_name, mast3r_model_str, landscape_only=True):  # "dune_vitbase_14_448_paper_encoder"
        super().__init__()
        self.dune_backbone = torch.hub.load("naver/dune", dune_backbone_name)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1))
        self.norm = torch.nn.LayerNorm(self.dune_backbone.num_features)
        assert 'landscape_only' not in mast3r_model_str
        self.landscape_only = landscape_only
        if not self.landscape_only:
            mast3r_model_str = mast3r_model_str[:-1] + f', landscape_only=False)'
        patch_size = self.dune_backbone.patch_size
        self.patch_size = patch_size
        self.square_ok = True
        if patch_size != 16:
            assert not "patch_size" in mast3r_model_str, mast3r_model_str
            assert mast3r_model_str.endswith(')'), mast3r_model_str
            mast3r_model_str = mast3r_model_str[:-1] + f', patch_size={patch_size})'
        self.mast3r = eval(mast3r_model_str)
        if patch_size != 16:  # might depend on mast3r stuff
            # we need to hack the head a bit ...
            for head in [self.mast3r.downstream_head1, self.mast3r.downstream_head2]:
                head.dpt.patch_size = (patch_size, patch_size)
                head.dpt.P_H = max(1, patch_size // head.dpt.stride_level)
                head.dpt.P_W = max(1, patch_size // head.dpt.stride_level)
                head.dpt.head[1].scale_factor *= 14 / 16
        # freeze dune
        self.dune_backbone.eval()
        for p in self.dune_backbone.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_dune_mast3r_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def train(self, mode=True):
        self = super().train(mode=mode)
        self.dune_backbone.eval()
        return self

    @torch.no_grad()
    def _encode_image(self, image, true_shape):
        # a bit tricky due to mixing aspect ratios at training ...
        B, C, H, W = image.shape
        patch_size = self.dune_backbone.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        image = image * 0.5 + 0.5  # undo normalization from mast3r
        image = (image - self.imagenet_mean) / self.imagenet_std  # and do the one from imagenet
        W //= patch_size[0]
        H //= patch_size[1]
        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape
        if self.landscape_only:
            assert W >= H, f'image should be in landscape mode, but got {W=} {H=}'
            assert H % patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({patch_size[0]})."
            assert W % patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({patch_size[1]})."
            assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"
            if torch.any(is_portrait):
                x1 = self.dune_backbone.prepare_tokens_with_masks(image[is_landscape])
                x2 = self.dune_backbone.prepare_tokens_with_masks(image[is_portrait].permute(0, 1, 3, 2))
                x = torch.empty(B, x1.size(1), x1.size(2), device=image.device, dtype=x1.dtype)
                x[is_landscape] = x1
                x[is_portrait] = x2
                pos = image.new_zeros(
                    (B, x.shape[1] - self.dune_backbone.num_register_tokens - 1, 2), dtype=torch.int64)
                pos[is_landscape] = self.mast3r.patch_embed.position_getter(1, H, W, pos.device)
                pos[is_portrait] = self.mast3r.patch_embed.position_getter(1, W, H, pos.device)
            else:
                x = self.dune_backbone.prepare_tokens_with_masks(image)
                pos = self.mast3r.patch_embed.position_getter(B, H, W, image.device)
        else:
            assert torch.all(is_landscape) or torch.all(is_portrait)
            x = self.dune_backbone.prepare_tokens_with_masks(image)
            pos = self.mast3r.patch_embed.position_getter(B, H, W, image.device)

        for blk in self.dune_backbone.blocks[0]:
            x = blk(x)
        features = x[:, self.dune_backbone.num_register_tokens + 1:, :]
        return features.detach(), pos

    @torch.no_grad()
    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos = self._encode_image(torch.cat((img1, img2), dim=0),
                                          torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    @torch.no_grad()
    def encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encode_symmetrized(view1, view2)
        feat1 = self.norm(feat1)
        feat2 = self.norm(feat2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self.mast3r._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self.mast3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self.mast3r._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
