# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Whitener and RetrievalModel
# --------------------------------------------------------
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

default_device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')


# from https://github.com/gtolias/how/blob/4d73c88e0ffb55506e2ce6249e2a015ef6ccf79f/how/utils/whitening.py#L20
def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2 * N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    eigval = np.clip(eigval, a_min=1e-14, a_max=None)
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5 * s))), eigvec.T)

    return m, P.T


class Dust3rInputFromImageList(torch.utils.data.Dataset):
    def __init__(self, image_list, imsize=512):
        super().__init__()
        self.image_list = image_list
        assert imsize == 512
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return load_images([self.image_list[index]], size=self.imsize, verbose=False)[0]


class Whitener(nn.Module):
    def __init__(self, dim, l2norm=None):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros((1, dim)).double())
        self.p = torch.nn.Parameter(torch.eye(dim, dim).double())
        self.l2norm = l2norm  # if not None, apply l2 norm along a given dimension

    def forward(self, x):
        with torch.autocast(self.m.device.type, enabled=False):
            shape = x.size()
            input_type = x.dtype
            x_reshaped = x.view(-1, shape[-1]).to(dtype=self.m.dtype)
            # Center the input data
            x_centered = x_reshaped - self.m
            # Apply PCA transformation
            pca_output = torch.matmul(x_centered, self.p)
            # reshape back
            pca_output_shape = shape  # list(shape[:-1]) + [shape[-1]]
            pca_output = pca_output.view(pca_output_shape)
            if self.l2norm is not None:
                return torch.nn.functional.normalize(pca_output, dim=self.l2norm).to(dtype=input_type)
            return pca_output.to(dtype=input_type)


def weighted_spoc(feat, attn):
    """
    feat: BxNxC
    attn: BxN
    output: BxC L2-normalization weighted-sum-pooling of features
    """
    return torch.nn.functional.normalize((feat * attn[:, :, None]).sum(dim=1), dim=1)


def how_select_local(feat, attn, nfeat):
    """
    feat: BxNxC
    attn: BxN
    nfeat: nfeat to keep
    """
    # get nfeat
    if nfeat < 0:
        assert nfeat >= -1.0
        nfeat = int(-nfeat * feat.size(1))
    else:
        nfeat = int(nfeat)
    # asort
    topk_attn, topk_indices = torch.topk(attn, min(nfeat, attn.size(1)), dim=1)
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, feat.size(2))
    topk_features = torch.gather(feat, 1, topk_indices_expanded)
    return topk_features, topk_attn, topk_indices


class RetrievalModel(nn.Module):
    def __init__(self, backbone, freeze_backbone=1, prewhiten=None, hdims=[1024], residual=False, postwhiten=None,
                 featweights='l2norm', nfeat=300, pretrained_retrieval=None):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_dim = backbone.enc_embed_dim
        self.prewhiten = nn.Identity() if prewhiten is None else Whitener(self.backbone_dim)
        self.prewhiten_freq = prewhiten
        if prewhiten is not None and prewhiten != -1:
            for p in self.prewhiten.parameters():
                p.requires_grad = False
        self.residual = residual
        self.projector = self.build_projector(hdims, residual)
        self.dim = hdims[-1] if len(hdims) > 0 else self.backbone_dim
        self.postwhiten_freq = postwhiten
        self.postwhiten = nn.Identity() if postwhiten is None else Whitener(self.dim)
        if postwhiten is not None and postwhiten != -1:
            assert len(hdims) > 0
            for p in self.postwhiten.parameters():
                p.requires_grad = False
        self.featweights = featweights
        if featweights == 'l2norm':
            self.attention = lambda x: x.norm(dim=-1)
        else:
            raise NotImplementedError(featweights)
        self.nfeat = nfeat
        self.pretrained_retrieval = pretrained_retrieval
        if self.pretrained_retrieval is not None:
            ckpt = torch.load(pretrained_retrieval, 'cpu')
            msg = self.load_state_dict(ckpt['model'], strict=False)
            assert len(msg.unexpected_keys) == 0 and all(k.startswith('backbone')
                                                         or k.startswith('postwhiten') for k in msg.missing_keys)

    def build_projector(self, hdims, residual):
        if self.residual:
            assert hdims[-1] == self.backbone_dim
        d = self.backbone_dim
        if len(hdims) == 0:
            return nn.Identity()
        layers = []
        for i in range(len(hdims) - 1):
            layers.append(nn.Linear(d, hdims[i]))
            d = hdims[i]
            layers.append(nn.LayerNorm(d))
            layers.append(nn.GELU())
        layers.append(nn.Linear(d, hdims[-1]))
        return nn.Sequential(*layers)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        ss = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.freeze_backbone:
            ss = {k: v for k, v in ss.items() if not k.startswith('backbone')}
        return ss

    def reinitialize_whitening(self, epoch, train_dataset, nimgs=5000, log_writer=None, max_nfeat_per_image=None, seed=0, device=default_device):
        do_prewhiten = self.prewhiten_freq is not None and self.pretrained_retrieval is None and \
            (epoch == 0 or (self.prewhiten_freq > 0 and epoch % self.prewhiten_freq == 0))
        do_postwhiten = self.postwhiten_freq is not None and ((epoch == 0 and self.postwhiten_freq in [0, -1])
                                                              or (self.postwhiten_freq > 0 and
                                                                  epoch % self.postwhiten_freq == 0 and epoch > 0))
        if do_prewhiten or do_postwhiten:
            self.eval()
            imdataset = train_dataset.imlist_dataset_n_images(nimgs, seed)
            loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        if do_prewhiten:
            print('Re-initialization of pre-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                       true_shape=d['true_shape'][0, ...])[0]
                    feat = feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < feat.size(0):
                        l2norms = torch.linalg.vector_norm(feat, dim=1)
                        feat = feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(feat.cpu())
            features = torch.cat(features, dim=0)
            features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.prewhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            prewhiten_time = time.time() - t
            print(f'Done in {prewhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/prewhiten', prewhiten_time, epoch)
        if do_postwhiten:
            print(f'Re-initialization of post-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    backbone_feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                                true_shape=d['true_shape'][0, ...])[0]
                    backbone_feat_prewhitened = self.prewhiten(backbone_feat)
                    proj_feat = self.projector(backbone_feat_prewhitened) + \
                        (0.0 if not self.residual else backbone_feat_prewhitened)
                    proj_feat = proj_feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < proj_feat.size(0):
                        l2norms = torch.linalg.vector_norm(proj_feat, dim=1)
                        proj_feat = proj_feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(proj_feat.cpu())
                features = torch.cat(features, dim=0)
                features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.postwhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            postwhiten_time = time.time() - t
            print(f'Done in {postwhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/postwhiten', postwhiten_time, epoch)

    def extract_features_and_attention(self, x):
        backbone_feat = self.backbone._encode_image(x['img'], true_shape=x['true_shape'])[0]
        backbone_feat_prewhitened = self.prewhiten(backbone_feat)
        proj_feat = self.projector(backbone_feat_prewhitened) + \
            (0.0 if not self.residual else backbone_feat_prewhitened)
        attention = self.attention(proj_feat)
        proj_feat_whitened = self.postwhiten(proj_feat)
        return proj_feat_whitened, attention

    def forward_local(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return how_select_local(feat, attn, self.nfeat)

    def forward_global(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return weighted_spoc(feat, attn)

    def forward(self, x):
        return self.forward_global(x)


def identity(x):  # to avoid Can't pickle local object 'extract_local_features.<locals>.<lambda>'
    return x


@torch.no_grad()
def extract_local_features(model, images, imsize, seed=0, tocpu=False, max_nfeat_per_image=None,
                           max_nfeat_per_image2=None, device=default_device):
    model.eval()
    imdataset = Dust3rInputFromImageList(images, imsize=imsize) if isinstance(images, list) else images
    loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, collate_fn=identity)
    with torch.no_grad():
        features = []
        imids = []
        for i, d in enumerate(tqdm(loader)):
            dd = d[0]
            dd['img'] = dd['img'].to(device, non_blocking=True)
            feat, _, _ = model.forward_local(dd)
            feat = feat.flatten(0, 1)
            if max_nfeat_per_image is not None and feat.size(0) > max_nfeat_per_image:
                feat = feat[torch.randperm(feat.size(0))[:max_nfeat_per_image], :]
            if max_nfeat_per_image2 is not None and feat.size(0) > max_nfeat_per_image2:
                feat = feat[:max_nfeat_per_image2, :]
            features.append(feat)
            if tocpu:
                features[-1] = features[-1].cpu()
            imids.append(i * torch.ones_like(features[-1][:, 0]).to(dtype=torch.int64))
    features = torch.cat(features, dim=0)
    imids = torch.cat(imids, dim=0)
    return features, imids
