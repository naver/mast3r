# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Sparse Global Alignement
# --------------------------------------------------------
from tqdm import tqdm
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import namedtuple
from functools import lru_cache
from scipy import sparse as sp
import copy

from mast3r.utils.misc import mkdir_for, hash_md5
from mast3r.cloud_opt.utils.losses import gamma_loss
from mast3r.cloud_opt.utils.schedules import linear_schedule, cosine_schedule
from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import inv, geotrf  # noqa
from dust3r.utils.device import to_cpu, to_numpy, todevice  # noqa
from dust3r.post_process import estimate_focal_knowing_depth  # noqa
from dust3r.optim_factory import adjust_learning_rate_by_lr  # noqa
from dust3r.cloud_opt.base_opt import clean_pointcloud
from dust3r.viz import SceneViz


class SparseGA():
    def __init__(self, img_paths, pairs_in, res_fine, anchors, canonical_paths=None):
        def fetch_img(im):
            def torgb(x): return (x[0].permute(1, 2, 0).numpy() * .5 + .5).clip(min=0., max=1.)
            for im1, im2 in pairs_in:
                if im1['instance'] == im:
                    return torgb(im1['img'])
                if im2['instance'] == im:
                    return torgb(im2['img'])
        self.canonical_paths = canonical_paths
        self.img_paths = img_paths
        self.imgs = [fetch_img(img) for img in img_paths]
        self.intrinsics = res_fine['intrinsics']
        self.cam2w = res_fine['cam2w']
        self.depthmaps = res_fine['depthmaps']
        self.pts3d = res_fine['pts3d']
        self.pts3d_colors = []
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        return self.cam2w

    def get_sparse_pts3d(self):
        return self.pts3d

    def get_dense_pts3d(self, clean_depth=True, subsample=8):
        assert self.canonical_paths, 'cache_path is required for dense 3d points'
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.cam2w, [
                                      d.ravel() for d in self.depthmaps], base_focals=base_focals, ret_depth=True)

        if clean_depth:
            confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        return self.pts3d_colors

    def get_depthmaps(self):
        return self.depthmaps

    def get_masks(self):
        return [slice(None, None) for _ in range(len(self.imgs))]

    def show(self, show_cams=True):
        pts3d, _, confs = self.get_dense_pts3d()
        show_reconstruction(self.imgs, self.intrinsics if show_cams else None, self.cam2w,
                            [p.clip(min=-50, max=50) for p in pts3d],
                            masks=[c > 1 for c in confs])


def convert_dust3r_pairs_naming(imgs, pairs_in):
    for pair_id in range(len(pairs_in)):
        for i in range(2):
            pairs_in[pair_id][i]['instance'] = imgs[pairs_in[pair_id][i]['idx']]
    return pairs_in


def sparse_global_alignment(imgs, pairs_in, cache_path, model, subsample=8, desc_conf='desc_conf',
                            device='cuda', dtype=torch.float32, shared_intrinsics=False, **kw):
    """ Sparse alignment with MASt3R
        imgs: list of image paths
        cache_path: path where to dump temporary files (str)

        lr1, niter1: learning rate and #iterations for coarse global alignment (3D matching)
        lr2, niter2: learning rate and #iterations for refinement (2D reproj error)

        lora_depth: smart dimensionality reduction with depthmaps
    """
    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)
    # forward pass
    pairs, cache_path = forward_mast3r(pairs_in, model,
                                       cache_path=cache_path, subsample=subsample,
                                       desc_conf=desc_conf, device=device)

    # extract canonical pointmaps
    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
        prepare_canonical_data(imgs, pairs, subsample, cache_path=cache_path, mode='avg-angle', device=device)

    # compute minimal spanning tree
    mst = compute_min_spanning_tree(pairwise_scores)

    # remove all edges not in the spanning tree?
    # min_spanning_tree = {(imgs[i],imgs[j]) for i,j in mst[1]}
    # tmp_pairs = {(a,b):v for (a,b),v in tmp_pairs.items() if {(a,b),(b,a)} & min_spanning_tree}

    # smartly combine all useful data
    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype)

    imgs, res_coarse, res_fine = sparse_scene_optimizer(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst,
        shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, **kw)

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths)


def sparse_scene_optimizer(imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d,
                           preds_21, canonical_paths, mst, cache_path,
                           lr1=0.2, niter1=500, loss1=gamma_loss(1.1),
                           lr2=0.02, niter2=500, loss2=gamma_loss(0.4),
                           lossd=gamma_loss(1.1),
                           opt_pp=True, opt_depth=True,
                           schedule=cosine_schedule, depth_mode='add', exp_depth=False,
                           lora_depth=False,  # dict(k=96, gamma=15, min_norm=.5),
                           shared_intrinsics=False,
                           init={}, device='cuda', dtype=torch.float32,
                           matching_conf_thr=5., loss_dust3r_w=0.01,
                           verbose=True, dbg=()):
    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f' >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}')

        K = init_values.get('intrinsics')
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]  # default principal_point would be (0.5, 0.5)

        depth = init_values.get('depthmap')
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        cam2w = init_values.get('cam2w')
        if cam2w is not None:
            rot = cam2w[:3, :3].detach()
            cam_center = cam2w[:3, 3].detach()
            quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
            trans_offset = med_depth * torch.cat((imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0]))
            trans[idx].data[:] = cam_center + rot @ trans_offset
            del rot
            assert False, 'inverse kinematic chain not yet implemented'

    # intrinsics parameters
    if shared_intrinsics:
        # Optimize a single set of intrinsics for all cameras. Use averages as init.
        confs = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
    max_focals = 10 * diags

    assert len(mst[1]) == len(pps) - 1

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal
        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1)
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K), 1, 4)), dim=1)

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append(global_scaling * core_depth_img)

        return K, (inv(cam2w), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics:
        print('init focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('init focals =', to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth)
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    core_depth = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres

    # Define which pairs are fine to use with matching
    def matching_check(x): return x.max() > matching_conf_thr
    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    def loss_dust3r(cam2w, pts3d, pix_loss):
        # In the case no correspondence could be established, fallback to DUSt3R GA regression loss formulation (sparsified)
        loss = 0.
        cf_sum = 0.
        for s in dust3r_slices:
            if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue
            # fallback to dust3r regression
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0. else 0.

    def loss_3d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        if any(v.get('freeze') for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.
            cf_sum = 1.

        return loss / cf_sum

    def loss_2d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        proj_matrix = K @ w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get('freeze', 0) >= 1:
                continue  # no need
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(pix1_filtered, reproj2d(proj_matrix[img1], pts3d_in_img1))
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = pps + log_focals + quats + trans + log_sizes + core_depth
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
                pts3d = make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    break

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            print(f'>> final loss = {loss}')
        return dict(intrinsics=K.detach(), cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps], pts3d=[p.detach() for p in pts3d])

    # at start, don't optimize 3d points
    for i, img in enumerate(imgs):
        trainable = not (init[img].get('freeze'))
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable)
        trans[i].requires_grad_(trainable)
        log_sizes[i].requires_grad_(trainable)
        core_depth[i].requires_grad_(False)

    res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

    res_fine = None
    if niter2:
        # now we can optimize 3d points
        for i, img in enumerate(imgs):
            if init[img].get('freeze', 0) >= 1:
                continue
            pps[i].requires_grad_(bool(opt_pp))
            log_focals[i].requires_grad_(True)
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)
    if shared_intrinsics:
        print('Final focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('Final focals =', to_numpy(K[:, 0, 0]))

    return imgs, res_coarse, res_fine


@lru_cache
def mask110(device, dtype):
    return torch.tensor((1, 1, 0), device=device, dtype=dtype)


def proj3d(inv_K, pixels, z):
    if pixels.shape[-1] == 2:
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    return z.unsqueeze(-1) * (pixels * inv_K.diag() + inv_K[:, 2] * mask110(z.device, z.dtype))


def make_pts3d(anchors, K, cam2w, depthmaps, base_focals=None, ret_depth=False):
    focals = K[:, 0, 0]
    invK = inv(K)
    all_pts3d = []
    depth_out = []

    for img, (pixels, idxs, offsets) in anchors.items():
        # from depthmaps to 3d points
        if base_focals is None:
            pass
        else:
            # compensate for focal
            # depth + depth * (offset - 1) * base_focal / focal
            # = depth * (1 + (offset - 1) * (base_focal / focal))
            offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])

        pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
        if ret_depth:
            depth_out.append(pts3d[..., 2])  # before camera rotation

        # rotate to world coordinate
        pts3d = geotrf(cam2w[img], pts3d)
        all_pts3d.append(pts3d)

    if ret_depth:
        return all_pts3d, depth_out
    return all_pts3d


def make_dense_pts3d(intrinsics, cam2w, depthmaps, canonical_paths, subsample, device='cuda'):
    base_focals = []
    anchors = {}
    confs = []
    for i, canon_path in enumerate(canonical_paths):
        (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
        confs.append(conf)
        base_focals.append(focal)
        H, W = conf.shape
        pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
        idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
        anchors[i] = (pixels, idxs[i], offsets[i])

    # densify sparse depthmaps
    pts3d, depthmaps_out = make_pts3d(anchors, intrinsics, cam2w, [
                                      d.ravel() for d in depthmaps], base_focals=base_focals, ret_depth=True)

    return pts3d, depthmaps_out, confs


@torch.no_grad()
def forward_mast3r(pairs, model, cache_path, desc_conf='desc_conf',
                   device='cuda', subsample=8, **matching_kw):
    res_paths = {}

    for img1, img2 in tqdm(pairs):
        idx1 = hash_md5(img1['instance'])
        idx2 = hash_md5(img2['instance'])

        path1 = cache_path + f'/forward/{idx1}/{idx2}.pth'
        path2 = cache_path + f'/forward/{idx2}/{idx1}.pth'
        path_corres = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{idx1}-{idx2}.pth'
        path_corres2 = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{idx2}-{idx1}.pth'

        if os.path.isfile(path_corres2) and not os.path.isfile(path_corres):
            score, (xy1, xy2, confs) = torch.load(path_corres2)
            torch.save((score, (xy2, xy1, confs)), path_corres)

        if not all(os.path.isfile(p) for p in (path1, path2, path_corres)):
            if model is None:
                continue
            res = symmetric_inference(model, img1, img2, device=device)
            X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
            C11, C21, C22, C12 = [r['conf'][0] for r in res]
            descs = [r['desc'][0] for r in res]
            qonfs = [r[desc_conf][0] for r in res]

            # save
            torch.save(to_cpu((X11, C11, X21, C21)), mkdir_for(path1))
            torch.save(to_cpu((X22, C22, X12, C12)), mkdir_for(path2))

            # perform reciprocal matching
            corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)

            conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
            matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))
            if cache_path is not None:
                torch.save((matching_score, corres), mkdir_for(path_corres))

        res_paths[img1['instance'], img2['instance']] = (path1, path2), path_corres

    del model
    torch.cuda.empty_cache()

    return res_paths, cache_path


def symmetric_inference(model, img1, img2, device):
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key='pred_desc'):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    # TODO add non symmetric / pixel_tol options
    for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()),
                         (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])

    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True)
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))

    return todevice(corres, device)


@torch.no_grad()
def prepare_canonical_data(imgs, tmp_pairs, subsample, order_imgs=False, min_conf_thr=0,
                           cache_path=None, device='cuda', **kw):
    canonical_views = {}
    pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
    canonical_paths = []
    preds_21 = {}

    for img in tqdm(imgs):
        if cache_path:
            cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}.pth')
            canonical_paths.append(cache)
        try:
            (canon, canon2, cconf), focal = torch.load(cache, map_location=device)
        except IOError:
            # cache does not exist yet, we create it!
            canon = focal = None

        # collect all pred1
        n_pairs = sum((img in pair) for pair in tmp_pairs)

        ptmaps11 = None
        pixels = {}
        n = 0
        for (img1, img2), ((path1, path2), path_corres) in tmp_pairs.items():
            score = None
            if img == img1:
                X, C, X2, C2 = torch.load(path1, map_location=device)
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img2] = xy1, confs
                if img not in preds_21:
                    preds_21[img] = {}
                # Subsample preds_21
                preds_21[img][img2] = X2[::subsample, ::subsample].reshape(-1, 3), C2[::subsample, ::subsample].ravel()

            if img == img2:
                X, C, X2, C2 = torch.load(path2, map_location=device)
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img1] = xy2, confs
                if img not in preds_21:
                    preds_21[img] = {}
                preds_21[img][img1] = X2[::subsample, ::subsample].reshape(-1, 3), C2[::subsample, ::subsample].ravel()

            if score is not None:
                i, j = imgs.index(img1), imgs.index(img2)
                # score = score[0]
                # score = np.log1p(score[2])
                score = score[2]
                pairwise_scores[i, j] = score
                pairwise_scores[j, i] = score

                if canon is not None:
                    continue
                if ptmaps11 is None:
                    H, W = C.shape
                    ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                    confs11 = torch.empty((n_pairs, H, W), device=device)

                ptmaps11[n] = X
                confs11[n] = C
                n += 1

        if canon is None:
            canon, canon2, cconf = canonical_view(ptmaps11, confs11, subsample, **kw)
            del ptmaps11
            del confs11

        # compute focals
        H, W = canon.shape[:2]
        pp = torch.tensor([W / 2, H / 2], device=device)
        if focal is None:
            focal = estimate_focal_knowing_depth(canon[None], pp, focal_mode='weiszfeld', min_focal=0.5, max_focal=3.5)
            if cache:
                torch.save(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))

        # extract depth offsets with correspondences
        core_depth = canon[subsample // 2::subsample, subsample // 2::subsample, 2]
        idxs, offsets = anchor_depth_offsets(canon2, pixels, subsample=subsample)

        canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

    return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21


def load_corres(path_corres, device, min_conf_thr):
    score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
    valid = confs > min_conf_thr if min_conf_thr else slice(None)
    # valid = (xy1 > 0).all(dim=1) & (xy2 > 0).all(dim=1) & (xy1 < 512).all(dim=1) & (xy2 < 512).all(dim=1)
    # print(f'keeping {valid.sum()} / {len(valid)} correspondences')
    return score, (xy1[valid], xy2[valid], confs[valid])


PairOfSlices = namedtuple(
    'ImgPair', 'img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum')


def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32):
    # aggregate all data properly
    set_imgs = set(imgs)

    principal_points = []
    shapes = []
    focals = []
    core_depth = []
    img_anchors = {}
    tmp_pixels = {}

    for idx1, img1 in enumerate(imgs):
        # load stuff
        pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views[img1]

        principal_points.append(pp)
        shapes.append(shape)
        focals.append(focal)
        core_depth.append(anchors)

        img_uv1 = []
        img_idxs = []
        img_offs = []
        cur_n = [0]

        for img2, (pixels, match_confs) in pixels_confs.items():
            if img2 not in set_imgs:
                continue
            assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
            img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
            img_idxs.append(idxs[img2])
            img_offs.append(offsets[img2])
            cur_n.append(cur_n[-1] + len(pixels))
            # store the position of 3d points
            tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
        img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))

    all_confs = []
    imgs_slices = []
    corres2d = {img: [] for img in range(len(imgs))}

    for img1, img2 in tmp_paths:
        try:
            pix1, confs1, slice1 = tmp_pixels[img1, img2]
            pix2, confs2, slice2 = tmp_pixels[img2, img1]
        except KeyError:
            continue
        img1 = imgs.index(img1)
        img2 = imgs.index(img2)
        confs = (confs1 * confs2).sqrt()

        # prepare for loss_3d
        all_confs.append(confs)
        anchor_idxs1 = canonical_views[imgs[img1]][5][imgs[img2]]
        anchor_idxs2 = canonical_views[imgs[img2]][5][imgs[img1]]
        imgs_slices.append(PairOfSlices(img1, slice1, pix1, anchor_idxs1,
                                        img2, slice2, pix2, anchor_idxs2,
                                        confs, float(confs.sum())))

        # prepare for loss_2d
        corres2d[img1].append((pix1, confs, img2, slice2))
        corres2d[img2].append((pix2, confs, img1, slice1))

    all_confs = torch.cat(all_confs)
    corres = (all_confs, float(all_confs.sum()), imgs_slices)

    def aggreg_matches(img1, list_matches):
        pix1, confs, img2, slice2 = zip(*list_matches)
        all_pix1 = torch.cat(pix1).to(dtype)
        all_confs = torch.cat(confs).to(dtype)
        return img1, all_pix1, all_confs, float(all_confs.sum()), [(j, sl2) for j, sl2 in zip(img2, slice2)]
    corres2d = [aggreg_matches(img, m) for img, m in corres2d.items()]

    imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)  # (W,H)
    principal_points = torch.stack(principal_points)
    focals = torch.cat(focals)

    # Subsample preds_21
    subsamp_preds_21 = {}
    for imk, imv in preds_21.items():
        subsamp_preds_21[imk] = {}
        for im2k, (pred, conf) in preds_21[imk].items():
            idxs = img_anchors[imgs.index(im2k)][1]
            subsamp_preds_21[imk][im2k] = (pred[idxs], conf[idxs])  # anchors subsample

    return imsizes, principal_points, focals, core_depth, img_anchors, corres, corres2d, subsamp_preds_21


def canonical_view(ptmaps11, confs11, subsample, mode='avg-angle'):
    assert len(ptmaps11) == len(confs11) > 0, 'not a single view1 for img={i}'

    # canonical pointmap is just a weighted average
    confs11 = confs11.unsqueeze(-1) - 0.999
    canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

    canon_depth = ptmaps11[..., 2].unsqueeze(1)
    S = slice(subsample // 2, None, subsample)
    center_depth = canon_depth[:, :, S, S]
    center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

    stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
    stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

    if mode == 'avg-reldepth':
        rel_depth = stacked_depth / center_depth
        stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
        canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

    elif mode == 'avg-angle':
        xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
        stacked_xy = F.pixel_unshuffle(xy, subsample)
        B, _, H, W = stacked_xy.shape
        stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
        stacked_radius.clip_(min=1e-8)

        stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
        avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

        # back to depth
        stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

        canon2 = F.pixel_shuffle((1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample).squeeze()
    else:
        raise ValueError(f'bad {mode=}')

    confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
    return canon, canon2, confs


def anchor_depth_offsets(canon_depth, pixels, subsample=8):
    device = canon_depth.device

    # create a 2D grid of anchor 3D points
    H1, W1 = canon_depth.shape
    yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
    H2, W2 = yx.shape[1:]
    cy, cx = yx.reshape(2, -1)
    core_depth = canon_depth[cy, cx]
    assert (core_depth > 0).all()

    # slave 3d points (attached to core 3d points)
    core_idxs = {}  # core_idxs[img2] = {corr_idx:core_idx}
    core_offs = {}  # core_offs[img2] = {corr_idx:3d_offset}

    for img2, (xy1, _confs) in pixels.items():
        px, py = xy1.long().T

        # find nearest anchor == block quantization
        core_idx = (py // subsample) * W2 + (px // subsample)
        core_idxs[img2] = core_idx.to(device)

        # compute relative depth offsets w.r.t. anchors
        ref_z = core_depth[core_idx]
        pts_z = canon_depth[py, px]
        offset = pts_z / ref_z
        core_offs[img2] = offset.detach().to(device)

    return core_idxs, core_offs


def spectral_clustering(graph, k=None, normalized_cuts=False):
    graph.fill_diagonal_(0)

    # graph laplacian
    degrees = graph.sum(dim=-1)
    laplacian = torch.diag(degrees) - graph
    if normalized_cuts:
        i_inv = torch.diag(degrees.sqrt().reciprocal())
        laplacian = i_inv @ laplacian @ i_inv

    # compute eigenvectors!
    eigval, eigvec = torch.linalg.eigh(laplacian)
    return eigval[:k], eigvec[:, :k]


def sim_func(p1, p2, gamma):
    diff = (p1 - p2).norm(dim=-1)
    avg_depth = (p1[:, :, 2] + p2[:, :, 2])
    rel_distance = diff / avg_depth
    sim = torch.exp(-gamma * rel_distance.square())
    return sim


def backproj(K, depthmap, subsample):
    H, W = depthmap.shape
    uv = np.mgrid[subsample // 2:subsample * W:subsample, subsample // 2:subsample * H:subsample].T.reshape(H, W, 2)
    xyz = depthmap.unsqueeze(-1) * geotrf(inv(K), todevice(uv, K.device), ncol=3)
    return xyz


def spectral_projection_depth(K, depthmap, subsample, k=64, cache_path='',
                              normalized_cuts=True, gamma=7, min_norm=5):
    try:
        if cache_path:
            cache_path = cache_path + f'_{k=}_norm={normalized_cuts}_{gamma=}.pth'
        lora_proj = torch.load(cache_path, map_location=K.device)

    except IOError:
        # reconstruct 3d points in camera coordinates
        xyz = backproj(K, depthmap, subsample)

        # compute all distances
        xyz = xyz.reshape(-1, 3)
        graph = sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
        _, lora_proj = spectral_clustering(graph, k, normalized_cuts=normalized_cuts)

        if cache_path:
            torch.save(lora_proj.cpu(), mkdir_for(cache_path))

    lora_proj, coeffs = lora_encode_normed(lora_proj, depthmap.ravel(), min_norm=min_norm)

    # depthmap ~= lora_proj @ coeffs
    return coeffs, lora_proj


def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
    # encode the pointmap
    coeffs = torch.linalg.pinv(lora_proj) @ x

    # rectify the norm of basis vector to be ~ equal
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if global_norm:
        lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
    elif min_norm:
        lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
    # can have rounding errors here!
    coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()

    return lora_proj.detach(), coeffs.detach()


@torch.no_grad()
def spectral_projection_of_depthmaps(imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw):
    # recover 3d points
    core_depth = []
    lora_proj = []

    for i, img in enumerate(tqdm(imgs)):
        cache = os.path.join(cache_path, 'lora_depth', hash_md5(img)) if cache_path else None
        depth, proj = spectral_projection_depth(intrinsics[i], depthmaps[i], subsample,
                                                cache_path=cache, **kw)
        core_depth.append(depth)
        lora_proj.append(proj)

    return core_depth, lora_proj


def reproj2d(Trf, pts3d):
    res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
    clipped_z = res[:, 2:3].clip(min=1e-3)  # make sure we don't have nans!
    uv = res[:, 0:2] / clipped_z
    return uv.clip(min=-1000, max=2000)


def bfs(tree, start_node):
    order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
    ranks = np.arange(len(order))
    ranks[order] = ranks.copy()
    return ranks, predecessors


def compute_min_spanning_tree(pws):
    sparse_graph = sp.dok_array(pws.shape)
    for i, j in pws.nonzero().cpu().tolist():
        sparse_graph[i, j] = -float(pws[i, j])
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph)

    # now reorder the oriented edges, starting from the central point
    ranks1, _ = bfs(msp, 0)
    ranks2, _ = bfs(msp, ranks1.argmax())
    ranks1, _ = bfs(msp, ranks2.argmax())
    # this is the point farther from any leaf
    root = np.minimum(ranks1, ranks2).argmax()

    # find the ordered list of edges that describe the tree
    order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
    order = order[1:]  # root not do not have a predecessor
    edges = [(predecessors[i], i) for i in order]

    return root, edges


def show_reconstruction(shapes_or_imgs, K, cam2w, pts3d, gt_cam2w=None, gt_K=None, cam_size=None, masks=None, **kw):
    viz = SceneViz()

    cc = cam2w[:, :3, 3]
    cs = cam_size or float(torch.cdist(cc, cc).fill_diagonal_(np.inf).min(dim=0).values.median())
    colors = 64 + np.random.randint(255 - 64, size=(len(cam2w), 3))

    if isinstance(shapes_or_imgs, np.ndarray) and shapes_or_imgs.ndim == 2:
        cam_kws = dict(imsizes=shapes_or_imgs[:, ::-1], cam_size=cs)
    else:
        imgs = shapes_or_imgs
        cam_kws = dict(images=imgs, cam_size=cs)
    if K is not None:
        viz.add_cameras(to_numpy(cam2w), to_numpy(K), colors=colors, **cam_kws)

    if gt_cam2w is not None:
        if gt_K is None:
            gt_K = K
        viz.add_cameras(to_numpy(gt_cam2w), to_numpy(gt_K), colors=colors, marker='o', **cam_kws)

    if pts3d is not None:
        for i, p in enumerate(pts3d):
            if not len(p):
                continue
            if masks is None:
                viz.add_pointcloud(to_numpy(p), color=tuple(colors[i].tolist()))
            else:
                viz.add_pointcloud(to_numpy(p), mask=masks[i], color=imgs[i])
    viz.show(**kw)
