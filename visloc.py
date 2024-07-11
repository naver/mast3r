#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# visloc script with support for coarse to fine
# --------------------------------------------------------
import os
import numpy as np
import random
import torch
import torchvision.transforms as tvf
import argparse
from tqdm import tqdm
from PIL import Image
import math

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from mast3r.utils.misc import mkdir_for
from mast3r.datasets.utils.cropping import crop_to_homography

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, loss_of_one_batch
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r_visloc.datasets import *
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats, export_results
from dust3r_visloc.datasets.utils import get_HW_resolution, rescale_points3d


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="visloc dataset to eval")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])

    parser.add_argument("--confidence_threshold", type=float, default=1.001,
                        help="confidence values higher than threshold are invalid")
    parser.add_argument('--pixel_tol', default=5, type=int)

    parser.add_argument("--coarse_to_fine", action='store_true', default=False,
                        help="do the matching from coarse to fine")
    parser.add_argument("--max_image_size", type=int, default=None,
                        help="max image size for the fine resolution")
    parser.add_argument("--c2f_crop_with_homography", action='store_true', default=False,
                        help="when using coarse to fine, crop with homographies to keep cx, cy centered")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--pnp_mode", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error", type=float, default=5.0, help="pnp reprojection error")
    parser_reproj.add_argument("--reprojection_error_diag_ratio", type=float, default=None,
                               help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--max_batch_size", type=int, default=48,
                        help="max batch size for inference on crops when using coarse to fine")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches", type=int, default=0, help="debug matches")

    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument("--output_label", type=str, default='', help="prefix for results files")
    return parser


@torch.no_grad()
def coarse_matching(query_view, map_view, model, device, pixel_tol, fast_nn_params):
    # prepare batch
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], [], []

    if pixel_tol == 0:
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params)
        HM, WM = map_view['rgb_rescaled'].shape[1:]
        HQ, WQ = query_view['rgb_rescaled'].shape[1:]
        # ignore small border around the edge
        valid_matches_map = (matches_im_map[:, 0] >= 3) & (matches_im_map[:, 0] < WM - 3) & (
            matches_im_map[:, 1] >= 3) & (matches_im_map[:, 1] < HM - 3)
        valid_matches_query = (matches_im_query[:, 0] >= 3) & (matches_im_query[:, 0] < WQ - 3) & (
            matches_im_query[:, 1] >= 3) & (matches_im_query[:, 1] < HQ - 3)
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        valid_pts3d = []
        matches_confs = []
    else:
        yM, xM = torch.where(map_view['valid_rescaled'])
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
        valid_pts3d = map_view['pts3d_rescaled'].cpu().numpy()[matches_im_map[:, 1], matches_im_map[:, 0]]
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
        )
    # from cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # rescale coordinates
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    # from colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def fine_matching(query_views, map_views, model, device, max_batch_size, pixel_tol, fast_nn_params):
    assert pixel_tol > 0
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    valid_pts3d, matches_im_map, matches_im_query, matches_confs = [], [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        valid_ppi = map_views['valid'][ppi]
        pts3d_ppi = map_views['pts3d'][ppi].cpu().numpy()
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]

        y_ppi, x_ppi = torch.where(valid_ppi)
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(pp2, pp1, (x_ppi, y_ppi),
                                                                       pixel_tol=pixel_tol, **fast_nn_params)

        valid_pts3d_ppi = pts3d_ppi[matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]]
        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi].cpu().numpy(), matches_im_map_ppi.copy(), norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi].cpu().numpy(), matches_im_query_ppi.copy(), norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        valid_pts3d.append(valid_pts3d_ppi)
        matches_confs.append(matches_confs_ppi)

    if len(valid_pts3d) == 0:
        return [], [], [], []

    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    valid_pts3d = np.concatenate(valid_pts3d, axis=0)
    matches_confs = np.concatenate(matches_confs, axis=0)
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


def crop(img, mask, pts3d, crop, intrinsics=None):
    out_cropped_img = img.clone()
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        K_old = intrinsics
        imsize, K_new, R, H = crop_to_homography(K_old, crop)
        # apply homography to image
        H /= H[2, 2]
        homo8 = H.ravel().tolist()[:8]
        # From float tensor to uint8 PIL Image
        pilim = Image.fromarray((255 * (img + 1.) / 2).to(torch.uint8).numpy())
        pilout_cropped_img = pilim.transform(imsize, Image.Transform.PERSPECTIVE,
                                             homo8, resample=Image.Resampling.BICUBIC)

        # From uint8 PIL Image to float tensor
        out_cropped_img = 2. * torch.tensor(np.array(pilout_cropped_img)).to(img) / 255. - 1.
        if out_cropped_mask is not None:
            pilmask = Image.fromarray((255 * out_cropped_mask).to(torch.uint8).numpy())
            pilout_cropped_mask = pilmask.transform(
                imsize, Image.Transform.PERSPECTIVE, homo8, resample=Image.Resampling.NEAREST)
            out_cropped_mask = torch.from_numpy(np.array(pilout_cropped_mask) > 0).to(out_cropped_mask.dtype)
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d.numpy()
            out_cropped_X = np.array(Image.fromarray(out_cropped_pts3d[:, :, 0]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Y = np.array(Image.fromarray(out_cropped_pts3d[:, :, 1]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Z = np.array(Image.fromarray(out_cropped_pts3d[:, :, 2]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))

            out_cropped_pts3d = torch.from_numpy(np.stack([out_cropped_X, out_cropped_Y, out_cropped_Z], axis=-1))

        to_orig = torch.tensor(H, device=img.device)
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig


def resize_image_to_max(max_image_size, rgb, K):
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = (W >= H)
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = tvf.Compose([ImgNorm, tvf.Resize(size=[HMax, WMax])])
        rgb_tensor = resize_op(rgb).permute(1, 2, 0)
        to_orig_max = np.array([[W / WMax, 0, 0],
                                [0, H / HMax, 0],
                                [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0],
                                  [0, HMax / H, 0],
                                  [0, 0, 1]])

        # Generate new camera parameters
        new_K = opencv_to_colmap_intrinsics(K)
        new_K[0, :] *= WMax / W
        new_K[1, :] *= HMax / H
        new_K = colmap_to_opencv_intrinsics(new_K)
    else:
        rgb_tensor = ImgNorm(rgb).permute(1, 2, 0)
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        new_K = K
    return rgb_tensor, new_K, to_orig_max, to_resize_max, (HMax, WMax)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    conf_thr = args.confidence_threshold
    device = args.device
    pnp_mode = args.pnp_mode
    assert args.pixel_tol > 0
    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points
    viz_matches = args.viz_matches

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)
    dataset = eval(args.dataset)
    dataset.set_resolution(model)

    query_names = []
    poses_pred = []
    pose_errors = []
    angular_errors = []
    params_str = f'tol_{args.pixel_tol}' + ("_c2f" if args.coarse_to_fine else '')
    if args.max_image_size is not None:
        params_str = params_str + f'_{args.max_image_size}'
    if args.coarse_to_fine and args.c2f_crop_with_homography:
        params_str = params_str + '_with_homography'
    for idx in tqdm(range(len(dataset))):
        views = dataset[(idx)]  # 0 is the query
        query_view = views[0]
        map_views = views[1:]
        query_names.append(query_view['image_name'])

        query_pts2d = []
        query_pts3d = []
        maxdim = max(model.patch_embed.img_size)
        query_rgb_tensor, query_K, query_to_orig_max, query_to_resize_max, (HQ, WQ) = resize_image_to_max(
            args.max_image_size, query_view['rgb'], query_view['intrinsics'])

        # pairs of crops have the same resolution
        query_resolution = get_HW_resolution(HQ, WQ, maxdim=maxdim, patchsize=model.patch_embed.patch_size)
        for map_view in map_views:
            if args.output_dir is not None:
                cache_file = os.path.join(args.output_dir, 'matches', params_str,
                                          query_view['image_name'], map_view['image_name'] + '.npz')
            else:
                cache_file = None

            if cache_file is not None and os.path.isfile(cache_file):
                matches = np.load(cache_file)
                valid_pts3d = matches['valid_pts3d']
                matches_im_query = matches['matches_im_query']
                matches_im_map = matches['matches_im_map']
                matches_conf = matches['matches_conf']
            else:
                # coarse matching
                if args.coarse_to_fine and (maxdim < max(WQ, HQ)):
                    # use all points
                    _, coarse_matches_im0, coarse_matches_im1, _ = coarse_matching(query_view, map_view, model, device,
                                                                                   0, fast_nn_params)

                    # visualize a few matches
                    if viz_matches > 0:
                        num_matches = coarse_matches_im1.shape[0]
                        print(f'found {num_matches} matches')

                        viz_imgs = [np.array(query_view['rgb']), np.array(map_view['rgb'])]
                        from matplotlib import pyplot as pl
                        n_viz = viz_matches
                        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                        viz_matches_im_query = coarse_matches_im0[match_idx_to_viz]
                        viz_matches_im_map = coarse_matches_im1[match_idx_to_viz]

                        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                                      'constant', constant_values=0)
                        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                                      'constant', constant_values=0)
                        img = np.concatenate((img0, img1), axis=1)
                        pl.figure()
                        pl.imshow(img)
                        cmap = pl.get_cmap('jet')
                        for i in range(n_viz):
                            (x0, y0), (x1, y1) = viz_matches_im_query[i].T, viz_matches_im_map[i].T
                            pl.plot([x0, x1 + W0], [y0, y1], '-+',
                                    color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                        pl.show(block=True)

                    valid_all = map_view['valid']
                    pts3d = map_view['pts3d']

                    WM_full, HM_full = map_view['rgb'].size
                    map_rgb_tensor, map_K, map_to_orig_max, map_to_resize_max, (HM, WM) = resize_image_to_max(
                        args.max_image_size, map_view['rgb'], map_view['intrinsics'])
                    if WM_full != WM or HM_full != HM:
                        y_full, x_full = torch.where(valid_all)
                        pos2d_cv2 = torch.stack([x_full, y_full], dim=-1).cpu().numpy().astype(np.float64)
                        sparse_pts3d = pts3d[y_full, x_full].cpu().numpy()
                        _, _, pts3d_max, valid_max = rescale_points3d(
                            pos2d_cv2, sparse_pts3d, map_to_resize_max, HM, WM)
                        pts3d = torch.from_numpy(pts3d_max)
                        valid_all = torch.from_numpy(valid_max)

                    coarse_matches_im0 = geotrf(query_to_resize_max, coarse_matches_im0, norm=True)
                    coarse_matches_im1 = geotrf(map_to_resize_max, coarse_matches_im1, norm=True)

                    crops1, crops2 = [], []
                    crops_v1, crops_p1 = [], []
                    to_orig1, to_orig2 = [], []
                    map_resolution = get_HW_resolution(HM, WM, maxdim=maxdim, patchsize=model.patch_embed.patch_size)

                    for crop_q, crop_b, pair_tag in select_pairs_of_crops(map_rgb_tensor,
                                                                          query_rgb_tensor,
                                                                          coarse_matches_im1,
                                                                          coarse_matches_im0,
                                                                          maxdim=maxdim,
                                                                          overlap=.5,
                                                                          forced_resolution=[map_resolution,
                                                                                             query_resolution]):
                        # Per crop processing
                        if not args.c2f_crop_with_homography:
                            map_K = None
                            query_K = None

                        c1, v1, p1, trf1 = crop(map_rgb_tensor, valid_all, pts3d, crop_q, map_K)
                        c2, _, _, trf2 = crop(query_rgb_tensor, None, None, crop_b, query_K)
                        crops1.append(c1)
                        crops2.append(c2)
                        crops_v1.append(v1)
                        crops_p1.append(p1)
                        to_orig1.append(trf1)
                        to_orig2.append(trf2)

                    if len(crops1) == 0 or len(crops2) == 0:
                        valid_pts3d, matches_im_query, matches_im_map, matches_conf = [], [], [], []
                    else:
                        crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
                        if len(crops1.shape) == 3:
                            crops1, crops2 = crops1[None], crops2[None]
                        crops_v1 = torch.stack(crops_v1)
                        crops_p1 = torch.stack(crops_p1)
                        to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
                        map_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                             instance=['1' for _ in range(crops1.shape[0])],
                                             valid=crops_v1, pts3d=crops_p1,
                                             to_orig=to_orig1)
                        query_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                               instance=['2' for _ in range(crops2.shape[0])],
                                               to_orig=to_orig2)

                        # Inference and Matching
                        valid_pts3d, matches_im_query, matches_im_map, matches_conf = fine_matching(query_crop_view,
                                                                                                    map_crop_view,
                                                                                                    model, device,
                                                                                                    args.max_batch_size,
                                                                                                    args.pixel_tol,
                                                                                                    fast_nn_params)
                        matches_im_query = geotrf(query_to_orig_max, matches_im_query, norm=True)
                        matches_im_map = geotrf(map_to_orig_max, matches_im_map, norm=True)
                else:
                    # use only valid 2d points
                    valid_pts3d, matches_im_query, matches_im_map, matches_conf = coarse_matching(query_view, map_view,
                                                                                                  model, device,
                                                                                                  args.pixel_tol,
                                                                                                  fast_nn_params)
                if cache_file is not None:
                    mkdir_for(cache_file)
                    np.savez(cache_file, valid_pts3d=valid_pts3d, matches_im_query=matches_im_query,
                             matches_im_map=matches_im_map, matches_conf=matches_conf)

            # apply conf
            if len(matches_conf) > 0:
                mask = matches_conf >= conf_thr
                valid_pts3d = valid_pts3d[mask]
                matches_im_query = matches_im_query[mask]
                matches_im_map = matches_im_map[mask]
                matches_conf = matches_conf[mask]

            # visualize a few matches
            if viz_matches > 0:
                num_matches = matches_im_map.shape[0]
                print(f'found {num_matches} matches')

                viz_imgs = [np.array(query_view['rgb']), np.array(map_view['rgb'])]
                from matplotlib import pyplot as pl
                n_viz = viz_matches
                match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                viz_matches_im_query = matches_im_query[match_idx_to_viz]
                viz_matches_im_map = matches_im_map[match_idx_to_viz]

                H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img = np.concatenate((img0, img1), axis=1)
                pl.figure()
                pl.imshow(img)
                cmap = pl.get_cmap('jet')
                for i in range(n_viz):
                    (x0, y0), (x1, y1) = viz_matches_im_query[i].T, viz_matches_im_map[i].T
                    pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                pl.show(block=True)

            if len(valid_pts3d) == 0:
                pass
            else:
                query_pts3d.append(valid_pts3d)
                query_pts2d.append(matches_im_query)

        if len(query_pts2d) == 0:
            success = False
            pr_querycam_to_world = None
        else:
            query_pts2d = np.concatenate(query_pts2d, axis=0).astype(np.float32)
            query_pts3d = np.concatenate(query_pts3d, axis=0)
            if len(query_pts2d) > pnp_max_points:
                idxs = random.sample(range(len(query_pts2d)), pnp_max_points)
                query_pts3d = query_pts3d[idxs]
                query_pts2d = query_pts2d[idxs]

            W, H = query_view['rgb'].size
            if reprojection_error_diag_ratio is not None:
                reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(W**2 + H**2)
            else:
                reprojection_error_img = reprojection_error
            success, pr_querycam_to_world = run_pnp(query_pts2d, query_pts3d,
                                                    query_view['intrinsics'], query_view['distortion'],
                                                    pnp_mode, reprojection_error_img, img_size=[W, H])

        if not success:
            abs_transl_error = float('inf')
            abs_angular_error = float('inf')
        else:
            abs_transl_error, abs_angular_error = get_pose_error(pr_querycam_to_world, query_view['cam_to_world'])

        pose_errors.append(abs_transl_error)
        angular_errors.append(abs_angular_error)
        poses_pred.append(pr_querycam_to_world)

    xp_label = params_str + f'_conf_{conf_thr}'
    if args.output_label:
        xp_label = args.output_label + "_" + xp_label
    if reprojection_error_diag_ratio is not None:
        xp_label = xp_label + f'_reproj_diag_{reprojection_error_diag_ratio}'
    else:
        xp_label = xp_label + f'_reproj_err_{reprojection_error}'
    export_results(args.output_dir, xp_label, query_names, poses_pred)
    out_string = aggregate_stats(f'{args.dataset}', pose_errors, angular_errors)
    print(out_string)
