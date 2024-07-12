# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Fast Nearest Neighbor
# --------------------------------------------------------
import torch
import numpy as np
import math
from scipy.spatial import KDTree

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy, todevice  # noqa


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cuda', block_size=None, dist='l2'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


class cdistMatcher:
    def __init__(self, db_pts, device='cuda'):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(queries, self.db_pts, device=self.device, **kw)
        dis = None
        return dis, nnA


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != 'y_x':
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    if 'dist' in matcher_kw or 'block_size' in matcher_kw \
            or (isinstance(device, str) and device.startswith('cuda')) \
            or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(pts1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged

        # n_notyet.append(notyet.sum())
        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2


def extract_correspondences_nonsym(A, B, confA, confB, subsample=8, device=None, ptmap_key='pred_desc', pixel_tol=0):
    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    # merge corres from opposite pairs
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    if pixel_tol == 0:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)
    else:
        S = subsample
        yA, xA = np.mgrid[S // 2:HA:S, S // 2:WA:S].reshape(2, -1)
        yB, xB = np.mgrid[S // 2:HB:S, S // 2:WB:S].reshape(2, -1)

        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=(xA, yA), ret_xy=False, pixel_tol=pixel_tol, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=(xB, yB), ret_xy=False, pixel_tol=pixel_tol, **opt)

    idx1 = np.r_[nn1to2[0], nn2to1[1]]
    idx2 = np.r_[nn1to2[1], nn2to1[0]]

    c1 = confA.ravel()[idx1]
    c2 = confB.ravel()[idx2]

    xy1, xy2, idx = merge_corres(idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True)
    conf = np.minimum(c1[idx], c2[idx])
    corres = (xy1.copy(), xy2.copy(), conf)
    return todevice(corres, device)
