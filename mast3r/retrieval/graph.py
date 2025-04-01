# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Building the graph based on retrieval results.
# --------------------------------------------------------
import numpy as np


def farthest_point_sampling(dist, N=None, dist_thresh=None):
    """Farthest point sampling.

    Args:
        dist: NxN distance matrix.
        N: Number of points to sample.
        dist_thresh: Distance threshold. Point sampling terminates once the
                     maximum distance is below this threshold.

    Returns:
        indices: Indices of the sampled points.
    """

    assert N is not None or dist_thresh is not None, "Either N or min_dist must be provided."

    if N is None:
        N = dist.shape[0]

    indices = []
    distances = [0]
    indices.append(np.random.choice(dist.shape[0]))
    for i in range(1, N):
        d = dist[indices].min(axis=0)
        bst = d.argmax()
        bst_dist = d[bst]
        if dist_thresh is not None and bst_dist < dist_thresh:
            break
        indices.append(bst)
        distances.append(bst_dist)
    return np.array(indices), np.array(distances)


def make_pairs_fps(sim_mat, Na=20, tokK=1, dist_thresh=None):
    dist_mat = 1 - sim_mat

    pairs = set()
    keyimgs_idx = np.array([])
    if Na != 0:
        keyimgs_idx, _ = farthest_point_sampling(dist_mat, N=Na, dist_thresh=dist_thresh)

        # 1. Complete graph between key images
        for i in range(len(keyimgs_idx)):
            for j in range(i + 1, len(keyimgs_idx)):
                idx_i, idx_j = keyimgs_idx[i], keyimgs_idx[j]
                pairs.add((idx_i, idx_j))

        # 2. Connect non-key images to the earest key image
        keyimg_dist_mat = dist_mat[:, keyimgs_idx]
        for i in range(keyimg_dist_mat.shape[0]):
            if i in keyimgs_idx:
                continue
            j = keyimg_dist_mat[i].argmin()
            i1, i2 = min(i, keyimgs_idx[j]), max(i, keyimgs_idx[j])
            if i1 != i2 and (i1, i2) not in pairs:
                pairs.add((i1, i2))

    # 3. Add some local connections (k-NN) for each view
    if tokK > 0:
        for i in range(dist_mat.shape[0]):
            idx = dist_mat[i].argsort()[:tokK]
            for j in idx:
                i1, i2 = min(i, j), max(i, j)
                if i1 != i2 and (i1, i2) not in pairs:
                    pairs.add((i1, i2))

    pairs = list(pairs)

    return pairs, keyimgs_idx
