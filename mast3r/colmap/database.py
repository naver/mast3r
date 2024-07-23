# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R to colmap export functions
# --------------------------------------------------------
import os
import torch
import copy
import numpy as np
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.transform import Rotation as R

from mast3r.utils.misc import hash_md5

from mast3r.fast_nn import extract_correspondences_nonsym, bruteforce_reciprocal_nns

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid, geotrf  # noqa


def convert_im_matches_pairs(img0, img1, image_to_colmap, im_keypoints, matches_im0, matches_im1, viz):
    if viz:
        from matplotlib import pyplot as pl

        image_mean = torch.as_tensor(
            [0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor(
            [0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        rgb0 = img0['img'] * image_std + image_mean
        rgb0 = torchvision.transforms.functional.to_pil_image(rgb0[0])
        rgb0 = np.array(rgb0)

        rgb1 = img1['img'] * image_std + image_mean
        rgb1 = torchvision.transforms.functional.to_pil_image(rgb1[0])
        rgb1 = np.array(rgb1)

        imgs = [rgb0, rgb1]
        # visualize a few matches
        n_viz = 100
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(
            0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        rgb0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)),
                                (0, 0), (0, 0)), 'constant', constant_values=0)
        rgb1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)),
                                (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((rgb0, rgb1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for ii in range(n_viz):
            (x0, y0), (x1,
                       y1) = viz_matches_im0[ii].T, viz_matches_im1[ii].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(ii /
                    (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)

    matches = [matches_im0.astype(np.float64), matches_im1.astype(np.float64)]
    imgs = [img0, img1]
    imidx0 = img0['idx']
    imidx1 = img1['idx']
    ravel_matches = []
    for j in range(2):
        H, W = imgs[j]['true_shape'][0]
        with np.errstate(invalid='ignore'):
            qx, qy = matches[j].round().astype(np.int32).T
        ravel_matches_j = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(min=0, max=H - 1, out=qy)
        ravel_matches.append(ravel_matches_j)
        imidxj = imgs[j]['idx']
        for m in ravel_matches_j:
            if m not in im_keypoints[imidxj]:
                im_keypoints[imidxj][m] = 0
            im_keypoints[imidxj][m] += 1
    imid0 = copy.deepcopy(image_to_colmap[imidx0]['colmap_imid'])
    imid1 = copy.deepcopy(image_to_colmap[imidx1]['colmap_imid'])
    if imid0 > imid1:
        colmap_matches = np.stack([ravel_matches[1], ravel_matches[0]], axis=-1)
        imid0, imid1 = imid1, imid0
        imidx0, imidx1 = imidx1, imidx0
    else:
        colmap_matches = np.stack([ravel_matches[0], ravel_matches[1]], axis=-1)
    colmap_matches = np.unique(colmap_matches, axis=0)
    return imidx0, imidx1, colmap_matches


def get_im_matches(pred1, pred2, pairs, image_to_colmap, im_keypoints, conf_thr,
                   is_sparse=True, subsample=8, pixel_tol=0, viz=False, device='cuda'):
    im_matches = {}
    for i in range(len(pred1['pts3d'])):
        imidx0 = pairs[i][0]['idx']
        imidx1 = pairs[i][1]['idx']
        if 'desc' in pred1:  # mast3r
            descs = [pred1['desc'][i], pred2['desc'][i]]
            confidences = [pred1['desc_conf'][i], pred2['desc_conf'][i]]
            desc_dim = descs[0].shape[-1]

            if is_sparse:
                corres = extract_correspondences_nonsym(descs[0], descs[1], confidences[0], confidences[1],
                                                        device=device, subsample=subsample, pixel_tol=pixel_tol)
                conf = corres[2]
                mask = conf >= conf_thr
                matches_im0 = corres[0][mask].cpu().numpy()
                matches_im1 = corres[1][mask].cpu().numpy()
            else:
                confidence_masks = [confidences[0] >=
                                    conf_thr, confidences[1] >= conf_thr]
                pts2d_list, desc_list = [], []
                for j in range(2):
                    conf_j = confidence_masks[j].cpu().numpy().flatten()
                    true_shape_j = pairs[i][j]['true_shape'][0]
                    pts2d_j = xy_grid(
                        true_shape_j[1], true_shape_j[0]).reshape(-1, 2)[conf_j]
                    desc_j = descs[j].detach().cpu(
                    ).numpy().reshape(-1, desc_dim)[conf_j]
                    pts2d_list.append(pts2d_j)
                    desc_list.append(desc_j)
                if len(desc_list[0]) == 0 or len(desc_list[1]) == 0:
                    continue

                nn0, nn1 = bruteforce_reciprocal_nns(desc_list[0], desc_list[1],
                                                     device=device, dist='dot', block_size=2**13)
                reciprocal_in_P0 = (nn1[nn0] == np.arange(len(nn0)))

                matches_im1 = pts2d_list[1][nn0][reciprocal_in_P0]
                matches_im0 = pts2d_list[0][reciprocal_in_P0]
        else:
            pts3d = [pred1['pts3d'][i], pred2['pts3d_in_other_view'][i]]
            confidences = [pred1['conf'][i], pred2['conf'][i]]

            if is_sparse:
                corres = extract_correspondences_nonsym(pts3d[0], pts3d[1], confidences[0], confidences[1],
                                                        device=device, subsample=subsample, pixel_tol=pixel_tol,
                                                        ptmap_key='3d')
                conf = corres[2]
                mask = conf >= conf_thr
                matches_im0 = corres[0][mask].cpu().numpy()
                matches_im1 = corres[1][mask].cpu().numpy()
            else:
                confidence_masks = [confidences[0] >=
                                    conf_thr, confidences[1] >= conf_thr]
                # find 2D-2D matches between the two images
                pts2d_list, pts3d_list = [], []
                for j in range(2):
                    conf_j = confidence_masks[j].cpu().numpy().flatten()
                    true_shape_j = pairs[i][j]['true_shape'][0]
                    pts2d_j = xy_grid(true_shape_j[1], true_shape_j[0]).reshape(-1, 2)[conf_j]
                    pts3d_j = pts3d[j].detach().cpu().numpy().reshape(-1, 3)[conf_j]
                    pts2d_list.append(pts2d_j)
                    pts3d_list.append(pts3d_j)

                PQ, PM = pts3d_list[0], pts3d_list[1]
                if len(PQ) == 0 or len(PM) == 0:
                    continue
                reciprocal_in_PM, nnM_in_PQ, num_matches = find_reciprocal_matches(
                    PQ, PM)

                matches_im1 = pts2d_list[1][reciprocal_in_PM]
                matches_im0 = pts2d_list[0][nnM_in_PQ][reciprocal_in_PM]

        if len(matches_im0) == 0:
            continue
        imidx0, imidx1, colmap_matches = convert_im_matches_pairs(pairs[i][0], pairs[i][1],
                                                                  image_to_colmap, im_keypoints,
                                                                  matches_im0, matches_im1, viz)
        im_matches[(imidx0, imidx1)] = colmap_matches
    return im_matches


def get_im_matches_from_cache(pairs, cache_path, desc_conf, subsample,
                              image_to_colmap, im_keypoints, conf_thr,
                              viz=False, device='cuda'):
    im_matches = {}
    for i in range(len(pairs)):
        imidx0 = pairs[i][0]['idx']
        imidx1 = pairs[i][1]['idx']

        corres_idx1 = hash_md5(pairs[i][0]['instance'])
        corres_idx2 = hash_md5(pairs[i][1]['instance'])

        path_corres = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{corres_idx1}-{corres_idx2}.pth'
        if os.path.isfile(path_corres):
            score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
        else:
            path_corres = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{corres_idx2}-{corres_idx1}.pth'
            score, (xy2, xy1, confs) = torch.load(path_corres, map_location=device)
        mask = confs >= conf_thr
        matches_im0 = xy1[mask].cpu().numpy()
        matches_im1 = xy2[mask].cpu().numpy()

        if len(matches_im0) == 0:
            continue
        imidx0, imidx1, colmap_matches = convert_im_matches_pairs(pairs[i][0], pairs[i][1],
                                                                  image_to_colmap, im_keypoints,
                                                                  matches_im0, matches_im1, viz)
        im_matches[(imidx0, imidx1)] = colmap_matches
    return im_matches


def export_images(db, images, image_paths, focals, ga_world_to_cam, camera_model):
    # add cameras/images to the db
    # with the output of ga as prior
    image_to_colmap = {}
    im_keypoints = {}
    for idx in range(len(image_paths)):
        im_keypoints[idx] = {}
        H, W = images[idx]["orig_shape"]
        if focals is None:
            focal_x = focal_y = 1.2 * max(W, H)
            prior_focal_length = False
            cx = W / 2.0
            cy = H / 2.0
        elif isinstance(focals[idx], np.ndarray) and len(focals[idx].shape) == 2:
            # intrinsics
            focal_x = focals[idx][0, 0]
            focal_y = focals[idx][1, 1]
            cx = focals[idx][0, 2] * images[idx]["to_orig"][0, 0]
            cy = focals[idx][1, 2] * images[idx]["to_orig"][1, 1]
            prior_focal_length = True
        else:
            focal_x = focal_y = float(focals[idx])
            prior_focal_length = True
            cx = W / 2.0
            cy = H / 2.0
        focal_x = focal_x * images[idx]["to_orig"][0, 0]
        focal_y = focal_y * images[idx]["to_orig"][1, 1]

        if camera_model == "SIMPLE_PINHOLE":
            model_id = 0
            focal = (focal_x + focal_y) / 2.0
            params = np.asarray([focal, cx, cy], np.float64)
        elif camera_model == "PINHOLE":
            model_id = 1
            params = np.asarray([focal_x, focal_y, cx, cy], np.float64)
        elif camera_model == "SIMPLE_RADIAL":
            model_id = 2
            focal = (focal_x + focal_y) / 2.0
            params = np.asarray([focal, cx, cy, 0.0], np.float64)
        elif camera_model == "OPENCV":
            model_id = 4
            params = np.asarray([focal_x, focal_y, cx, cy, 0.0, 0.0, 0.0, 0.0], np.float64)
        else:
            raise ValueError(f"invalid camera model {camera_model}")

        H, W = int(H), int(W)
        # OPENCV camera model
        camid = db.add_camera(
            model_id, W, H, params, prior_focal_length=prior_focal_length)
        if ga_world_to_cam is None:
            prior_t = np.zeros(3)
            prior_q = np.zeros(4)
        else:
            q = R.from_matrix(ga_world_to_cam[idx][:3, :3]).as_quat()
            prior_t = ga_world_to_cam[idx][:3, 3]
            prior_q = np.array([q[-1], q[0], q[1], q[2]])
        imid = db.add_image(
            image_paths[idx], camid, prior_q=prior_q, prior_t=prior_t)
        image_to_colmap[idx] = {
            'colmap_imid': imid,
            'colmap_camid': camid
        }
    return image_to_colmap, im_keypoints


def export_matches(db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification):
    colmap_image_pairs = []
    # 2D-2D are quite dense
    # we want to remove the very small tracks
    # and export only kpt for which we have values
    # build tracks
    print("building tracks")
    keypoints_to_track_id = {}
    track_id_to_kpt_list = []
    to_merge = []
    for (imidx0, imidx1), colmap_matches in tqdm(im_matches.items()):
        if imidx0 not in keypoints_to_track_id:
            keypoints_to_track_id[imidx0] = {}
        if imidx1 not in keypoints_to_track_id:
            keypoints_to_track_id[imidx1] = {}

        for m in colmap_matches:
            if m[0] not in keypoints_to_track_id[imidx0] and m[1] not in keypoints_to_track_id[imidx1]:
                # new pair of kpts never seen before
                track_idx = len(track_id_to_kpt_list)
                keypoints_to_track_id[imidx0][m[0]] = track_idx
                keypoints_to_track_id[imidx1][m[1]] = track_idx
                track_id_to_kpt_list.append(
                    [(imidx0, m[0]), (imidx1, m[1])])
            elif m[1] not in keypoints_to_track_id[imidx1]:
                # 0 has a track, not 1
                track_idx = keypoints_to_track_id[imidx0][m[0]]
                keypoints_to_track_id[imidx1][m[1]] = track_idx
                track_id_to_kpt_list[track_idx].append((imidx1, m[1]))
            elif m[0] not in keypoints_to_track_id[imidx0]:
                # 1 has a track, not 0
                track_idx = keypoints_to_track_id[imidx1][m[1]]
                keypoints_to_track_id[imidx0][m[0]] = track_idx
                track_id_to_kpt_list[track_idx].append((imidx0, m[0]))
            else:
                # both have tracks, merge them
                track_idx0 = keypoints_to_track_id[imidx0][m[0]]
                track_idx1 = keypoints_to_track_id[imidx1][m[1]]
                if track_idx0 != track_idx1:
                    # let's deal with them later
                    to_merge.append((track_idx0, track_idx1))

    # regroup merge targets
    print("merging tracks")
    unique = np.unique(to_merge)
    tree = DisjointSet(unique)
    for track_idx0, track_idx1 in tqdm(to_merge):
        tree.merge(track_idx0, track_idx1)

    subsets = tree.subsets()
    print("applying merge")
    for setvals in tqdm(subsets):
        new_trackid = len(track_id_to_kpt_list)
        kpt_list = []
        for track_idx in setvals:
            kpt_list.extend(track_id_to_kpt_list[track_idx])
            for imidx, kpid in track_id_to_kpt_list[track_idx]:
                keypoints_to_track_id[imidx][kpid] = new_trackid
        track_id_to_kpt_list.append(kpt_list)

    # binc = np.bincount([len(v) for v in track_id_to_kpt_list])
    # nonzero = np.nonzero(binc)
    # nonzerobinc = binc[nonzero[0]]
    # print(nonzero[0].tolist())
    # print(nonzerobinc)
    num_valid_tracks = sum(
        [1 for v in track_id_to_kpt_list if len(v) >= min_len_track])

    keypoints_to_idx = {}
    print(f"squashing keypoints - {num_valid_tracks} valid tracks")
    for imidx, keypoints_imid in tqdm(im_keypoints.items()):
        imid = image_to_colmap[imidx]['colmap_imid']
        keypoints_kept = []
        keypoints_to_idx[imidx] = {}
        for kp in keypoints_imid.keys():
            if kp not in keypoints_to_track_id[imidx]:
                continue
            track_idx = keypoints_to_track_id[imidx][kp]
            track_length = len(track_id_to_kpt_list[track_idx])
            if track_length < min_len_track:
                continue
            keypoints_to_idx[imidx][kp] = len(keypoints_kept)
            keypoints_kept.append(kp)
        if len(keypoints_kept) == 0:
            continue
        keypoints_kept = np.array(keypoints_kept)
        keypoints_kept = np.unravel_index(keypoints_kept, images[imidx]['true_shape'][0])[
            0].base[:, ::-1].copy().astype(np.float32)
        # rescale coordinates
        keypoints_kept[:, 0] += 0.5
        keypoints_kept[:, 1] += 0.5
        keypoints_kept = geotrf(images[imidx]['to_orig'], keypoints_kept, norm=True)

        H, W = images[imidx]['orig_shape']
        keypoints_kept[:, 0] = keypoints_kept[:, 0].clip(min=0, max=W - 0.01)
        keypoints_kept[:, 1] = keypoints_kept[:, 1].clip(min=0, max=H - 0.01)

        db.add_keypoints(imid, keypoints_kept)

    print("exporting im_matches")
    for (imidx0, imidx1), colmap_matches in im_matches.items():
        imid0, imid1 = image_to_colmap[imidx0]['colmap_imid'], image_to_colmap[imidx1]['colmap_imid']
        assert imid0 < imid1
        final_matches = np.array([[keypoints_to_idx[imidx0][m[0]], keypoints_to_idx[imidx1][m[1]]]
                                  for m in colmap_matches
                                  if m[0] in keypoints_to_idx[imidx0] and m[1] in keypoints_to_idx[imidx1]])
        if len(final_matches) > 0:
            colmap_image_pairs.append(
                (images[imidx0]['instance'], images[imidx1]['instance']))
            db.add_matches(imid0, imid1, final_matches)
            if skip_geometric_verification:
                db.add_two_view_geometry(imid0, imid1, final_matches)
    return colmap_image_pairs
