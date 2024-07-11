import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as pl

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from dust3r.cloud_opt.base_opt import clean_pointcloud


class TSDFPostProcess:
    """ Optimizes a signed distance-function to improve depthmaps.
    """

    def __init__(self, optimizer, subsample=8, TSDF_thresh=0., TSDF_batchsize=int(1e7)):
        self.TSDF_thresh = TSDF_thresh  # None -> no TSDF
        self.TSDF_batchsize = TSDF_batchsize
        self.optimizer = optimizer

        pts3d, depthmaps, confs = optimizer.get_dense_pts3d(clean_depth=False, subsample=subsample)
        pts3d, depthmaps = self._TSDF_postprocess_or_not(pts3d, depthmaps, confs)
        self.pts3d = pts3d
        self.depthmaps = depthmaps
        self.confs = confs

    def _get_depthmaps(self, TSDF_filtering_thresh=None):
        if TSDF_filtering_thresh:
            self._refine_depths_with_TSDF(self.optimizer, TSDF_filtering_thresh)  # compute refined depths if needed
        dms = self.TSDF_im_depthmaps if TSDF_filtering_thresh else self.im_depthmaps
        return [d.exp() for d in dms]

    @torch.no_grad()
    def _refine_depths_with_TSDF(self, TSDF_filtering_thresh, niter=1, nsamples=1000):
        """
        Leverage TSDF to post-process estimated depths
        for each pixel, find zero level of TSDF along ray (or closest to 0)
        """
        print("Post-Processing Depths with TSDF fusion.")
        self.TSDF_im_depthmaps = []
        alldepths, allposes, allfocals, allpps, allimshapes = self._get_depthmaps(), self.optimizer.get_im_poses(
        ), self.optimizer.get_focals(), self.optimizer.get_principal_points(), self.imshapes
        for vi in tqdm(range(self.optimizer.n_imgs)):
            dm, pose, focal, pp, imshape = alldepths[vi], allposes[vi], allfocals[vi], allpps[vi], allimshapes[vi]
            minvals = torch.full(dm.shape, 1e20)

            for it in range(niter):
                H, W = dm.shape
                curthresh = (niter - it) * TSDF_filtering_thresh
                dm_offsets = (torch.randn(H, W, nsamples).to(dm) - 1.) * \
                    curthresh  # decreasing search std along with iterations
                newdm = dm[..., None] + dm_offsets  # [H,W,Nsamp]
                curproj = self._backproj_pts3d(in_depths=[newdm], in_im_poses=pose[None], in_focals=focal[None], in_pps=pp[None], in_imshapes=[
                    imshape])[0]  # [H,W,Nsamp,3]
                # Batched TSDF eval
                curproj = curproj.view(-1, 3)
                tsdf_vals = []
                valids = []
                for batch in range(0, len(curproj), self.TSDF_batchsize):
                    values, valid = self._TSDF_query(
                        curproj[batch:min(batch + self.TSDF_batchsize, len(curproj))], curthresh)
                    tsdf_vals.append(values)
                    valids.append(valid)
                tsdf_vals = torch.cat(tsdf_vals, dim=0)
                valids = torch.cat(valids, dim=0)

                tsdf_vals = tsdf_vals.view([H, W, nsamples])
                valids = valids.view([H, W, nsamples])

                # keep depth value that got us the closest to 0
                tsdf_vals[~valids] = torch.inf  # ignore invalid values
                tsdf_vals = tsdf_vals.abs()
                mins = torch.argmin(tsdf_vals, dim=-1, keepdim=True)
                # when all samples live on a very flat zone, do nothing
                allbad = (tsdf_vals == curthresh).sum(dim=-1) == nsamples
                dm[~allbad] = torch.gather(newdm, -1, mins)[..., 0][~allbad]

            # Save refined depth map
            self.TSDF_im_depthmaps.append(dm.log())

    def _TSDF_query(self, qpoints, TSDF_filtering_thresh, weighted=True):
        """
        TSDF query call: returns the weighted TSDF value for each query point [N, 3]
        """
        N, three = qpoints.shape
        assert three == 3
        qpoints = qpoints[None].repeat(self.optimizer.n_imgs, 1, 1)  # [B,N,3]
        # get projection coordinates and depths onto images
        coords_and_depth = self._proj_pts3d(pts3d=qpoints, cam2worlds=self.optimizer.get_im_poses(
        ), focals=self.optimizer.get_focals(), pps=self.optimizer.get_principal_points())
        image_coords = coords_and_depth[..., :2].round().to(int)  # for now, there's no interpolation...
        proj_depths = coords_and_depth[..., -1]
        # recover depth values after scene optim
        pred_depths, pred_confs, valids = self._get_pixel_depths(image_coords)
        # Gather TSDF scores
        all_SDF_scores = pred_depths - proj_depths  # SDF
        unseen = all_SDF_scores < -TSDF_filtering_thresh  # handle visibility
        # all_TSDF_scores = all_SDF_scores.clip(-TSDF_filtering_thresh,TSDF_filtering_thresh) # SDF -> TSDF
        all_TSDF_scores = all_SDF_scores.clip(-TSDF_filtering_thresh, 1e20)  # SDF -> TSDF
        # Gather TSDF confidences and ignore points that are unseen, either OOB during reproj or too far behind seen depth
        all_TSDF_weights = (~unseen).float() * valids.float()
        if weighted:
            all_TSDF_weights = pred_confs.exp() * all_TSDF_weights
        # Aggregate all votes, ignoring zeros
        TSDF_weights = all_TSDF_weights.sum(dim=0)
        valids = TSDF_weights != 0.
        TSDF_wsum = (all_TSDF_weights * all_TSDF_scores).sum(dim=0)
        TSDF_wsum[valids] /= TSDF_weights[valids]
        return TSDF_wsum, valids

    def _get_pixel_depths(self, image_coords, TSDF_filtering_thresh=None, with_normals_conf=False):
        """ Recover depth value for each input pixel coordinate, along with OOB validity mask
        """
        B, N, two = image_coords.shape
        assert B == self.optimizer.n_imgs and two == 2
        depths = torch.zeros([B, N], device=image_coords.device)
        valids = torch.zeros([B, N], dtype=bool, device=image_coords.device)
        confs = torch.zeros([B, N], device=image_coords.device)
        curconfs = self._get_confs_with_normals() if with_normals_conf else self.im_conf
        for ni, (imc, depth, conf) in enumerate(zip(image_coords, self._get_depthmaps(TSDF_filtering_thresh), curconfs)):
            H, W = depth.shape
            valids[ni] = torch.logical_and(0 <= imc[:, 1], imc[:, 1] <
                                           H) & torch.logical_and(0 <= imc[:, 0], imc[:, 0] < W)
            imc[~valids[ni]] = 0
            depths[ni] = depth[imc[:, 1], imc[:, 0]]
            confs[ni] = conf.cuda()[imc[:, 1], imc[:, 0]]
        return depths, confs, valids

    def _get_confs_with_normals(self):
        outconfs = []
        # Confidence basedf on depth gradient

        class Sobel(nn.Module):
            def __init__(self):
                super().__init__()
                self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
                Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
                Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
                G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
                G = G.unsqueeze(1)
                self.filter.weight = nn.Parameter(G, requires_grad=False)

            def forward(self, img):
                x = self.filter(img)
                x = torch.mul(x, x)
                x = torch.sum(x, dim=1, keepdim=True)
                x = torch.sqrt(x)
                return x

        grad_op = Sobel().to(self.im_depthmaps[0].device)
        for conf, depth in zip(self.im_conf, self.im_depthmaps):
            grad_confs = (1. - grad_op(depth[None, None])[0, 0]).clip(0)
            if not 'dbg show':
                pl.imshow(grad_confs.cpu())
                pl.show()
            outconfs.append(conf * grad_confs.to(conf))
        return outconfs

    def _proj_pts3d(self, pts3d, cam2worlds, focals, pps):
        """
        Projection operation: from 3D points to 2D coordinates + depths
        """
        B = pts3d.shape[0]
        assert pts3d.shape[0] == cam2worlds.shape[0]
        # prepare Extrinsincs
        R, t = cam2worlds[:, :3, :3], cam2worlds[:, :3, -1]
        Rinv = R.transpose(-2, -1)
        tinv = -Rinv @ t[..., None]

        # prepare intrinsics
        intrinsics = torch.eye(3).to(cam2worlds)[None].repeat(focals.shape[0], 1, 1)
        if len(focals.shape) == 1:
            focals = torch.stack([focals, focals], dim=-1)
        intrinsics[:, 0, 0] = focals[:, 0]
        intrinsics[:, 1, 1] = focals[:, 1]
        intrinsics[:, :2, -1] = pps
        # Project
        projpts = intrinsics @ (Rinv @ pts3d.transpose(-2, -1) + tinv)  # I(RX+t) : [B,3,N]
        projpts = projpts.transpose(-2, -1)  # [B,N,3]
        projpts[..., :2] /= projpts[..., [-1]]  # [B,N,3] (X/Z , Y/Z, Z)
        return projpts

    def _backproj_pts3d(self, in_depths=None, in_im_poses=None,
                        in_focals=None, in_pps=None, in_imshapes=None):
        """
        Backprojection operation: from image depths to 3D points
        """
        # Get depths and  projection params if not provided
        focals = self.optimizer.get_focals() if in_focals is None else in_focals
        im_poses = self.optimizer.get_im_poses() if in_im_poses is None else in_im_poses
        depth = self._get_depthmaps() if in_depths is None else in_depths
        pp = self.optimizer.get_principal_points() if in_pps is None else in_pps
        imshapes = self.imshapes if in_imshapes is None else in_imshapes
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *imshapes[i])
        dm_to_3d = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[[i]]) for i in range(im_poses.shape[0])]

        def autoprocess(x):
            x = x[0]
            return x.transpose(-2, -1) if len(x.shape) == 4 else x
        return [geotrf(pose, autoprocess(pt)) for pose, pt in zip(im_poses, dm_to_3d)]

    def _pts3d_to_depth(self, pts3d, cam2worlds, focals, pps):
        """
        Projection operation: from 3D points to 2D coordinates + depths
        """
        B = pts3d.shape[0]
        assert pts3d.shape[0] == cam2worlds.shape[0]
        # prepare Extrinsincs
        R, t = cam2worlds[:, :3, :3], cam2worlds[:, :3, -1]
        Rinv = R.transpose(-2, -1)
        tinv = -Rinv @ t[..., None]

        # prepare intrinsics
        intrinsics = torch.eye(3).to(cam2worlds)[None].repeat(self.optimizer.n_imgs, 1, 1)
        if len(focals.shape) == 1:
            focals = torch.stack([focals, focals], dim=-1)
        intrinsics[:, 0, 0] = focals[:, 0]
        intrinsics[:, 1, 1] = focals[:, 1]
        intrinsics[:, :2, -1] = pps
        # Project
        projpts = intrinsics @ (Rinv @ pts3d.transpose(-2, -1) + tinv)  # I(RX+t) : [B,3,N]
        projpts = projpts.transpose(-2, -1)  # [B,N,3]
        projpts[..., :2] /= projpts[..., [-1]]  # [B,N,3] (X/Z , Y/Z, Z)
        return projpts

    def _depth_to_pts3d(self, in_depths=None, in_im_poses=None, in_focals=None, in_pps=None, in_imshapes=None):
        """
        Backprojection operation: from image depths to 3D points
        """
        # Get depths and  projection params if not provided
        focals = self.optimizer.get_focals() if in_focals is None else in_focals
        im_poses = self.optimizer.get_im_poses() if in_im_poses is None else in_im_poses
        depth = self._get_depthmaps() if in_depths is None else in_depths
        pp = self.optimizer.get_principal_points() if in_pps is None else in_pps
        imshapes = self.imshapes if in_imshapes is None else in_imshapes

        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *imshapes[i])

        dm_to_3d = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i + 1]) for i in range(im_poses.shape[0])]

        def autoprocess(x):
            x = x[0]
            H, W, three = x.shape[:3]
            return x.transpose(-2, -1) if len(x.shape) == 4 else x
        return [geotrf(pp, autoprocess(pt)) for pp, pt in zip(im_poses, dm_to_3d)]

    def _get_pts3d(self, TSDF_filtering_thresh=None, **kw):
        """
        return 3D points (possibly filtering depths with TSDF) 
        """
        return self._backproj_pts3d(in_depths=self._get_depthmaps(TSDF_filtering_thresh=TSDF_filtering_thresh), **kw)

    def _TSDF_postprocess_or_not(self, pts3d, depthmaps, confs, niter=1):
        # Setup inner variables
        self.imshapes = [im.shape[:2] for im in self.optimizer.imgs]
        self.im_depthmaps = [dd.log().view(imshape) for dd, imshape in zip(depthmaps, self.imshapes)]
        self.im_conf = confs

        if self.TSDF_thresh > 0.:
            # Create or update self.TSDF_im_depthmaps that contain logdepths filtered with TSDF
            self._refine_depths_with_TSDF(self.TSDF_thresh, niter=niter)
            depthmaps = [dd.exp() for dd in self.TSDF_im_depthmaps]
            # Turn them into 3D points
            pts3d = self._backproj_pts3d(in_depths=depthmaps)
            depthmaps = [dd.flatten() for dd in depthmaps]
            pts3d = [pp.view(-1, 3) for pp in pts3d]
        return pts3d, depthmaps

    def get_dense_pts3d(self, clean_depth=True):
        if clean_depth:
            confs = clean_pointcloud(self.confs, self.optimizer.intrinsics, inv(self.optimizer.cam2w),
                                     self.depthmaps, self.pts3d)
        return self.pts3d, self.depthmaps, confs
