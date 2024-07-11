# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Matches Triangulation Utils
# --------------------------------------------------------

import numpy as np
import torch

# Batched Matches Triangulation
def batched_triangulate(pts2d,        # [B, Ncams, Npts, 2]
                        proj_mats):   # [B, Ncams, 3, 4] I@E projection matrix
    B, Ncams, Npts, two = pts2d.shape
    assert two==2
    assert proj_mats.shape == (B, Ncams, 3, 4)
    # P - xP
    x = proj_mats[...,0,:][...,None,:] - torch.einsum('bij,bik->bijk', pts2d[...,0], proj_mats[...,2,:]) # [B, Ncams, Npts, 4]
    y = proj_mats[...,1,:][...,None,:] - torch.einsum('bij,bik->bijk', pts2d[...,1], proj_mats[...,2,:]) # [B, Ncams, Npts, 4]
    eq = torch.cat([x, y], dim=1).transpose(1, 2) # [B, Npts, 2xNcams, 4]
    return torch.linalg.lstsq(eq[...,:3], -eq[...,3]).solution

def matches_to_depths(intrinsics, # input camera intrinsics     [B, Ncams, 3, 3]
                      extrinsics, # input camera extrinsics     [B, Ncams, 3, 4]
                      matches,    # input correspondences       [B, Ncams, Npts, 2]
                      batchsize=16, # bs for batched processing 
                      min_num_valids_ratio=.3 # at least this ratio of image pairs need to predict a match for a given pixel of img1
                      ):
    B, Nv, H, W, five = matches.shape 
    min_num_valids = np.floor(Nv*min_num_valids_ratio)
    out_aggregated_points, out_depths, out_confs = [], [], []
    for b in range(B//batchsize+1): # batched processing 
        start, stop = b*batchsize,min(B,(b+1)*batchsize)
        sub_batch=slice(start,stop) 
        sub_batchsize = stop-start
        if sub_batchsize==0:continue
        points1, points2, confs = matches[sub_batch, ..., :2], matches[sub_batch, ..., 2:4], matches[sub_batch, ..., -1]
        allpoints = torch.cat([points1.view([sub_batchsize*Nv,1,H*W,2]), points2.view([sub_batchsize*Nv,1,H*W,2])],dim=1) # [BxNv, 2, HxW, 2]

        allcam_Ps = intrinsics[sub_batch] @ extrinsics[sub_batch,:,:3,:]
        cam_Ps1, cam_Ps2 = allcam_Ps[:,[0]].repeat([1,Nv,1,1]), allcam_Ps[:,1:] # [B, Nv, 3, 4]
        formatted_camPs = torch.cat([cam_Ps1.reshape([sub_batchsize*Nv,1,3,4]), cam_Ps2.reshape([sub_batchsize*Nv,1,3,4])],dim=1) # [BxNv, 2, 3, 4]
        
        # Triangulate matches to 3D
        points_3d_world = batched_triangulate(allpoints, formatted_camPs) # [BxNv, HxW, three] 
        
        # Aggregate pairwise predictions
        points_3d_world = points_3d_world.view([sub_batchsize,Nv,H,W,3])
        valids = points_3d_world.isfinite()
        valids_sum = valids.sum(dim=-1)
        validsuni=valids_sum.unique()
        assert torch.all(torch.logical_or(validsuni == 0 , validsuni == 3)), "Error, can only be nan for none or all XYZ values, not a subset"
        confs[valids_sum==0] = 0.
        points_3d_world = points_3d_world*confs[...,None]
        
        # Take care of NaNs
        normalization = confs.sum(dim=1)[:,None].repeat(1,Nv,1,1)
        normalization[normalization <= 1e-5] = 1.
        points_3d_world[valids] /= normalization[valids_sum==3][:,None].repeat(1,3).view(-1)
        points_3d_world[~valids] = 0.
        aggregated_points = points_3d_world.sum(dim=1) # weighted average (by confidence value) ignoring nans
        
        # Reset invalid values to nans, with a min visibility threshold
        aggregated_points[valids_sum.sum(dim=1)/3 <= min_num_valids] = torch.nan
        
        # From 3D to depths
        refcamE = extrinsics[sub_batch, 0]
        points_3d_camera = (refcamE[:,:3, :3] @ aggregated_points.view(sub_batchsize,-1,3).transpose(-2,-1) + refcamE[:,:3,[3]]).transpose(-2,-1) # [B,HxW,3]
        depths = points_3d_camera.view(sub_batchsize,H,W,3)[..., 2] # [B,H,W]
        
        # Cat results
        out_aggregated_points.append(aggregated_points.cpu())
        out_depths.append(depths.cpu())
        out_confs.append(confs.sum(dim=1).cpu())
    
    out_aggregated_points = torch.cat(out_aggregated_points,dim=0)
    out_depths            = torch.cat(out_depths,dim=0)
    out_confs             = torch.cat(out_confs,dim=0)
            
    return out_aggregated_points, out_depths, out_confs
