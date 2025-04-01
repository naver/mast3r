#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# mast3r exec for running standard sfm
# --------------------------------------------------------
import pycolmap
import os
import os.path as path
import argparse

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import (kapture_import_image_folder_or_list, run_mast3r_matching, pycolmap_run_triangulator,
                                   pycolmap_run_mapper, glomap_run_mapper)
from kapture.io.csv import kapture_from_dir

from kapture.converter.colmap.database_extra import kapture_to_colmap, generate_priors_for_reconstruction
from kapture_localization.utils.pairsfile import get_pairs_from_file
from kapture.io.records import get_image_fullpath
from kapture.converter.colmap.database import COLMAPDatabase


def get_argparser():
    parser = argparse.ArgumentParser(description='point triangulator with mast3r from kapture data')
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])

    parser_input = parser.add_mutually_exclusive_group(required=True)
    parser_input.add_argument('-i', '--input', default=None, help='kdata')
    parser_input.add_argument('--dir', default=None, help='image dir (individual intrinsics)')
    parser_input.add_argument('--dir_same_camera', default=None, help='image dir (shared intrinsics)')

    parser.add_argument('-o', '--output', required=True, help='output path to reconstruction')
    parser.add_argument('--pairsfile_path', required=True, help='pairsfile')

    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')

    parser_mapper = parser.add_mutually_exclusive_group()
    parser_mapper.add_argument('--ignore_pose', action='store_true', default=False)
    parser_mapper.add_argument('--use_glomap_mapper', action='store_true', default=False)

    parser_matching = parser.add_mutually_exclusive_group()
    parser_matching.add_argument('--dense_matching', action='store_true', default=False)
    parser_matching.add_argument('--pixel_tol', default=0, type=int)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--conf_thr', default=1.001, type=float)
    parser.add_argument('--skip_geometric_verification', action='store_true', default=False)
    parser.add_argument('--min_len_track', default=5, type=int)

    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    maxdim = max(model.patch_embed.img_size)
    patch_size = model.patch_embed.patch_size

    if args.input is not None:
        kdata = kapture_from_dir(args.input)
        records_data_path = get_image_fullpath(args.input)
    else:
        if args.dir_same_camera is not None:
            use_single_camera = True
            records_data_path = args.dir_same_camera
        elif args.dir is not None:
            use_single_camera = False
            records_data_path = args.dir
        else:
            raise ValueError('all inputs choices are None')
        kdata = kapture_import_image_folder_or_list(records_data_path, use_single_camera)
    has_pose = kdata.trajectories is not None
    image_pairs = get_pairs_from_file(args.pairsfile_path, kdata.records_camera, kdata.records_camera)

    colmap_db_path = path.join(args.output, 'colmap.db')
    reconstruction_path = path.join(args.output, "reconstruction")
    priors_txt_path = path.join(args.output, "priors_for_reconstruction")
    for path_i in [reconstruction_path, priors_txt_path]:
        os.makedirs(path_i, exist_ok=True)
    assert not os.path.isfile(colmap_db_path)

    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, args.input, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        if has_pose:
            generate_priors_for_reconstruction(kdata, colmap_db, priors_txt_path)

        colmap_image_pairs = run_mast3r_matching(model, maxdim, patch_size, args.device,
                                                 kdata, records_data_path, image_pairs, colmap_db,
                                                 args.dense_matching, args.pixel_tol, args.conf_thr,
                                                 args.skip_geometric_verification, args.min_len_track)
        colmap_db.close()
    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        exit(1)

    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    # colmap db is now full, run colmap
    colmap_world_to_cam = {}
    if not args.skip_geometric_verification:
        print("verify_matches")
        f = open(args.output + '/pairs.txt', "w")
        for image_path1, image_path2 in colmap_image_pairs:
            f.write("{} {}\n".format(image_path1, image_path2))
        f.close()
        pycolmap.verify_matches(colmap_db_path, args.output + '/pairs.txt')

    print("running mapping")
    if has_pose and not args.ignore_pose and not args.use_glomap_mapper:
        pycolmap_run_triangulator(colmap_db_path, priors_txt_path, reconstruction_path, records_data_path)
    elif not args.use_glomap_mapper:
        pycolmap_run_mapper(colmap_db_path, reconstruction_path, records_data_path)
    else:
        glomap_run_mapper(args.glomap_bin, colmap_db_path, reconstruction_path, records_data_path)
