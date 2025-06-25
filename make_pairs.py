#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# make pairs using mast3r scene_graph, including retrieval
# --------------------------------------------------------
import argparse
import torch
import os
import os.path as path
import PIL
from PIL import Image
import pathlib
from kapture.io.csv import table_to_file

from mast3r.model import AsymmetricMASt3R
try:
    from mast3r.retrieval.processor import Retriever
    has_retrieval = True
except Exception as e:
    has_retrieval = False
from mast3r.image_pairs import make_pairs  # noqa


def get_argparser():
    parser = argparse.ArgumentParser(description='point triangulator with mast3r from kapture data')
    parser.add_argument('--dir', required=True, help='image dir')
    parser.add_argument('--scene_graph', default='retrieval-20-1-10-1')
    parser.add_argument('--output', required=True, help='txt file')

    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")

    return parser


def get_image_list(images_path):
    file_list = [path.relpath(path.join(dirpath, filename), images_path)
                 for dirpath, dirs, filenames in os.walk(images_path)
                 for filename in filenames]
    file_list = sorted(file_list)
    image_list = []
    for filename in file_list:
        # test if file is a valid image
        try:
            # lazy load
            with Image.open(path.join(images_path, filename)) as im:
                width, height = im.size
                image_list.append(filename)
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            print(f'Skipping invalid image file {filename}')
            continue
    return image_list


def main(dir, scene_graph, output, backbone=None, retrieval_model=None):
    imgs = get_image_list(dir)

    sim_matrix = None
    if 'retrieval' in scene_graph:
        assert has_retrieval
        retriever = Retriever(retrieval_model, backbone=backbone)
        imgs_fp = [path.join(dir, filename) for filename in imgs]
        with torch.no_grad():
            sim_matrix = retriever(imgs_fp)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    pairs = [(p1, p2, 1.0) for p1, p2 in pairs]
    pairs = sorted(set(pairs))

    os.umask(0o002)
    p = pathlib.Path(output)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)

    with open(output, 'w') as fid:
        table_to_file(fid, pairs, header='# query_image, map_image, score')


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    if "retrieval" in args.scene_graph:
        assert args.retrieval_model is not None
        if args.weights is not None:
            weights_path = args.weights
        else:
            weights_path = "naver/" + args.model_name
        backbone = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
        retrieval_model = args.retrieval_model
    else:
        backbone = None
        retrieval_model = None
    main(args.dir, args.scene_graph, args.output, backbone, retrieval_model)
