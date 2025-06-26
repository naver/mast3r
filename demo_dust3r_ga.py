#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# mast3r gradio demo executable
# --------------------------------------------------------
import os
import torch
import tempfile

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from mast3r.model import load_dune_mast3r_model
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from dust3r.demo import main_demo, set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def get_args_parser():
    parser = dust3r_get_args_parser()

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices.append('MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
        if action.dest == 'image_size':
            action.choices.append(518)
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = None
    try:
        if not args.silent:
            print(f'attempting to load {weights_path} as a AsymmetricMASt3R')
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
        if not args.silent:
            print(f'... Success.')
    except Exception as e:
        if not args.silent:
            print(f'... Failed.')

    if model is None:
        try:
            if not args.silent:
                print(f'attempting to load {weights_path} as a AsymmetricCroCo3DStereo')
            model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
            if not args.silent:
                print(f'... Success.')
        except Exception as e:
            if not args.silent:
                print(f'... Failed.')

    if model is None:
        try:
            if not args.silent:
                print(f'attempting to load {weights_path} as a AsymmetricMASt3RWithDUNEBackbone')
            model = load_dune_mast3r_model(weights_path, args.device)
            if not args.silent:
                print(f'... Success.')
        except Exception as e:
            if not args.silent:
                print(f'... Failed.')

    if model is None:
        raise Exception(f"Could not load {weights_path}.")

    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)
