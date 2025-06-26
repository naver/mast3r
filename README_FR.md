![bannière](assets/mast3r.jpg)

# Implémentation officielle de `Grounding Image Matching in 3D with MASt3R`  
[[Page du projet](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)], [[MASt3R arxiv](https://arxiv.org/abs/2406.09756)], [[DUSt3R arxiv](https://arxiv.org/abs/2312.14132)]  

![Exemple de résultats de correspondance obtenus avec MASt3R](assets/examples.jpg)

![Aperçu du modèle d'architecture de MASt3R](assets/mast3r_archi.jpg)

```bibtex
@misc{mast3r_arxiv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon et Jerome Revaud},
      year={2024},
      eprint={2406.09756},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang et Vincent Leroy et Yohann Cabon et Boris Chidlovskii et Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}
```

## Table des matières

- [Table des matières](#table-des-matières)
- [Licence](#licence)
- [Commencer](#commencer)
  - [Installation](#installation)
  - [Points de contrôle](#points-de-contrôle)
  - [Démo interactive](#démo-interactive)
  - [Démo interactive avec docker](#démo-interactive-avec-docker)
- [Utilisation](#utilisation)
- [Entraînement](#entraînement)
  - [Jeux de données](#jeux-de-données)
  - [Démo](#démo)
  - [Nos hyperparamètres](#nos-hyperparamètres)
- [Localisation visuelle](#localisation-visuelle)
  - [Préparation du jeu de données](#préparation-du-jeu-de-données)
  - [Exemples de commandes](#exemples-de-commandes)

## Licence

Le code est distribué sous la licence CC BY-NC-SA 4.0.  
Voir [LICENCE](LICENCE) pour plus d'informations.

```python
# Copyright (C) 2024-present Naver Corporation. Tous droits réservés.
# Sous licence CC BY-NC-SA 4.0 (usage non commercial uniquement).
```

## Commencer

### Installation

1. Clonez MASt3R.
```bash
git clone --recursive https://github.com/naver/mast3r
cd mast3r
# si vous avez déjà cloné mast3r :
# git submodule update --init --recursive
```
2. Créez l'environnement, voici un exemple utilisant conda.
  ```bash
conda create -n mast3r python=3.11 cmake=3.14.0
conda activate mast3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # utilisez la version correcte de cuda pour votre système
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optionnel : vous pouvez également installer des paquets supplémentaires pour :
# - ajouter la prise en charge des images HEIC
# - ajouter les paquets nécessaires pour visloc.py
pip install -r dust3r/requirements_optional.txt
```
3. Optionnel, compilez les noyaux cuda pour RoPE (comme dans CroCo v2).
```bash
# DUST3R repose sur les embeddings positionnels RoPE pour lesquels vous pouvez compiler des noyaux cuda pour un temps d'exécution plus rapide.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```
 ### Points de contrôle

Vous pouvez obtenir les points de contrôle de deux manières :

1) Vous pouvez utiliser notre intégration `huggingface_hub` : les modèles seront téléchargés automatiquement.

2) Sinon, nous fournissons plusieurs modèles pré-entraînés :

| Nom du modèle   | Résolutions d'entraînement | Tête      | Encodeur | Décodeur |
|------------------|----------------------------|-----------|----------|----------|
| [`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | CatMLP+DPT | ViT-L    | ViT-B    |

Vous pouvez consulter les hyperparamètres que nous avons utilisés pour entraîner ces modèles dans la [section : Nos hyperparamètres](#nos-hyperparamètres).  
Assurez-vous de vérifier la licence des jeux de données que nous avons utilisés.

Pour télécharger un modèle spécifique, par exemple `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` :
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```
> Pour ces points de contrôle, assurez-vous d'accepter la licence de tous les jeux de données d'entraînement que nous avons utilisés, en plus de CC-BY-NC-SA 4.0.
La licence du jeu de données mapfree, en particulier, est très restrictive. Pour plus d'informations, consultez [CHECKPOINTS_NOTICE](CHECKPOINTS_NOTICE).

### Démonstration interactive

Nous avons créé un espace huggingface exécutant le nouvel alignement global épars dans une démonstration simplifiée pour de petites scènes : [naver/MASt3R](https://huggingface.co/spaces/naver/MASt3R).

Il y a deux démonstrations disponibles pour une exécution locale :

```bash
demo.py est la démonstration mise à jour pour MASt3R. Elle utilise notre nouvelle méthode d'alignement global épars qui vous permet de reconstruire des scènes plus grandes.

python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

# Utilisez --weights pour charger un point de contrôle à partir d'un fichier local, par exemple --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# Utilisez --local_network pour le rendre accessible sur le réseau local, ou --server_name pour spécifier l'URL manuellement
# Utilisez --server_port pour changer le port, par défaut il cherchera un port disponible à partir de 7860
# Utilisez --device pour utiliser un appareil différent, par défaut c'est "cuda"

demo_dust3r_ga.py est la même démonstration que dans dust3r (+ compatibilité pour les modèles MASt3R).
voir https://github.com/naver/dust3r?tab=readme-ov-file#interactive-demo pour plus de détails.
```

# MASt3R Interactive Demo with Docker

## Overview

To run MASt3R using Docker, including support for NVIDIA CUDA, follow these instructions:

## Installation Steps

1. **Install Docker**: If not already installed, download and install Docker and Docker Compose from the [Docker website](https://www.docker.com/get-started).

2. **Install NVIDIA Docker Toolkit**: For GPU support, install the NVIDIA Docker toolkit from the [NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. **Build and Run the Docker Image**:
   - Navigate to the `./docker` directory and run the following commands:

   For CUDA support:

```bash 
   
   cd docker
   bash run.sh --with-cuda --model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
```

By default, `demo.py` is launched with the option `--local_network`.  
Visit `http://localhost:7860/` to access the web UI (or replace `localhost` with the machine's name to access it from the network).  

`run.sh` will launch docker-compose using either the [docker-compose-cuda.yml](docker/docker-compose-cuda.yml) or [docker-compose-cpu.yml](docker/docker-compose-cpu.yml) config file, then it starts the demo using [entrypoint.sh](docker/files/entrypoint.sh).

___

![demo](assets/demo.jpg)

## Usage

```python
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)
```

![exemple de correspondance sur croco pair](assets/matching.jpg)

## Entraînement

Dans cette section, nous présentons une courte démonstration pour commencer à entraîner MASt3R.

### Jeux de données

Voir [la section Jeux de données dans DUSt3R](https://github.com/naver/dust3r?tab=readme-ov-file#datasets)

### Démo

Comme pour la démo d'entraînement DUSt3R, nous allons télécharger et préparer le même sous-ensemble de [CO3Dv2](https://github.com/facebookresearch/co3d) - [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/co3d/blob/main/LICENSE) et lancer le code d'entraînement dessus. C'est exactement le même processus que DUSt3R. 
Le modèle de démonstration sera entraîné pendant quelques époques sur un très petit jeu de données. Il ne sera pas très bon.

```bash
# télécharger et préparer le sous-ensemble co3d
mkdir -p data/co3d_subset
cd data/co3d_subset
git clone https://github.com/facebookresearch/co3d
cd co3d
python3 ./co3d/download_dataset.py --download_folder ../ --single_sequence_subset
rm ../*.zip
cd ../../..

python3 datasets_preprocess/preprocess_co3d.py --co3d_dir data/co3d_subset --output_dir data/co3d_subset_processed --single_sequence_subset

# télécharger le point de contrôle pré-entraîné dust3r
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

# pour cet exemple, nous ferons moins d'époques, pour les véritables hyperparamètres que nous avons utilisés dans l'article, voir la section suivante : "Nos Hyperparamètres"
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=(512,384), n_corres=1024, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/mast3r_demo"
```

## Nos Hyperparamètres
```
# MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric - entraîner mast3r avec régression métrique et perte de correspondance
# nous avons utilisé cosxl pour générer des variations de DL3DV : "brumeux", "nuit", "pluie", "neige", "ensoleillé" mais nous n'étions pas convaincus par cela.

torchrun --nproc_per_node=8 train.py \
    --train_dataset "57_000 @ Habitat512(1_000_000, split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ TartanAir(pairs_subset='', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 4_560 @ UnrealStereo4K(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 1_140 @ VirtualKitti(optical_center_is_centered=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRgbd(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 145_920 @ NianticMapFree(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "100 @ Habitat512(1_000_000, split='val', resolution=(512,384), n_corres=1024, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/mast3r_demo"

```

## Localisation Visuelle

### Préparation du Dataset

Voir la [section Visloc dans DUSt3R](https://github.com/naver/dust3r/blob/main/dust3r_visloc/README.md#dataset-preparation)

### Exemples de Commandes

Avec `visloc.py`, vous pouvez exécuter nos expériences de localisation visuelle sur Aachen-Day-Night, InLoc, Cambridge Landmarks et 7 Scenes.

```bash
# Aachen-Day-Night-v1.1:
# scène en 'jour' 'nuit'
# la scène peut aussi être 'toutes'
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/Aachen-Day-Night-v1.1/', subscene='${scene}', pairsfile='fire_top50', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/Aachen-Day-Night-v1.1/${scene}/loc

# ou avec un passage grossier à fin:

python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/Aachen-Day-Night-v1.1/', subscene='${scene}', pairsfile='fire_top50', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/Aachen-Day-Night-v1.1/${scene}/loc --coarse_to_fine --max_batch_size 48 --c2f_crop_with_homography

# InLoc
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocInLoc('/path/to/prepared/InLoc/', pairsfile='pairs-query-netvlad40-temporal', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/InLoc/loc

# ou avec un passage grossier à fin:

python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocInLoc('/path/to/prepared/InLoc/', pairsfile='pairs-query-netvlad40-temporal', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/InLoc/loc --coarse_to_fine --max_image_size 1200 --max_batch_size 48 --c2f_crop_with_homography

# 7-scenes:
# scène en 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes/', subscene='${scene}', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7-scenes/${scene}/loc

# Cambridge Landmarks:
# scène en 'ShopFacade' 'GreatCourt' 'KingsCollege' 'OldHospital' 'StMarysChurch'
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocCambridgeLandmarks('/path/to/prepared/Cambridge_Landmarks/', subscene='${scene}', pairsfile='APGeM-LM18_top50', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/Cambridge_Landmarks/${scene}/loc
```
