<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AuraFusion-3DGS (Aurafusion-modules)

[AuraFusion360](https://github.com/kkennethwu/AuraFusion360_official) (Wu et al., CVPR 2025) object removal and
inpainting for any method that writes the standard 3D Gaussian Splatting PLY (3DGS, FastGS, Taming-3DGS, …).

The official code is built on 2DGS. This repository reimplements its 3D parts for plain 3DGS, rendered with
graphdeco's `diff_gaussian_rasterization`. The 2D models are the official ones: SAM2, Marigold, LaMa, and
LeftRefill on SD2-inpainting.

```
point_cloud.ply + COLMAP + 2D object masks ──► removed scene ──► unseen masks ──► inpainted reference + AGDD depth
                                                                                         │
                     inpainted point_cloud.ply ◄── finetune ◄── SDEdit targets ◄── new Gaussians
```

## Stages

| # | module | env | what it does |
|---|---|---|---|
| 0 | `flashsplat` | GS | Object Gaussians from 2D masks: FlashSplat's closed-form argmax on per-Gaussian alpha·T weights inside/outside the masks. Writes `object.npy`. |
| 1 | `render_views` | GS | Removal (object + background inside its trimmed convex hull). Per view: removed RGB / z-depth / opacity, object and deleted opacity. Writes `cameras.json` and `removed.ply`. |
| 2 | `unseen_contour` | GS | Paper Eq. 1–2. Removal-region pixels are lifted with the removed depth; a pixel stays if it is hidden in > 60% of the views. |
| 3 | `sam2_unseen` | 2D | Eq. 3. SAM2 box-prompted by each contour, then opened, reduced to its largest component and dilated. |
| 4 | `agdd` | 2D | Builds the reference (photo + removed render + LaMa, or `REF_IMAGE`), then Adaptive Guided Depth Diffusion (Marigold v1-0). |
| 5 | `init_gaussians` | GS | One Gaussian per unseen reference pixel, at the AGDD depth. |
| 6 | `sdedit` | 2D | LeftRefill SDEdit of every view, conditioned on the reference. Calls the official `sdedit_utils.py`. |
| 7 | `finetune` | GS | 10k iterations on the new Gaussians only (Eq. 12). Writes `final/point_cloud.ply` and `final/rgb/`. |

## Run

```bash
PLY=.../point_cloud/iteration_30000/point_cloud.ply COLMAP=.../scene MASK_DIR=.../masks OUT=output/myscene \
GS_PY=... AF_PY=... AF360_REPO=.../AuraFusion360_official CKPT_DIR=checkpoints \
bash scripts/run.sh
```

- `scripts/run.sh` lists every variable.
- Variables you will often set:
  - `IMAGES=images_4` when the 3DGS was trained on downsampled images;
  - `OBJECT_MASK=<bool .npy>` to skip FlashSplat;
  - `STEPS=render+contour` to run some stages only;
  - `REF_IMAGE` + `REF_INDEX` to use your own inpainted reference.
- `scripts/leonardo/run.sbatch` is the wrapper used on CINECA Leonardo.

Masks: one image per COLMAP view, named `<image stem>.<any extension>`, with the object > 127.

## Environments

- **GS env**:
  - Python with graphdeco's [`diff_gaussian_rasterization`](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
    (the `gaussian-splatting` submodule, any version that returns `(color, radii, invdepth)`),
  - plus `plyfile`, `scipy`, `fused-ssim`, `torchmetrics`, `pillow`.
  - A working gaussian-splatting / FastGS environment plus `pip install scipy torchmetrics` is enough.
- **2D env**:
  - `sam2`, `diffusers` (Marigold), `opencv-python`;
  - LeftRefill's vendored `ldm` deps from AuraFusion360_official: `pytorch_lightning`, `open_clip_torch`, CLIP,
    `transformers<5`, `omegaconf`, `einops`.
  - Keep it separate from the GS env.
- **Checkpoints** (`CKPT_DIR`):
  - `big-lama.pt` (TorchScript LaMa);
  - `marigold-v1-0/` (`prs-eth/marigold-v1-0`);
  - `sd2-inpainting/512-inpainting-ema.ckpt` (Stable Diffusion 2 inpainting).
- SAM2 comes from `SAM2_MODEL` (default `facebook/sam2-hiera-large`). OpenCLIP ViT-H weights come from the HF cache.

## Differences from the official code

- **3DGS instead of 2DGS:**
  - depth is the alpha-weighted mean of the Gaussian centres' camera z;
  - new Gaussians are isotropic 3D Gaussians, not surfels;
  - there are no normal or distortion regularisers.
- **Cameras:** the principal point is forced to the image centre, as the graphdeco rasteriser and loader assume.
- **Removal:**
  - FlashSplat labels plus the object's convex hull, instead of the official learned per-Gaussian mask;
  - the removal region R is object opacity > 0.02;
  - Gaussians that no training view sees are kept.
- **Unseen contour on partial captures:** when the official mean leaves the contour empty in every view, it is
  recomputed over only the views each point lands in.
- **Reference:**
  - built from the photo, the removed render and LaMa, or given with `REF_IMAGE`;
  - the official Other-360 configs load a pre-inpainted image;
  - AGDD normalises depth by its 0.99 quantile instead of the global max, which far floaters break.
- **AGDD optimiser:** torch AdamW with an fp32 master latent, instead of bitsandbytes AdamW8bit.
- **New Gaussians** start at opacity 0.99, because the official 1.0 is `inverse_sigmoid(1) = inf`.
- **Finetune:** official learning rates, no densification (the official densifies every 300 iterations until 1000).

The `--virtual_json` options of `unseen_contour` and `frame_views` belong to a virtual-camera variant that is not
included here.

## Credits

- AuraFusion360: Wu et al., CVPR 2025, Apache-2.0.
- FlashSplat: Shen et al., ECCV 2024.
- Marigold: Ke et al., CVPR 2024.
- LaMa: Suvorov et al., WACV 2022.
- SAM2: Ravi et al., 2024.

The code comes from the AuraFusion360 port in the authors' ArtiFixer fork (Apache-2.0, see `LICENSE`).
