#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AuraFusion360 on a graphdeco 3DGS, all stages. Every setting is an environment variable:
#
# Required
#   PLY          full-scene point_cloud.ply (object present)
#   COLMAP       scene dir with sparse/(0/) and the training images
#   MASK_DIR     2D object masks, one per view named <image stem>.*  (or OBJECT_MASK, a bool .npy over the PLY)
#   OUT          output dir
#   GS_PY        python with diff_gaussian_rasterization (graphdeco), plyfile, scipy, fused_ssim, torchmetrics
#   AF_PY        python with sam2, diffusers (Marigold), and the LeftRefill deps of AuraFusion360_official
#   AF360_REPO   clone of https://github.com/kkennethwu/AuraFusion360_official (sdedit imports its LeftRefill)
#   CKPT_DIR     holds big-lama.pt, marigold-v1-0/, sd2-inpainting/512-inpainting-ema.ckpt
# Optional
#   STEPS        subset of flashsplat,render,contour,sam2,agdd,init,sdedit,finetune (a+b+c also works)
#   IMAGES       image folder in COLMAP the 3DGS trained on (default images)
#   VARIANT      inpainting output subdir (default lama_ref); stages 0-3 are shared through symlinks
#   REF_IMAGE    an inpainted reference image instead of LaMa; REF_INDEX its view index (default 0)
#   HULL_EXPAND  grow the object hull (default 0); GAMMA FlashSplat background bias (default -0.9)
#   SAM2_MODEL   HF id or local dir (default facebook/sam2-hiera-large)
#   FT_ITERS     finetune iterations (default 10000)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
STEPS=${STEPS:-flashsplat,render,contour,sam2,agdd,init,sdedit,finetune}; STEPS=${STEPS//+/,}
IMAGES=${IMAGES:-images}; VARIANT=${VARIANT:-lama_ref}; REF_INDEX=${REF_INDEX:-0}
HULL_EXPAND=${HULL_EXPAND:-0}; GAMMA=${GAMMA:--0.9}; FT_ITERS=${FT_ITERS:-10000}
SAM2_MODEL=${SAM2_MODEL:-facebook/sam2-hiera-large}
OUT=$(realpath -m "$OUT")   # absolute: the variant dir symlinks into base
BASE="$OUT/base"; R="$OUT/$VARIANT"
OBJECT_MASK=${OBJECT_MASK:-$BASE/flashsplat/object.npy}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
echo "ply=$PLY colmap=$COLMAP out=$OUT steps=$STEPS"

has flashsplat && "$GS_PY" -m aurafusion_3dgs.flashsplat --ply "$PLY" --colmap_dir "$COLMAP" --images "$IMAGES" \
                     --mask_dir "$MASK_DIR" --gamma "$GAMMA" --output "$OBJECT_MASK"
has render  && "$GS_PY" -m aurafusion_3dgs.render_views --ply "$PLY" --colmap_dir "$COLMAP" --images "$IMAGES" \
                 --object_mask "$OBJECT_MASK" --hull_expand "$HULL_EXPAND" --output_dir "$BASE"
has contour && "$GS_PY" -m aurafusion_3dgs.unseen_contour --render_dir "$BASE"
has sam2    && "$AF_PY" -m aurafusion_3dgs.sam2_unseen --render_dir "$BASE" --sam2_model "$SAM2_MODEL"
mkdir -p "$R"
for f in cameras.json removed.ply removed object deleted unseen_contour unseen unseen_dilated; do
    [ -e "$R/$f" ] || ln -s "$BASE/$f" "$R/$f"
done
has agdd    && "$AF_PY" -m aurafusion_3dgs.agdd --render_dir "$R" --colmap_dir "$COLMAP" --reference_index "$REF_INDEX" \
                 --lama_model "$CKPT_DIR/big-lama.pt" --marigold "$CKPT_DIR/marigold-v1-0" \
                 ${REF_IMAGE:+--reference_image "$REF_IMAGE"}
has init    && "$GS_PY" -m aurafusion_3dgs.init_gaussians --render_dir "$R" --reference_index "$REF_INDEX"
has sdedit  && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$AF_PY" -m aurafusion_3dgs.sdedit --render_dir "$R" \
                 --reference_index "$REF_INDEX" --official_repo "$AF360_REPO" \
                 --sd2_inpainting_ckpt "$CKPT_DIR/sd2-inpainting/512-inpainting-ema.ckpt"
has finetune && "$GS_PY" -m aurafusion_3dgs.finetune --render_dir "$R" --iterations "$FT_ITERS"
echo "AURAFUSION_3DGS_DONE: $R/final/point_cloud.ply"
