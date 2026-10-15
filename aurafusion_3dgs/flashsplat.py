#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 0 (GS env): which Gaussians are the object, from 2D object masks.

FlashSplat (Shen et al., ECCV 2024) in one pass, no training: a rendered mask is linear in per-Gaussian
labels with coefficient = the Gaussian's alpha * T weight, so per Gaussian sum that weight over the mask
pixels (in) and the other pixels (out) of every view, then ``multi_instance_opt``'s (the official FlashSplat function)
argmax on the L2-normalised (out, in) pair with the background bias ``gamma`` (< 0 grows the object;
-0.9 = the median of the official removal configs, as SurfaceLift's gs3d removal). Unlike the official
solver, a Gaussian no view sees (in = out = 0) stays background: it cannot be the object in any photo,
and dropping it would only open holes in novel views.

Masks: one image per COLMAP view, ``--mask_dir/<image stem>.*``, object = value > ``--mask_threshold``,
resized (nearest) to the render size; views without a mask count as all-background.

Writes --output: a bool .npy over the Gaussians of --ply (True = object) + <output>.json (counts).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

from aurafusion_3dgs.gaussians import blend_weights, load_colmap_cameras, load_ply


def multi_instance_opt(all_contrib: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Assign each Gaussian to each object.

    Args:
        all_contrib: ``[num_objects + 1, num_gaussians]`` accumulated alpha-blending weight,
            row 0 being background. This is what the instrumented rasterizer produces.
        gamma: background bias in ``[-1, 1]``. Larger values pull Gaussians towards
            background, giving tighter (more conservative) objects; negative values
            grow them. FlashSplat calls this the slackness.

    Returns:
        Boolean ``[num_objects + 1, num_gaussians]``, where ``[i, j]`` is True when
        Gaussian *j* belongs to object *i*. Rows are solved independently, so a Gaussian
        may belong to several objects where masks genuinely overlap.
    """
    if all_contrib.ndim != 2:
        raise ValueError(f"all_contrib must be [num_objects + 1, num_gaussians], got {tuple(all_contrib.shape)}")

    all_contrib = all_contrib.float()
    total_contrib = all_contrib.sum(dim=0)

    labels = torch.zeros_like(all_contrib, dtype=torch.bool)
    for obj_idx, obj_contrib in enumerate(all_contrib):
        if obj_contrib.sum() == 0:  # object never seen; leave its row empty
            continue
        # Row 0 is "everything but this object", row 1 is "this object".
        pair = torch.stack([total_contrib - obj_contrib, obj_contrib], dim=0)
        pair = F.normalize(pair, dim=0)
        pair[0] += gamma
        labels[obj_idx] = pair.argmax(dim=0).bool()
    return labels


def load_masks(mask_dir: Path, cameras, threshold: int, dilate: int) -> list[torch.Tensor]:
    masks, missing = [], 0
    for camera in cameras:
        found = sorted(mask_dir.glob(Path(camera.name).stem + ".*"))
        if not found:
            masks.append(torch.zeros(camera.height, camera.width, dtype=torch.bool))
            missing += 1
            continue
        m = Image.open(found[0]).convert("L").resize((camera.width, camera.height), Image.NEAREST)
        m = np.asarray(m) > threshold
        if dilate:
            m = binary_dilation(m, iterations=dilate)
        masks.append(torch.from_numpy(m))
    if missing:
        print(f"WARNING: {missing}/{len(cameras)} views have no mask in {mask_dir}; they count as all-background")
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, required=True, help="graphdeco point_cloud.ply of the full scene.")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--images", default="images", help="Image folder in --colmap_dir the 3DGS was trained on.")
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--mask_dilate", type=int, default=0, help="Pixels (official FlashSplat: none).")
    parser.add_argument("--gamma", type=float, default=-0.9)
    parser.add_argument("--output", type=Path, required=True, help="object.npy")
    args = parser.parse_args()

    start = time.time()
    g = load_ply(args.ply)
    cameras = load_colmap_cameras(args.colmap_dir, args.images)
    masks = load_masks(args.mask_dir, cameras, args.mask_threshold, args.mask_dilate)
    out_w, in_w = blend_weights(g, cameras, masks)
    seen = (in_w + out_w) > 0
    obj = multi_instance_opt(torch.stack([out_w, in_w]), args.gamma)[1] & seen
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, obj.cpu().numpy())
    stats = {"gaussians": len(g), "object": int(obj.sum()), "unseen": int((~seen).sum()), "gamma": args.gamma,
             "views": len(cameras), "object_weight_in_mask_frac": float(in_w[obj].sum() / (in_w[obj] + out_w[obj]).sum()),
             "mask_weight_on_object_frac": float(in_w[obj].sum() / in_w.sum()), "seconds": round(time.time() - start, 1)}
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
