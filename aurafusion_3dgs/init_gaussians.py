#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 5 (GS env): lift the inpainted reference into new 3DGS Gaussians.

As the official ( ``inpaint_init`` + ``GaussianModel.inpaint_setup``): every pixel
of the reference's processed unseen mask, unprojected with the AGDD-aligned depth, becomes one Gaussian
with the reference colour (SH DC, higher bands zero), isotropic scale sqrt(mean squared distance to its 3
nearest neighbours), random rotation and opacity 0.99. The removed scene comes first, unchanged.

Writes <render_dir>/init/{point_cloud.ply, meta.json (num_fixed), rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from aurafusion_3dgs.cameras import ViewCamera, load_cameras, save_png, unproject
from aurafusion_3dgs.gaussians import SH_C0, Gaussians, load_ply, render_view, save_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--initial_opacity", type=float, default=0.99)
    args = parser.parse_args()

    root, r = args.render_dir, args.reference_index
    out = root / "init"
    cameras = load_cameras(root / "cameras.json")
    camera: ViewCamera = cameras[r]
    mask = torch.from_numpy(np.asarray(Image.open(root / "unseen_dilated" / f"{r:05d}.png")) > 127).cuda()
    depth = torch.from_numpy(np.load(root / "reference" / "depth_aligned.npy")).cuda()
    reference = torch.from_numpy(np.asarray(Image.open(root / "reference" / "reference.png").convert("RGB"))).cuda()
    points = unproject(camera, depth, mask)
    colors = reference[mask].float() / 255.0
    print(f"unprojected {points.shape[0]} unseen pixels of reference view {r}")

    removed = load_ply(root / "removed.ply")
    distances, _ = cKDTree(points.cpu().numpy()).query(points.cpu().numpy(), k=4)
    scale = torch.from_numpy(np.sqrt((distances[:, 1:] ** 2).mean(1)).clip(1e-7)).float().cuda()
    n = points.shape[0]
    new = Gaussians(
        xyz=points,
        features_dc=((colors - 0.5) / SH_C0)[:, None],
        features_rest=torch.zeros((n, removed.features_rest.shape[1], 3), device="cuda"),
        opacity=torch.logit(torch.full((n, 1), args.initial_opacity, device="cuda")),
        scaling=torch.log(scale)[:, None].repeat(1, 3),
        rotation=torch.rand((n, 4), device="cuda"),
    )
    model = Gaussians.cat(removed, new)
    out.mkdir(parents=True, exist_ok=True)
    save_ply(model, out / "point_cloud.ply")
    (out / "meta.json").write_text(json.dumps({"num_fixed": len(removed), "num_new": n, "reference_index": r}))
    print(f"model: {len(removed)} fixed + {n} new Gaussians -> {out / 'point_cloud.ply'}")
    for c in cameras:
        save_png(render_view(model, c)["rgb"], out / "rgb" / f"{c.index:05d}.png")
    print(f"Done: rendered {len(cameras)} views -> {out / 'rgb'}")


if __name__ == "__main__":
    main()
