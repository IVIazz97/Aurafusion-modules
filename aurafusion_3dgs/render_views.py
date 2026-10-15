#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 1 (GS env): object removal and per-view renders of a graphdeco 3DGS.

Same removal rule as the 3DGUT port in ArtiFixer (``aurafusion/render_views.py``): the object is the
Gaussians of ``--object_mask`` (``aurafusion_3dgs.flashsplat`` output, or any bool .npy over the PLY) that
some view sees (total alpha * T weight > ``--min_weight``), plus every other Gaussian inside the object's
convex hull (trimmed at the 99.5th distance percentile, optionally grown by ``--hull_expand``).
Gaussians no view sees are kept (the 3DGUT port drops them with its visibility filter).

Writes the render_dir that the 2D stages (unseen_contour,
sam2_unseen, agdd, sdedit) read:
  removed/rgb/<i>.png, removed/depth/<i>.npy, removed/opacity/<i>.png, object/opacity/<i>.png,
  deleted/opacity/<i>.png, cameras.json, and removed.ply (the fixed background, graphdeco PLY).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

from aurafusion_3dgs.cameras import project, save_cameras, save_png, unproject
from aurafusion_3dgs.gaussians import blend_weights, load_colmap_cameras, load_ply, render_view, save_ply


def inside_hull(points: np.ndarray, hull_points: np.ndarray, trim_percentile: float, expand: float) -> np.ndarray:
    """The 3DGUT port.s hull test: which ``points`` lie inside the convex hull
    of ``hull_points`` trimmed at ``trim_percentile`` of their distance to the centroid, every plane pushed
    out by ``expand`` x their mean radius."""
    if trim_percentile < 100:
        dist = np.linalg.norm(hull_points - hull_points.mean(0), axis=1)
        hull_points = hull_points[dist <= np.percentile(dist, trim_percentile)]
    if hull_points.shape[0] > 10_000:
        hull_points = hull_points[np.random.default_rng(0).choice(hull_points.shape[0], 10_000, replace=False)]
    try:
        hull = ConvexHull(hull_points)
    except QhullError as error:
        raise SystemExit(f"Could not compute object convex hull: {error}") from error
    tol = 1e-7 * max(float(np.ptp(hull_points, axis=0).max()), 1.0)
    margin = expand * float(np.linalg.norm(hull_points - hull_points.mean(0), axis=1).mean())
    inside = np.zeros(len(points), dtype=bool)
    for s in range(0, len(points), 65536):
        p = points[s:s + 65536]
        inside[s:s + 65536] = ((p @ hull.equations[:, :3].T + hull.equations[:, 3]) <= tol + margin).all(axis=1)
    return inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, required=True, help="graphdeco point_cloud.ply of the full scene.")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--images", default="images", help="Image folder in --colmap_dir the 3DGS was trained on.")
    parser.add_argument("--object_mask", type=Path, required=True, help="bool .npy over the PLY's Gaussians.")
    parser.add_argument("--min_weight", type=float, default=0.1,
                        help="Object Gaussians with less total alpha*T weight over all views are ignored.")
    parser.add_argument("--hull_expand", type=float, default=0.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    full = load_ply(args.ply)
    cameras = load_colmap_cameras(args.colmap_dir, args.images)
    labels = torch.from_numpy(np.load(args.object_mask)).to(full.xyz.device)
    assert labels.dtype == torch.bool and len(labels) == len(full), f"{args.object_mask} is not a bool mask over {args.ply}"
    weight = blend_weights(full, cameras)
    foreground = labels & (weight > args.min_weight)
    others = torch.where(~foreground)[0]
    inside = inside_hull(full.xyz[others].cpu().numpy(), full.xyz[foreground].cpu().numpy(), 99.5, args.hull_expand)
    deleted = torch.zeros_like(foreground)
    deleted[others[torch.from_numpy(inside).to(others.device)]] = True
    background = ~foreground & ~deleted
    print(f"{len(full)} Gaussians: object={int(foreground.sum())} (of {int(labels.sum())} labelled; the rest no view "
          f"sees), kept background={int(background.sum())}, background deleted inside the hull={int(deleted.sum())}")

    removed, obj, gone = full.subset(background), full.subset(foreground), full.subset(deleted)
    out.mkdir(parents=True, exist_ok=True)
    save_ply(removed, out / "removed.ply")
    (out / "removal.json").write_text(json.dumps({"ply": str(args.ply), "object_mask": str(args.object_mask),
                                                 "object": int(foreground.sum()), "deleted": int(deleted.sum()),
                                                 "kept": int(background.sum())}))
    for camera in cameras:
        index = camera.index
        r = render_view(removed, camera)
        save_png(r["rgb"], out / "removed" / "rgb" / f"{index:05d}.png")
        save_png(r["opacity"], out / "removed" / "opacity" / f"{index:05d}.png")
        save_png(render_view(obj, camera)["opacity"], out / "object" / "opacity" / f"{index:05d}.png")
        save_png(render_view(gone, camera)["opacity"] if len(gone) else torch.zeros_like(r["opacity"]),
                 out / "deleted" / "opacity" / f"{index:05d}.png")
        (out / "removed" / "depth").mkdir(parents=True, exist_ok=True)
        np.save(out / "removed" / "depth" / f"{index:05d}.npy", r["depth"].cpu().numpy().astype(np.float32))
        if index == 0:  # geometry self-check: unproject the removed render, reproject into view 1, compare colours
            valid = r["opacity"] > 0.95
            points = unproject(camera, r["depth"], valid)
            other = cameras[1]
            u, v, z = project([other], points)
            ok = (u[0] >= 0) & (u[0] < other.width) & (v[0] >= 0) & (v[0] < other.height) & (z[0] > 0)
            r1 = render_view(removed, other)
            d1 = r1["depth"][v[0][ok], u[0][ok]]
            front = (z[0][ok] - d1).abs() < 0.02 * d1  # points visible in view 1
            err = (r["rgb"][valid][ok][front] - r1["rgb"][v[0][ok][front], u[0][ok][front]]).abs().mean().item()
            print(f"reprojection self-check view 0 -> 1: {int(front.sum())} points, mean colour error {err:.4f}")
            assert front.sum() > 100 and err < 0.1, "camera convention mismatch between the rasteriser and unproject"
        if index % 50 == 0:
            print(f"rendered {index + 1}/{len(cameras)}", flush=True)
    save_cameras(cameras, out / "cameras.json")
    print(f"Done: {len(cameras)} views -> {out}")


if __name__ == "__main__":
    main()
