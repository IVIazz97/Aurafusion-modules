#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 2: depth-aware unseen contour (paper Eq. 1-2).

For view n, every pixel of its removal region R_n is lifted to 3D with the removed-scene depth
and projected into every other view i in which the object is visible. The pixel counts as
hidden in view i if it lands inside R_i. C_n = [mean_i hidden > theta] & R_n, theta = 0.6.

Same as the official ``GaussianExtractor.depth_aware_unseen_mask_generation`` with
``removal_region=depth_diff``: R is where the object changes the render (here: object opacity
above ``--region_threshold``), no dilation; projections outside the image count as seen.
One addition: points behind a camera count as seen (the official warp does not test z).

``--denominator in_view`` averages only over the views the point lands in, and a point that no
other view frames counts as unseen (nothing can show it). For a capture that does not go all
around the object the official mean never reaches theta, because every view that looks away
counts as having seen the point. The default ``auto`` keeps the official result unless it is
empty in every view, and only then uses ``in_view``; the choice goes to unseen_contour.json.

``--virtual_json`` (af360v, from ``virtual_views``): only the real views vote on whether a point is
seen. The virtual views are renders, not observations; they still get a contour of their own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aurafusion_3dgs.cameras import load_cameras, project, save_png, unproject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True, help="render_views.py output.")
    parser.add_argument("--region_threshold", type=float, default=0.02, help="Object opacity defining R_i.")
    parser.add_argument("--aggregate_threshold", type=float, default=0.6, help="theta in Eq. 2.")
    parser.add_argument("--denominator", choices=("all", "in_view", "auto"), default="auto",
                        help="all: official mean over every view showing the object; in_view: mean over the views "
                             "the point lands in; auto: all, or in_view when all is empty in every view.")
    parser.add_argument("--virtual_json", type=Path, default=None,
                        help="virtual.json of virtual_views: its virtual views receive a contour but never vote.")
    parser.add_argument("--chunk", type=int, default=65536)
    args = parser.parse_args()

    root = args.render_dir
    cameras = load_cameras(root / "cameras.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions = torch.stack([
        torch.from_numpy(np.asarray(Image.open(root / "object" / "opacity" / f"{c.index:05d}.png"), dtype=np.float32) / 255.0)
        > args.region_threshold
        for c in cameras
    ]).to(device)  # [J, H, W]
    J, H, W = regions.shape
    has_object = regions.flatten(1).any(1)
    voters = has_object.clone()
    if args.virtual_json is not None:
        num_real = json.loads(args.virtual_json.read_text())["num_real"]
        if not 0 < num_real <= J:
            raise SystemExit(f"unseen_contour: {args.virtual_json} says {num_real} real views but {root} has {J} views")
        voters[num_real:] = False
        print(f"{int(voters.sum())} real views showing the object vote; {J - num_real} virtual views get a contour only")

    def unseen_contours(denominator: str) -> tuple[list[torch.Tensor], list[float]]:
        contours, fractions = [], []
        for n, camera in enumerate(cameras):
            region = regions[n]
            contour = torch.zeros_like(region)
            if region.any():
                depth = torch.from_numpy(np.load(root / "removed" / "depth" / f"{n:05d}.npy")).to(device)
                points = unproject(camera, depth, region)
                others = [j for j in range(J) if j != n and bool(voters[j])]
                hidden = torch.zeros(points.shape[0], device=device)
                framed = torch.zeros_like(hidden)
                for s in range(0, points.shape[0], args.chunk):
                    u, v, z = project([cameras[j] for j in others], points[s:s + args.chunk])
                    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
                    flat = torch.as_tensor(others, device=device)[:, None] * (H * W) + v.clamp(0, H - 1) * W + u.clamp(0, W - 1)
                    hidden[s:s + args.chunk] = (regions.flatten()[flat] & inside).float().sum(0)
                    framed[s:s + args.chunk] = inside.float().sum(0)
                if denominator == "all":
                    unseen = hidden / max(len(others), 1) > args.aggregate_threshold
                else:
                    unseen = (framed == 0) | (hidden / framed.clamp_min(1.0) > args.aggregate_threshold)
                contour[region] = unseen
                fractions.append(float(contour.sum()) / float(region.sum()))
            contours.append(contour.cpu())
            if n % 50 == 0:
                print(f"{denominator} view {n}: |R|={int(region.sum())} |C|={int(contour.sum())}", flush=True)
        return contours, fractions

    denominator = "in_view" if args.denominator == "in_view" else "all"
    contours, fractions = unseen_contours(denominator)
    if args.denominator == "auto" and not any(bool(c.any()) for c in contours):
        print("the official contour is empty in every view (views that look away count as having seen the "
              "point): recomputing with --denominator in_view", flush=True)
        denominator = "in_view"
        contours, fractions = unseen_contours(denominator)
    for n, contour in enumerate(contours):
        save_png(contour, root / "unseen_contour" / f"{n:05d}.png")
    empty = sum(not bool(c.any()) for c in contours)
    (root / "unseen_contour.json").write_text(json.dumps(
        {"denominator": denominator, "aggregate_threshold": args.aggregate_threshold, "views": J, "empty_views": empty}))
    print(f"Done: {J} views ({empty} with an empty contour), denominator {denominator}, contour/region area ratio "
          f"mean={np.mean(fractions):.3f} min={np.min(fractions):.3f} max={np.max(fractions):.3f}")


if __name__ == "__main__":
    main()
