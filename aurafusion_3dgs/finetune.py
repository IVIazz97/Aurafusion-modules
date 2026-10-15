#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 7 (GS env): finetune only the new Gaussians on the SDEdit targets.

The loss is ``losses.af360_loss`` (paper Eq. 12, as the official ``inpaint_finetune``); the
removed-scene Gaussians are constants, the new ones the only parameters. 10k iterations, one random view
each, with the official AuraFusion360 learning rates (position 1.6e-4 -> 1.6e-6 log-linear, x the graphdeco
scene extent; features 2.5e-3, rest /20; opacity 0.05; scaling 5e-3; rotation 1e-3). No densification
(the official densifies every 300 iterations until 1000), as the 3DGUT port.

``--in_frame_only``: the in-mask terms only on views whose unseen mask is inside the image
(``frame_views.in_frame``).

Writes <render_dir>/final/{point_cloud.ply, rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aurafusion_3dgs.losses import af360_loss
from aurafusion_3dgs.cameras import load_cameras, save_png
from aurafusion_3dgs.gaussians import Gaussians, load_ply, rasterize, render_view, save_ply, scene_extent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    parser.add_argument("--in_frame_only", action="store_true")
    parser.add_argument("--frame_margin", type=float, default=0.01)
    parser.add_argument("--summary", type=Path, default=None, help="Append a one-line human-readable summary here.")
    args = parser.parse_args()
    root = args.render_dir
    out = root / "final"

    cameras = load_cameras(root / "cameras.json")
    model = load_ply(root / "init" / "point_cloud.ply")
    num_fixed = json.loads((root / "init" / "meta.json").read_text())["num_fixed"]
    fixed = model.subset(slice(0, num_fixed))
    new = model.subset(slice(num_fixed, None))
    print(f"finetuning {len(new)} new Gaussians, {num_fixed} frozen")
    extent = scene_extent(cameras)
    lrs = {"xyz": 1.6e-4 * extent, "features_dc": 2.5e-3, "features_rest": 2.5e-3 / 20, "opacity": 0.05,
           "scaling": 5e-3, "rotation": 1e-3}
    params = {name: torch.nn.Parameter(getattr(new, name).clone()) for name in lrs}
    optimizer = torch.optim.Adam([{"params": [p], "lr": lrs[n], "name": n} for n, p in params.items()], eps=1e-15)
    position_lr_final = 1.6e-6 * extent

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda().eval()
    for p in lpips.parameters():
        p.requires_grad_(False)

    missing = [c.index for c in cameras if not (root / "sdedit" / f"{c.index:05d}.png").exists()]
    if missing:
        raise SystemExit(f"finetune: {len(missing)} of {len(cameras)} SDEdit targets missing in {root / 'sdedit'} "
                         f"(e.g. {missing[:5]}); rerun sdedit")
    targets = [torch.from_numpy(np.asarray(Image.open(root / "sdedit" / f"{c.index:05d}.png").convert("RGB"))).cuda()
               for c in cameras]
    masks = [torch.from_numpy(np.asarray(Image.open(root / "unseen_dilated" / f"{c.index:05d}.png")) > 127).cuda()
             for c in cameras]
    supervise_hole = [True] * len(cameras)
    if args.in_frame_only:
        from aurafusion_3dgs.frame_views import in_frame

        supervise_hole = [in_frame(m.cpu().numpy(), args.frame_margin) for m in masks]
        print(f"in-mask loss on {sum(supervise_hole)}/{len(cameras)} views: {[i for i, s in enumerate(supervise_hole) if s]}")
        if not any(supervise_hole):
            print("WARNING FALLBACK: no view frames the whole unseen mask; the in-mask loss goes back to every view")
            supervise_hole = [True] * len(cameras)

    generator = torch.Generator().manual_seed(0)
    order: list[int] = []
    for iteration in range(1, args.iterations + 1):
        if not order:
            order = torch.randperm(len(cameras), generator=generator).tolist()
        index = order.pop()
        t = iteration / args.iterations  # 3DGS log-linear position lr decay
        optimizer.param_groups[0]["lr"] = float(np.exp((1 - t) * np.log(lrs["xyz"]) + t * np.log(position_lr_final)))

        g = Gaussians.cat(fixed, Gaussians(**params))
        image = rasterize(g, cameras[index]).clamp(0, 1).permute(1, 2, 0)
        loss = af360_loss(image, targets[index].float() / 255.0, masks[index], supervise_hole[index], lpips,
                          args.lambda_dssim, args.lambda_lpips)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 1000 == 0 or iteration == 1:
            print(f"iter {iteration}/{args.iterations}: loss {loss.item():.4f}", flush=True)

    final = Gaussians.cat(fixed, Gaussians(**params)).detach()
    save_ply(final, out / "point_cloud.ply")
    for c in cameras:
        save_png(render_view(final, c)["rgb"], out / "rgb" / f"{c.index:05d}.png")
    print(f"final renders: {out / 'rgb'} ({len(cameras)} views)")
    if args.summary is not None:
        with open(args.summary, "a") as f:
            f.write(f"af360-3dgs finetune: {args.iterations} iters, in-hole loss on {sum(supervise_hole)}/{len(cameras)} "
                    f"views, last loss {loss.item():.4f}, {len(new)} new Gaussians -> {out}\n")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
