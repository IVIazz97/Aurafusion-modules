#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 3 (2D env): SAM2 unseen masks (paper Eq. 3) and their processed form.

As the official ``utils/sam2_utils.py``: the bounding box of each view's unseen contour is a box
prompt for SAM2VideoPredictor (facebook/sam2-hiera-large) on the removed-scene renders, then
masks are propagated through the sequence. Writes ``unseen/<i>.png``.

Then, as the official ``utils/camera_utils.py`` does at the inpaint stage, each mask is opened,
reduced to its largest connected component and dilated (kernel 5, 3 iterations for the
Other-360 configs). That processed mask ``unseen_dilated/<i>.png`` is what every later stage
uses: reference unprojection, SDEdit and finetuning.

Addition: the box is taken from the contour's largest connected component, not every contour
pixel, so a stray pixel from a floater Gaussian cannot inflate the prompt.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == (np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)


def process_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Official inpaint-stage mask processing: open, largest component, dilate."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.dilate(largest_component(opened).astype(np.uint8), kernel, iterations=iterations) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    parser.add_argument("--dilate_kernel", type=int, default=5)
    parser.add_argument("--dilate_iter", type=int, default=3)
    parser.add_argument("--union_deleted", action="store_true",
                        help="Add the surface the hull filter deleted (deleted/opacity) to the unseen mask. "
                             "With --hull_expand the removal takes real background, and the depth-aware test "
                             "cannot flag it: other views do see that deeper surface, so it is 'seen' -- but "
                             "nothing renders it any more, so it still has to be inpainted.")
    parser.add_argument("--deleted_threshold", type=float, default=0.5)
    args = parser.parse_args()

    from sam2.sam2_video_predictor import SAM2VideoPredictor

    root = args.render_dir
    renders = sorted((root / "removed" / "rgb").glob("*.png"))
    predictor = SAM2VideoPredictor.from_pretrained(args.sam2_model)

    with tempfile.TemporaryDirectory() as frame_dir, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, path in enumerate(renders):  # init_state reads JPEG frames named by integer index
            Image.open(path).convert("RGB").save(Path(frame_dir) / f"{i:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True)
        predictor.reset_state(state)
        prompted = 0
        for i in range(len(renders)):
            contour = np.asarray(Image.open(root / "unseen_contour" / f"{i:05d}.png")) > 127
            contour = largest_component(contour)
            if not contour.any():
                continue
            ys, xs = np.nonzero(contour)
            predictor.add_new_points_or_box(state, frame_idx=i, obj_id=1,
                                            box=np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], dtype=np.float32))
            prompted += 1
        print(f"SAM2: box prompts on {prompted}/{len(renders)} frames")
        if prompted == 0:
            raise SystemExit("unseen_contour/ is empty in every view, so there is nothing to prompt SAM2 with "
                             "(see the unseen_contour log)")
        masks = {}
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (logits[0, 0] > 0).cpu().numpy()

    for sub in ("unseen", "unseen_dilated"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    areas = []
    for i, path in enumerate(renders):
        h, w = np.asarray(Image.open(path)).shape[:2]
        mask = masks.get(i, np.zeros((h, w), dtype=bool))
        if args.union_deleted:
            gone = np.asarray(Image.open(root / "deleted" / "opacity" / f"{i:05d}.png"), dtype=np.float32) / 255.0
            mask = mask | (gone > args.deleted_threshold)
        dilated = process_mask(mask, args.dilate_kernel, args.dilate_iter) if mask.any() else mask
        Image.fromarray(mask.astype(np.uint8) * 255).save(root / "unseen" / f"{i:05d}.png")
        Image.fromarray(dilated.astype(np.uint8) * 255).save(root / "unseen_dilated" / f"{i:05d}.png")
        areas.append(dilated.mean())
    print(f"Done: unseen_dilated area mean={np.mean(areas):.3f} max={np.max(areas):.3f}, empty={sum(a == 0 for a in areas)}")


if __name__ == "__main__":
    main()
