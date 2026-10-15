#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion-3DGS stage 6 (2D env): SDEdit detail enhancement with LeftRefill (paper Eq. 9-11).

Runs the official ``utils/LeftRefill/sdedit_utils.py`` unchanged, imported from the official
clone: for every view, the rendered initial Gaussians are DDIM-inverted to t = T * 0.85 and
denoised by SD2-inpainting + LeftRefill's learned prompt with the inpainted reference
concatenated on the left, only inside the processed unseen mask. The reference view itself gets
the reference image. Settings = official Other-360/kitchen sdedit.config
(use_ddim_inversion=True, strength=0.85, eta=1.0, scale=2.5).

Writes <render_dir>/sdedit/<i>.png, the finetune targets.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--official_repo", type=Path, required=True, help="AuraFusion360_official clone.")
    parser.add_argument("--sd2_inpainting_ckpt", type=Path, required=True, help="512-inpainting-ema.ckpt")
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--no_ddim_inversion", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.render_dir.resolve()
    leftrefill = (args.official_repo / "utils" / "LeftRefill").resolve()
    pretrained = leftrefill / "pretrained_models" / "512-inpainting-ema.ckpt"
    if not pretrained.exists():  # sdedit_utils would otherwise wget it at import
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        pretrained.symlink_to(args.sd2_inpainting_ckpt.resolve())

    # LeftRefill pairs the natsorted listings of source/ref/mask dirs; the reference file's stem
    # must equal the reference view's name in ref_root for it to be copied through unchanged.
    reference_dir = root / "sdedit_reference"
    reference_dir.mkdir(exist_ok=True)
    reference_path = reference_dir / f"{args.reference_index:05d}.png"
    shutil.copy(root / "reference" / "reference.png", reference_path)
    output = root / "sdedit"
    output.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(leftrefill)
    sys.path.insert(0, str(leftrefill))
    sys.argv = [sys.argv[0]]  # sdedit_utils builds the model at import and must not see our flags
    import sdedit_utils  # noqa: E402  (loads SD2-inpainting + LeftRefill prompt, ~1 min)

    with torch.no_grad():
        sdedit_utils.LeftRefill(
            str(reference_path),
            source_root=str(root / "init" / "rgb"),
            ref_root=str(root / "init" / "rgb"),
            mask_root=str(root / "unseen_dilated"),
            output_root=str(output),
            strength=None if args.strength >= 1 else args.strength,
            eta=args.eta,
            scale=args.scale,
            use_ddim_inversion=not args.no_ddim_inversion,
        )
    print(f"Done: {len(list(output.glob('*.png')))} SDEdit targets -> {output}")


if __name__ == "__main__":
    main()
