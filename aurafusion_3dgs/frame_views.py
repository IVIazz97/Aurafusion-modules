#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion af360v: which views frame the whole unseen mask, and the reference view among them.

A view's processed unseen mask (``unseen_dilated``) is "in frame" when no mask pixel lies within
``--margin`` (fraction of the shorter image side, at least 2 px) of the border. Only those views
show the whole hole, so only they can give a consistent reference and supervise the hole's inside
in ``finetune --in_frame_only``.

Reference: the in-frame real view with the largest unseen mask; if no real view frames it, the
in-frame virtual view with the largest mask (``--virtual_json``); if none at all, the largest mask
overall (logged as a warning). Writes ``<render_dir>/in_frame.json`` and prints the index (run_removal.sh
stores it as reference_index.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def in_frame(mask: np.ndarray, margin: float) -> bool:
    """True if the (non-empty) mask stays ``margin`` x the shorter side away from every border."""
    if not mask.any():
        return False
    m = max(2, int(round(margin * min(mask.shape))))
    return not (mask[:m].any() or mask[-m:].any() or mask[:, :m].any() or mask[:, -m:].any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--virtual_json", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--summary", type=Path, default=None, help="Append a one-line human-readable summary here.")
    args = parser.parse_args()

    root = args.render_dir
    paths = sorted((root / "unseen_dilated").glob("*.png"))
    num_cameras = len(json.loads((root / "cameras.json").read_text()))
    if len(paths) != num_cameras:
        raise SystemExit(f"frame_views: {len(paths)} masks in {root / 'unseen_dilated'} but {num_cameras} views in "
                         f"{root / 'cameras.json'}; rerun sam2_unseen")
    num_real = len(paths)
    if args.virtual_json is not None:
        num_real = json.loads(args.virtual_json.read_text())["num_real"]
    masks = [np.asarray(Image.open(p)) > 127 for p in paths]
    area = [int(m.sum()) for m in masks]
    framed = [in_frame(m, args.margin) for m in masks]
    if not any(area):
        raise SystemExit(f"frame_views: every unseen mask in {root / 'unseen_dilated'} is empty; nothing to inpaint")
    for i, p in enumerate(paths):
        kind = "real" if i < num_real else "virtual"
        state = "empty" if not area[i] else "in frame" if framed[i] else "CUT by the border"
        print(f"view {p.stem} ({kind}): unseen mask {area[i]} px ({100 * area[i] / masks[i].size:.1f}%), {state}", file=sys.stderr)
    real = [i for i in range(num_real) if framed[i]]
    virtual = [i for i in range(num_real, len(paths)) if framed[i]]
    pool, source = (real, "real") if real else (virtual, "virtual") if virtual else (list(range(len(paths))), "any")
    if source == "any":
        print("WARNING: no view frames the whole unseen mask; taking the largest mask", file=sys.stderr)
    reference = max(pool, key=lambda i: area[i])
    (root / "in_frame.json").write_text(json.dumps({
        "margin": args.margin, "num_real": num_real, "in_frame": [int(p.stem) for p, f in zip(paths, framed) if f],
        "real_in_frame": len(real), "virtual_in_frame": len(virtual), "reference": int(paths[reference].stem),
        "reference_from": source, "reference_area": area[reference],
    }, indent=1))
    line = (f"af360v: hole fully in frame in {len(real)}/{num_real} real + {len(virtual)}/{len(paths) - num_real} virtual "
            f"views (these supervise the hole), reference = view {paths[reference].stem} ({source}, "
            f"{area[reference]} px unseen)" + ("; WARNING: no view frames the whole hole" if source == "any" else ""))
    print(line, file=sys.stderr)
    if args.summary is not None:
        with open(args.summary, "a") as f:
            f.write(line + "\n")
    print(f"in frame: {len(real)}/{num_real} real, {len(virtual)}/{len(paths) - num_real} virtual views; "
          f"reference {paths[reference].stem} ({source}, {area[reference]} px)", file=sys.stderr)
    print(int(paths[reference].stem))


if __name__ == "__main__":
    main()
