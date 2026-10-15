# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinhole cameras (``cameras.json``) and pixel <-> world helpers shared by all stages.

Pixel centres sit at index + 0.5; ``c2w`` maps OpenCV camera coordinates to world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class ViewCamera:
    """Pinhole camera of one dataset view. ``c2w`` maps OpenCV camera coordinates to world."""

    index: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray

    def to_json(self) -> dict:
        return {**{k: getattr(self, k) for k in ("index", "name", "width", "height", "fx", "fy", "cx", "cy")},
                "c2w": self.c2w.tolist()}

    @staticmethod
    def from_json(d: dict) -> "ViewCamera":
        return ViewCamera(**{**d, "c2w": np.asarray(d["c2w"], dtype=np.float64)})


def load_cameras(path: Path) -> list[ViewCamera]:
    return [ViewCamera.from_json(d) for d in json.loads(Path(path).read_text())]


def save_cameras(cameras: list[ViewCamera], path: Path) -> None:
    Path(path).write_text(json.dumps([c.to_json() for c in cameras], indent=1))


def pixel_grid(camera: ViewCamera, device) -> torch.Tensor:
    """[H, W, 2] continuous image coordinates of pixel centres (3DGRUT convention: index + 0.5)."""
    v, u = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32),
        torch.arange(camera.width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([u + 0.5, v + 0.5], dim=-1)


def unproject(camera: ViewCamera, depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """World points of the (masked) pixels of ``camera`` at camera-space z ``depth`` [H, W]."""
    uv = pixel_grid(camera, depth.device)
    if mask is not None:
        uv, depth = uv[mask], depth[mask]
    else:
        uv, depth = uv.reshape(-1, 2), depth.reshape(-1)
    x = (uv[:, 0] - camera.cx) / camera.fx * depth
    y = (uv[:, 1] - camera.cy) / camera.fy * depth
    points_cam = torch.stack([x, y, depth], dim=-1)
    c2w = torch.as_tensor(camera.c2w, dtype=torch.float32, device=depth.device)
    return points_cam @ c2w[:3, :3].T + c2w[:3, 3]


def project(cameras: list[ViewCamera], points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points [P, 3] into every camera. Returns pixel column/row indices [J, P] and z [J, P]."""
    w2c = torch.as_tensor(np.stack([np.linalg.inv(c.c2w) for c in cameras]), dtype=torch.float32, device=points.device)
    intr = torch.as_tensor([[c.fx, c.fy, c.cx, c.cy] for c in cameras], dtype=torch.float32, device=points.device)
    cam = torch.einsum("jab,pb->jpa", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    safe_z = z.clamp_min(1e-6)
    u = cam[..., 0] / safe_z * intr[:, 0:1] + intr[:, 2:3]
    v = cam[..., 1] / safe_z * intr[:, 1:2] + intr[:, 3:4]
    return torch.floor(u).long(), torch.floor(v).long(), z


def save_png(array: np.ndarray | torch.Tensor, path: Path) -> None:
    """Save a [H,W] mask/opacity in [0,1] or bool, or an [H,W,3] image in [0,1], as 8-bit PNG."""
    if torch.is_tensor(array):
        array = array.detach().float().cpu().numpy()
    if array.dtype == bool:
        array = array.astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).save(path)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127
