# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphdeco 3DGS access for the AuraFusion stages: point_cloud.ply I/O, COLMAP cameras, rendering.

Rendering calls graphdeco's ``diff_gaussian_rasterization`` directly (env ``aurafusion_3dgs``: the
gaussian-splatting env + scipy/torchmetrics), with the camera built exactly like graphdeco's ``Camera``
(znear 0.01, zfar 100, FoV from the focal). That rasteriser assumes the principal point at the image
centre, so the ``ViewCamera`` of every view gets cx = W/2, cy = H/2 (``cameras`` puts pixel
centres at index + 0.5), whatever COLMAP says -- graphdeco's loader ignores cx, cy too.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aurafusion_3dgs.cameras import ViewCamera

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    """Raw (pre-activation) graphdeco parameters. features_dc [N, 1, 3], features_rest [N, (D+1)^2 - 1, 3]."""

    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacity: torch.Tensor  # [N, 1] logit
    scaling: torch.Tensor  # [N, 3] log
    rotation: torch.Tensor  # [N, 4] unnormalised wxyz

    def __len__(self) -> int:
        return self.xyz.shape[0]

    @property
    def sh_degree(self) -> int:
        return round(math.sqrt(self.features_rest.shape[1] + 1)) - 1

    def subset(self, keep: torch.Tensor) -> "Gaussians":
        return Gaussians(*(getattr(self, f.name)[keep] for f in fields(self)))

    @staticmethod
    def cat(a: "Gaussians", b: "Gaussians") -> "Gaussians":
        return Gaussians(*(torch.cat([getattr(a, f.name), getattr(b, f.name)]) for f in fields(a)))

    def detach(self) -> "Gaussians":
        return Gaussians(*(getattr(self, f.name).detach() for f in fields(self)))


def load_ply(path: Path, device: str = "cuda") -> Gaussians:
    """graphdeco ``GaussianModel.load_ply``; extra vertex properties (e.g. Inpaint360GS's objects) are ignored."""
    from plyfile import PlyData

    v = PlyData.read(str(path)).elements[0]

    def stack(names):
        return np.stack([np.asarray(v[n], dtype=np.float32) for n in names], axis=1)

    def by_index(prefix):
        names = sorted((p.name for p in v.properties if p.name.startswith(prefix)), key=lambda n: int(n.split("_")[-1]))
        return stack(names)

    rest = by_index("f_rest_")
    rest = rest.reshape(len(rest), 3, -1).transpose(0, 2, 1)  # stored channel-major, as graphdeco
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return Gaussians(xyz=t(stack(["x", "y", "z"])), features_dc=t(stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None]),
                     features_rest=t(rest), opacity=t(stack(["opacity"])), scaling=t(by_index("scale_")),
                     rotation=t(by_index("rot")))


def save_ply(g: Gaussians, path: Path) -> None:
    """graphdeco ``GaussianModel.save_ply`` layout, readable by any 3DGS viewer / loader."""
    from plyfile import PlyData, PlyElement

    n = len(g)
    cols = {
        "x": g.xyz[:, 0], "y": g.xyz[:, 1], "z": g.xyz[:, 2],
        **{k: torch.zeros(n) for k in ("nx", "ny", "nz")},
        **{f"f_dc_{i}": g.features_dc[:, 0, i] for i in range(3)},
        **{f"f_rest_{i}": c for i, c in enumerate(g.features_rest.transpose(1, 2).reshape(n, -1).T)},
        "opacity": g.opacity[:, 0],
        **{f"scale_{i}": g.scaling[:, i] for i in range(3)},
        **{f"rot_{i}": g.rotation[:, i] for i in range(4)},
    }
    data = np.empty(n, dtype=[(k, "f4") for k in cols])
    for k, c in cols.items():
        data[k] = c.detach().float().cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


# ----------------------------------------------------------------------------------------------- cameras

def _qvec2rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5,
              "FULL_OPENCV": 6, "FOV": 7, "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10}


def _focals(model: int, p) -> tuple[float, float]:
    return (p[0], p[0]) if model in (0, 2, 3, 8, 9) else (p[0], p[1])


def _read_colmap(sparse: Path) -> list[tuple[str, tuple[int, int, float, float], np.ndarray, np.ndarray]]:
    """[(image name, (w, h, fx, fy), R_w2c, t_w2c)] from cameras/images .bin, else .txt."""
    cams, out = {}, []
    if (sparse / "images.bin").exists():
        def read(f, n, fmt):
            return struct.unpack("<" + fmt, f.read(n))
        with open(sparse / "cameras.bin", "rb") as f:
            for _ in range(read(f, 8, "Q")[0]):
                cid, model, w, h = read(f, 24, "iiQQ")
                p = read(f, 8 * _NUM_PARAMS[model], "d" * _NUM_PARAMS[model])
                cams[cid] = (w, h, *_focals(model, p))
        with open(sparse / "images.bin", "rb") as f:
            for _ in range(read(f, 8, "Q")[0]):
                _, qw, qx, qy, qz, tx, ty, tz, cid = read(f, 64, "idddddddi")
                name = b""
                while (c := f.read(1)) != b"\x00":
                    name += c
                f.read(24 * read(f, 8, "Q")[0])
                out.append((name.decode(), cams[cid], _qvec2rotmat([qw, qx, qy, qz]), np.array([tx, ty, tz])))
        return out
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            e = line.split()
            cams[int(e[0])] = (int(e[2]), int(e[3]), *_focals(_MODEL_IDS[e[1]], [float(x) for x in e[4:]]))
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    k = 0
    while k < len(lines):  # pairs of lines: pose + name, then the (possibly empty) 2D points
        e = lines[k].split()
        if len(e) >= 10:
            out.append((" ".join(e[9:]), cams[int(e[8])], _qvec2rotmat([float(x) for x in e[1:5]]),
                        np.array([float(x) for x in e[5:8]])))
            k += 2
        else:
            k += 1
    return out


def load_colmap_cameras(colmap_dir: Path, images: str = "images") -> list[ViewCamera]:
    """Every COLMAP view, sorted by image name (graphdeco's order), intrinsics rescaled to the image files
    in ``colmap_dir/images`` (e.g. ``images_4``); principal point at the image centre (see module doc)."""
    colmap_dir = Path(colmap_dir)
    sparse = colmap_dir / "sparse" / "0" if (colmap_dir / "sparse" / "0").exists() else colmap_dir / "sparse"
    cameras = []
    for index, (name, (w, h, fx, fy), R, t) in enumerate(sorted(_read_colmap(sparse), key=lambda e: e[0])):
        W, H = Image.open(colmap_dir / images / name).size
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        cameras.append(ViewCamera(index=index, name=name, width=W, height=H, fx=fx * W / w, fy=fy * H / h,
                                  cx=W / 2, cy=H / 2, c2w=c2w))
    return cameras


def scene_extent(cameras: list[ViewCamera]) -> float:
    """graphdeco ``getNerfppNorm`` radius = the spatial_lr_scale of the position learning rate."""
    centres = np.stack([c.c2w[:3, 3] for c in cameras])
    return float(1.1 * np.linalg.norm(centres - centres.mean(0), axis=1).max())


# ----------------------------------------------------------------------------------------------- rendering

def _projection(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    top, right = math.tan(fovy / 2) * znear, math.tan(fovx / 2) * znear
    P = torch.zeros(4, 4)
    P[0, 0], P[1, 1] = znear / right, znear / top
    P[3, 2], P[2, 2], P[2, 3] = 1.0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)
    return P


def rasterize(g: Gaussians, camera: ViewCamera, colors: torch.Tensor | None = None, background=(0.0, 0.0, 0.0),
              antialiasing: bool = False) -> torch.Tensor:
    """[3, H, W] unclamped colour: SH colours, or ``colors`` [N, 3] blended as is (alpha * T weights)."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = g.xyz.device
    w2c = torch.as_tensor(np.linalg.inv(camera.c2w), dtype=torch.float32, device=device)
    view = w2c.T  # graphdeco world_view_transform (row-vector convention)
    fovx, fovy = 2 * math.atan(camera.width / (2 * camera.fx)), 2 * math.atan(camera.height / (2 * camera.fy))
    proj = view @ _projection(0.01, 100.0, fovx, fovy).T.to(device)
    settings = GaussianRasterizationSettings(
        image_height=camera.height, image_width=camera.width, tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
        bg=torch.tensor(background, dtype=torch.float32, device=device), scale_modifier=1.0, viewmatrix=view,
        projmatrix=proj, sh_degree=g.sh_degree, campos=torch.as_tensor(camera.c2w[:3, 3], dtype=torch.float32, device=device),
        prefiltered=False, debug=False, antialiasing=antialiasing)
    means2d = torch.zeros_like(g.xyz, requires_grad=g.xyz.requires_grad)
    image, _, _ = GaussianRasterizer(settings)(
        means3D=g.xyz, means2D=means2d, opacities=torch.sigmoid(g.opacity),
        shs=None if colors is not None else torch.cat([g.features_dc, g.features_rest], 1),
        colors_precomp=colors, scales=torch.exp(g.scaling), rotations=torch.nn.functional.normalize(g.rotation))
    return image


def camera_z(g: Gaussians, camera: ViewCamera) -> torch.Tensor:
    w2c = torch.as_tensor(np.linalg.inv(camera.c2w), dtype=torch.float32, device=g.xyz.device)
    return g.xyz @ w2c[2, :3] + w2c[2, 3]


@torch.no_grad()
def render_view(g: Gaussians, camera: ViewCamera) -> dict[str, torch.Tensor]:
    """RGB [H, W, 3] (clamped like graphdeco's render), opacity [H, W] and camera z-depth [H, W]: the
    alpha-weighted mean of the Gaussian centres' z, from one extra render with colours (z, 1, 0)."""
    rgb = rasterize(g, camera).clamp(0, 1).permute(1, 2, 0)
    z = camera_z(g, camera)
    geometry = rasterize(g, camera, colors=torch.stack([z, torch.ones_like(z), torch.zeros_like(z)], 1))
    opacity = geometry[1]
    return {"rgb": rgb, "opacity": opacity.clamp(0, 1), "depth": geometry[0] / opacity.clamp_min(1e-6)}


def blend_weights(g: Gaussians, cameras: list[ViewCamera], masks: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Per Gaussian, the alpha * T weight it draws over every pixel of every view: [N] total, or with ``masks``
    (bool [H, W] per view) [2, N] = (weight outside the masks, weight inside). The render is linear in the
    precomputed colours, so the gradient of the masked pixel sum w.r.t. a Gaussian's colour is that weight
    (FlashSplat's linear-in-labels observation)."""
    total = torch.zeros(2 if masks is not None else 1, len(g), device=g.xyz.device)
    frozen = g.detach()
    for n, camera in enumerate(cameras):
        colors = torch.ones(len(g), 3, device=g.xyz.device, requires_grad=True)
        image = rasterize(frozen, camera, colors=colors)
        if masks is None:
            image[0].sum().backward()
            total[0] += colors.grad[:, 0]
        else:
            m = masks[n].to(image.device).float()
            ((image[0] * (1 - m)).sum() + (image[1] * m).sum()).backward()
            total += colors.grad[:, :2].T
    return total if masks is not None else total[0]
