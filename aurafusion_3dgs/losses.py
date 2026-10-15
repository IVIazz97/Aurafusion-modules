#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper Eq. 12 (official ``inpaint_finetune``): the per-view finetune loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from fused_ssim import fused_ssim

    return fused_ssim(a.permute(2, 0, 1)[None].contiguous(), b.permute(2, 0, 1)[None].contiguous())


def af360_loss(image: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, supervise_hole: bool, lpips,
               lambda_dssim: float = 0.2, lambda_lpips: float = 0.5) -> torch.Tensor:
    """Paper Eq. 12 on one view: image, gt [H, W, 3] in [0, 1], mask [H, W] bool (processed unseen mask)."""
    m = mask[..., None].float()
    loss = 0.8 * F.l1_loss(image * (1 - m), gt * (1 - m)) + 0.2 * (1 - ssim(image * (1 - m), gt * (1 - m)))
    if mask.any() and supervise_hole:
        loss = loss + (1 - lambda_dssim) * F.l1_loss(image * m, gt * m) + lambda_dssim * (1 - ssim(image * m, gt * m))
        ys, xs = torch.nonzero(mask, as_tuple=True)
        crop = (slice(ys.min().item(), ys.max().item() + 1), slice(xs.min().item(), xs.max().item() + 1))
        ci, cg = image[crop].permute(2, 0, 1)[None], gt[crop].permute(2, 0, 1)[None]
        ph, pw = ci.shape[2] // 2, ci.shape[3] // 2
        if ph >= 16 and pw >= 16:  # 2x2 patches, as the official divide_into_patches(K=2)
            patches_i = torch.cat([ci[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
            patches_g = torch.cat([cg[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
            loss = loss + lambda_lpips * lpips(patches_i.clamp(0, 1), patches_g)
    return loss
