#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# AGDDv2 below is adapted from AuraFusion360_official utils/marigold_di_utils.py, which builds on
# diffusers' MarigoldDepthPipeline (Copyright 2024 Marigold authors, PRS ETH Zurich / HuggingFace,
# Apache-2.0).

"""AuraFusion-3DGS stage 4 (2D env): reference inpainting + Adaptive Guided Depth Diffusion.

Reference RGB: either a given image (``--reference_image``, e.g. the official Other-360 kitchen
reference) or built here: the photo, with the removed-scene render inside the object's footprint
(the seen background) and LaMa filling the unseen mask.

AGDD (paper Eq. 5-8): Marigold v1-0 depth diffusion whose latent is optimised at every
denoising step so the predicted depth matches the removed-scene depth outside the unseen mask,
inside a box around it (Huber loss), with re-noise averaging in the late steps. Hyperparameters
default to the official Other-360/kitchen config. Deviation: torch AdamW instead of
bitsandbytes' AdamW8bit (8-bit state only saves memory).

Writes <render_dir>/reference/{reference.png,composite.png,depth_aligned.npy,depth_vis.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def lama_inpaint(model_path: Path, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    model = torch.jit.load(str(model_path), map_location="cuda").eval()
    h, w = mask.shape
    x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    m = torch.from_numpy(mask.copy())[None, None].float().cuda()
    ph, pw = (-h) % 8, (-w) % 8
    with torch.inference_mode():
        y = model(F.pad(x, (0, pw, 0, ph), mode="reflect"), F.pad(m, (0, pw, 0, ph), mode="reflect"))
    y = (y[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return np.where(mask[..., None], y, image)


def dilate(mask: torch.Tensor, iterations: int, kernel_size: int) -> torch.Tensor:
    for _ in range(iterations):
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return mask


def adaptive_loss(prediction, incomplete_depth, unseen_mask, opt, greater_zero_mask):
    """Official adaptive_loss: Huber on the visible part of a box around the dilated unseen mask."""
    unseen_dilated = dilate(unseen_mask[None], opt.dilate_iter, opt.kernel_size)[0]
    visible = (1 - unseen_mask) * greater_zero_mask.bool()
    ys, xs = torch.where(unseen_dilated[0] > 0.5)
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    crop_pred = prediction[0, y0:y1, x0:x1]
    crop_gt = incomplete_depth[0, y0:y1, x0:x1]
    crop_mask = visible[0, y0:y1, x0:x1]
    return F.huber_loss(crop_pred * crop_mask, crop_gt * crop_mask, reduction="mean", delta=opt.delta)


def run_agdd(pipe, image, incomplete_depth, unseen_mask, opt, generator, num_inference_steps=50,
             processing_resolution=768):
    """AGDDv2.__call__ of the official code, with is_latent_optimizing=True."""
    device, dtype = pipe._execution_device, pipe.dtype
    with torch.no_grad():
        if pipe.empty_text_embedding is None:
            ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
            pipe.empty_text_embedding = pipe.text_encoder(ids)[0]
        image, padding, original_resolution = pipe.image_processor.preprocess(
            image, processing_resolution, "bilinear", device=device, dtype=dtype)
        image_latent, pred_latent = pipe.prepare_latents(image, None, generator, 1, 1)
    text = pipe.empty_text_embedding.to(device=device, dtype=dtype)
    # fp32 master latent: torch AdamW state in fp16 underflows eps (the official uses 8-bit AdamW).
    pred_latent = torch.nn.Parameter(pred_latent.float())
    optimizer = torch.optim.AdamW([pred_latent], lr=opt.agdd_lr, weight_decay=0.0)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    greater_zero = incomplete_depth > 0
    for p in list(pipe.unet.parameters()) + list(pipe.vae.parameters()):
        p.requires_grad_(False)

    def unet(latent, t):
        return pipe.unet(torch.cat([image_latent, latent.to(dtype)], dim=1), t, encoder_hidden_states=text,
                         return_dict=False)[0].float()

    def decode(latent):
        prediction = pipe.decode_prediction(latent.to(dtype)).float()
        prediction = pipe.image_processor.unpad_image(prediction, padding)
        return pipe.image_processor.resize_antialias(prediction, original_resolution, "bilinear", is_aa=False)[0]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    for i, t in enumerate(timesteps):
        for _ in range(opt.optimize_iter):
            latent_t0 = pipe.scheduler.step(unet(pred_latent, t), t, pred_latent, generator=generator).pred_original_sample
            loss = adaptive_loss(decode(latent_t0), incomplete_depth, unseen_mask, opt, greater_zero) * opt.agdd_loss_scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred_latent.data.copy_(pipe.scheduler.step(unet(pred_latent, t), t, pred_latent).prev_sample)
            if opt.use_renoise and opt.renoise_start_iter <= i < num_inference_steps - 1 and i % opt.renoise_step == 0:
                current = pred_latent.detach().clone()
                alpha = pipe.scheduler.alphas_cumprod[t] / pipe.scheduler.alphas_cumprod[timesteps[i + 1]]
                averaged = []
                for _ in range(opt.infer_iter):
                    eps = torch.randn(current.shape, generator=generator).to(device)
                    noisy = alpha.sqrt() * current + (1 - alpha).sqrt() * eps
                    averaged.append(pipe.scheduler.step(unet(noisy, t), t, current, generator=generator).prev_sample)
                pred_latent.data.copy_(torch.stack(averaged).mean(0))
        lr_schedule.step()
        if i % 10 == 0:
            print(f"AGDD step {i}/{num_inference_steps}: loss {loss.item():.5f}", flush=True)
    with torch.no_grad():
        return decode(pred_latent)  # [1, H, W] in [0, 1]


def normalize_ignore_zeros(depth: torch.Tensor, region: torch.Tensor | None = None,
                           quantile: float = 0.99) -> tuple[torch.Tensor, float, float]:
    """Map the guide depth to [0, 1] over the range the loss actually looks at.

    The official code takes the whole image's min/max. In a 360 scene a handful of far
    background pixels sit tens of times further away than the hole, which squeezes the
    loss crop into a sliver of [0, 1] -- and un-normalising multiplies whatever residual
    is left there by that same full range. Taking a robust range over the loss crop
    (``region``) keeps the crop spread across [0, 1]; anything further away clamps to 1,
    which the loss never sees.
    """
    valid = depth != 0
    sample = depth[valid if region is None else (valid & region)]
    lo = float(sample.quantile(1 - quantile))
    hi = float(sample.quantile(quantile))
    out = torch.zeros_like(depth)
    out[valid] = ((depth[valid] - lo) / (hi - lo)).clamp(0, 1)
    return out, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True, help="Prepared COLMAP scene with images/.")
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--reference_image", type=Path, default=None, help="Use this inpainted reference instead of LaMa.")
    parser.add_argument("--lama_model", type=Path, required=True)
    parser.add_argument("--marigold", default="prs-eth/marigold-v1-0")
    parser.add_argument("--object_threshold", type=float, default=0.02)
    parser.add_argument("--object_dilate", type=int, default=7, help="Pixels of object footprint taken from the removed render.")
    # Official Other-360/kitchen inpaint.config + OptimizationParams defaults.
    parser.add_argument("--dilate_iter", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--optimize_iter", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--infer_iter", type=int, default=4)
    parser.add_argument("--agdd_lr", type=float, default=0.01)
    parser.add_argument("--agdd_loss_scale", type=float, default=1e4)
    parser.add_argument("--renoise_start_iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7777)
    args = parser.parse_args()
    opt = SimpleNamespace(dilate_iter=args.dilate_iter, kernel_size=args.kernel_size, optimize_iter=args.optimize_iter,
                          delta=args.delta, infer_iter=args.infer_iter, agdd_lr=args.agdd_lr,
                          agdd_loss_scale=args.agdd_loss_scale, use_renoise=True,
                          renoise_start_iter=args.renoise_start_iter, renoise_step=1)

    root, r = args.render_dir, args.reference_index
    out = root / "reference"
    out.mkdir(parents=True, exist_ok=True)
    camera = json.loads((root / "cameras.json").read_text())[r]
    unseen = np.asarray(Image.open(root / "unseen_dilated" / f"{r:05d}.png")) > 127
    assert unseen.any(), f"reference view {r} has an empty unseen mask; pick another --reference_index"

    # 1. Inpainted reference RGB.
    removed = np.asarray(Image.open(root / "removed" / "rgb" / f"{r:05d}.png").convert("RGB"))
    if args.reference_image is not None:
        reference = np.asarray(Image.open(args.reference_image).convert("RGB").resize(removed.shape[1::-1], Image.BICUBIC))
        composite = reference
    else:
        photo = Image.open(args.colmap_dir / "images" / camera["name"]).convert("RGB")
        if photo.size != removed.shape[1::-1]:  # scene trained on downsampled photos
            photo = photo.resize(removed.shape[1::-1], Image.LANCZOS)
        photo = np.asarray(photo)
        obj = torch.from_numpy(np.asarray(Image.open(root / "object" / "opacity" / f"{r:05d}.png"), dtype=np.float32) / 255.0)
        obj = dilate((obj > args.object_threshold).float()[None, None], args.object_dilate, 3)[0, 0].numpy() > 0.5
        composite = np.where(obj[..., None], removed, photo)
        reference = lama_inpaint(args.lama_model, composite, unseen)
    Image.fromarray(composite).save(out / "composite.png")
    Image.fromarray(reference).save(out / "reference.png")

    # 2. AGDD depth for the reference, guided by the removed-scene depth outside the unseen mask.
    from diffusers import DDPMScheduler, MarigoldDepthPipeline

    try:
        pipe = MarigoldDepthPipeline.from_pretrained(args.marigold, variant="fp16", torch_dtype=torch.float16)
    except (OSError, ValueError):  # a local copy without the *.fp16.safetensors files
        pipe = MarigoldDepthPipeline.from_pretrained(args.marigold, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    depth = torch.from_numpy(np.load(root / "removed" / "depth" / f"{r:05d}.npy")).cuda()[None]  # [1, H, W]
    mask = torch.from_numpy(unseen).cuda().float()[None]  # [1, H, W]
    depth[mask == 1] = 0
    box = dilate(mask[None], opt.dilate_iter, opt.kernel_size)[0, 0] > 0.5   # adaptive_loss's crop
    ys, xs = torch.where(box)
    region = torch.zeros_like(mask, dtype=torch.bool)
    region[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    guide, lo, hi = normalize_ignore_zeros(depth, region)
    rgb = torch.from_numpy(reference).permute(2, 0, 1).float().div(255).cuda()
    generator = torch.Generator().manual_seed(args.seed)
    aligned = run_agdd(pipe, rgb.half(), guide, mask, opt, generator) * (hi - lo) + lo
    err = (aligned[mask == 0] - depth[mask == 0]).abs()[depth[mask == 0] > 0]
    print(f"AGDD: final mean |aligned - render| outside unseen mask = {err.mean().item():.4f} (depth range {lo:.3f}..{hi:.3f})")
    np.save(out / "depth_aligned.npy", aligned[0].cpu().numpy().astype(np.float32))
    vis = aligned[0].cpu().numpy()
    Image.fromarray((255 * (vis - vis.min()) / max(vis.max() - vis.min(), 1e-6)).astype(np.uint8)).save(out / "depth_vis.png")
    print(f"Done: reference view {r} -> {out}")


if __name__ == "__main__":
    main()
