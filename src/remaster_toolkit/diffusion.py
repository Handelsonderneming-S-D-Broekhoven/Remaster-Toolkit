# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in controlled diffusion refinement: SDXL base + tile ControlNet, img2img.

Headless, through diffusers (Apache-2.0). Models are local directory bundles
referenced by content hash (roles ``checkpoint`` = SDXL diffusers directory,
``controlnet`` = ControlNet directory). Locked parameters: seed (CPU
``torch.Generator``, float32 noise), sampler, steps, denoise, CFG, prompts,
ControlNet strength, precision, attention backend. Latents are decoded by us
in float32 (VAE upcast) so the decode precision is explicit.

``tiling.circularConv`` patches every Conv2d of UNet, ControlNet and VAE to
circular padding (keeps tileable textures seamless; RD-003 hypothesis).
Alpha never enters this step.
"""

from __future__ import annotations

import numpy as np

from .resample import resize

_PIPES: dict = {}


def _patch_circular(module, wrap_x: bool, wrap_y: bool) -> int:
    """Per-axis circular padding for every Conv2d (instance-level override of
    ``_conv_forward``); with both flags False the original method is restored."""
    import torch
    import torch.nn.functional as F

    n = 0
    for m in module.modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if "_conv_forward" in m.__dict__:
            del m._conv_forward
        if not (wrap_x or wrap_y) or isinstance(m.padding, str):
            continue
        ph, pw = m.padding
        if m.padding_mode != "zeros" or (ph == 0 and pw == 0):
            continue

        def fwd(x, weight, bias, m=m, ph=ph, pw=pw):
            if pw:
                x = F.pad(x, (pw, pw, 0, 0), mode="circular" if wrap_x else "constant")
            if ph:
                x = F.pad(x, (0, 0, ph, ph), mode="circular" if wrap_y else "constant")
            return F.conv2d(x, weight, bias, m.stride, 0, m.dilation, m.groups)

        m._conv_forward = fwd
        n += 1
    return n


def _load(sdxl_dir: str, cn_dir: str, compute: str):
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                           StableDiffusionXLControlNetImg2ImgPipeline, UNet2DConditionModel)
    from transformers import (CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer)

    key = (sdxl_dir, cn_dir, compute)
    if key in _PIPES:
        return _PIPES[key]
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[compute]
    variant = "fp16" if compute == "fp16" else None
    kw = dict(local_files_only=True, use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(sdxl_dir, subfolder="unet", torch_dtype=dt,
                                                variant=variant, **kw)
    te1 = CLIPTextModel.from_pretrained(sdxl_dir, subfolder="text_encoder", torch_dtype=dt,
                                        variant=variant, **kw)
    te2 = CLIPTextModelWithProjection.from_pretrained(sdxl_dir, subfolder="text_encoder_2",
                                                      torch_dtype=dt, variant=variant, **kw)
    tok1 = CLIPTokenizer.from_pretrained(sdxl_dir, subfolder="tokenizer", local_files_only=True)
    tok2 = CLIPTokenizer.from_pretrained(sdxl_dir, subfolder="tokenizer_2", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(sdxl_dir, subfolder="vae", torch_dtype=torch.float32, **kw)
    cn = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dt, **kw)
    sched = EulerDiscreteScheduler.from_pretrained(sdxl_dir, subfolder="scheduler",
                                                   local_files_only=True)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=vae, text_encoder=te1, text_encoder_2=te2, tokenizer=tok1, tokenizer_2=tok2,
        unet=unet, controlnet=cn, scheduler=sched)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    _PIPES[key] = pipe
    return pipe


def _work_size(h: int, w: int, target: int) -> tuple[int, int]:
    f = max(1.0, target / max(h, w))
    return (max(64, int(round(h * f / 64)) * 64), max(64, int(round(w * f / 64)) * 64))


def run(img: np.ndarray, sdxl_dir: str, cn_dir: str, step: dict, device: str) -> tuple[np.ndarray, dict]:
    """Refine RGB ``img`` (HxWx3, [0,1]); returns (image, info)."""
    import torch
    from PIL import Image

    d = step["diffusion"]
    prec = step.get("precision") or {}
    if d.get("rngDevice", "cpu") != "cpu" or d.get("noiseDtype", "float32") != "float32":
        raise ValueError("only CPU float32 noise is supported (cross-device seed stability)")
    if d.get("sampler", "euler") != "euler":
        raise ValueError("only the non-ancestral euler sampler is implemented")
    pipe = _load(sdxl_dir, cn_dir, prec.get("compute", "fp16"))
    circ = bool((step.get("tiling") or {}).get("circularConv", False))
    ax = (step.get("params") or {}).get("wrapAxes", "uv")
    wx, wy = circ and "u" in ax, circ and "v" in ax
    n = sum(_patch_circular(m, wx, wy) for m in (pipe.unet, pipe.controlnet, pipe.vae))
    h, w = img.shape[:2]
    wh, ww = _work_size(h, w, int(d.get("workingResolution", 1024)))
    mode = ("wrap" if wy else "clamp", "wrap" if wx else "clamp")
    x = resize(img[:, :, :3], ww, wh, "lanczos3", mode) if (wh, ww) != (h, w) else img[:, :, :3]
    pil = Image.fromarray(np.clip(np.floor(x * 255 + 0.5), 0, 255).astype(np.uint8))
    gen = torch.Generator("cpu").manual_seed(int(d["seed"]))
    from .sr import _sdpa_ctx
    with _sdpa_ctx(prec.get("attentionBackend", "default")), torch.inference_mode():
        lat = pipe(prompt=d.get("prompt", ""), negative_prompt=d.get("negativePrompt", ""),
                   image=pil, control_image=pil, strength=float(d["denoise"]),
                   num_inference_steps=int(d["steps"]), guidance_scale=float(d["cfg"]),
                   controlnet_conditioning_scale=float(d.get("controlnetStrength", 1.0)),
                   control_guidance_start=float(d.get("controlStart", 0.0)),
                   control_guidance_end=float(d.get("controlEnd", 1.0)),
                   height=wh, width=ww, generator=gen, output_type="latent").images
        # diffusers' img2img casts the (force_upcast) VAE back to the pipeline
        # dtype after encoding; decode explicitly in float32.
        vae = pipe.vae.to(dtype=torch.float32)  # offload hook moves it to the GPU
        lat = lat.to(device=pipe._execution_device, dtype=torch.float32) / vae.config.scaling_factor
        out = vae.decode(lat, return_dict=False)[0]
    y = ((out[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if (wh, ww) != (h, w):
        y = resize(y, w, h, "area" if wh > h else "lanczos3", mode)
    return y, {"workSize": [ww, wh], "circularConvLayers": n if circ else 0}
