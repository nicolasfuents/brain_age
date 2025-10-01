#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------
# explain_clinical_v2.py — versión clínica mínima
# Mantiene SOLO:
#  - Filtros: --age-min, --prefer-perfect-ensemble, --perfect-tol, --n, --planes
#  - Explicabilidad: IG firmado, Occlusion firmado, Grad-CAM global firmado
#  - slice-mode, alpha, outdir
#  - Montajes por plano y ensemble, variantes _nobar y NPY
# Elimina TODO QC, máscaras y morfología, y cualquier parámetro asociado.
# Ejemplo de uso:
# python explain_clinical_v2.py   --axial-id 20250822-1437 --coronal-id 20250814-0853 --sagittal-id 20250822-1437   --planes axial coronal sagittal --n 1 --slice-mode middle   --ig-steps 96 --ig-baseline mean   --occ-patch 12 --occ-stride 8 --occ-baseline zero   --gradcam global --alpha 0.4   --prefer-perfect-ensemble --perfect-tol 0.5   --age-min 60 --topk 10   --outdir ./explanations_clinical
# ---------------------------------------------------------

import os, sys, re, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import csv
from collections import defaultdict
from typing import Dict
from matplotlib import colormaps as cmaps
from skimage import measure
from PIL import Image
import imageio.v2 as imageio

# === Importar tu modelo desde el root del repo ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Utils --------------------
def _py(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().item() if obj.numel()==1 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    return obj

def _to_numpy01(t):
    t = t.detach().float().cpu()
    t = t - t.min()
    if t.max() > 0: t = t / t.max()
    return t.numpy()

def custom_fmt(x, pos):
    if abs(x-0) < 1e-8 or abs(x-1) < 1e-8 or abs(x+1) < 1e-8:
        return f"{int(x)}"
    return f"{x:.2f}"

def atlas_slices_matched(atlas_path, subj_shape, plane, indices):
    import numpy as np, nibabel as nib
    from skimage.transform import resize
    Hs, Ws, Ds = map(int, subj_shape)
    try:
        import SimpleITK as sitk
        atl = sitk.ReadImage(atlas_path)  # etiquetas MNI
        dest = sitk.Image(Hs, Ws, Ds, atl.GetPixelID())   # espacio destino = shape del sujeto
        dest.SetSpacing(atl.GetSpacing()); dest.SetOrigin(atl.GetOrigin()); dest.SetDirection(atl.GetDirection())
        init = sitk.CenteredTransformInitializer(dest, atl, sitk.Euler3DTransform(),
                                                 sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 100,
                                                     relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(init, inPlace=False)
        T = reg.Execute(dest, atl)
        atlas_rs = sitk.Resample(atl, dest, T, sitk.sitkNearestNeighbor, 0)
        vol_rs = sitk.GetArrayFromImage(atlas_rs).astype(np.int16)  # z,y,x
        vol_rs = np.transpose(vol_rs, (2,1,0))  # -> (H,W,D)
    except Exception:
        vol = nib.load(atlas_path).get_fdata().astype(np.int16)
        vol_rs = resize(vol, (Hs, Ws, Ds), order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)

    out = []
    for idx in indices:
        if plane == "sagittal":   sl = vol_rs[idx, :, :]
        elif plane == "coronal":  sl = vol_rs[:, idx, :]
        else:                     sl = vol_rs[:, :, idx]
        sl = resize(sl, (160,160), order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)
        out.append(sl)
    return np.stack(out, 0)


def _apply_display_orient(arr, plane):
    a = np.rot90(np.asarray(arr), k=1)  # Rotar 90° CCW para visualización
    return a

def _add_orient_labels(ax, plane):
    # Convención radiológica; solo cambiamos sagittal: left=P, right=A
    if plane == "axial":
        lbl = {"left":"R", "right":"L", "top":"A", "bottom":"P"}
    elif plane == "coronal":
        lbl = {"left":"R", "right":"L", "top":"S", "bottom":"I"}
    else:  # sagittal
        lbl = {"left":"P", "right":"A", "top":"S", "bottom":"I"}

    ax.text(0.01, 0.50, lbl["left"],  transform=ax.transAxes, va="center", ha="left",
            fontsize=11, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))
    ax.text(0.99, 0.50, lbl["right"], transform=ax.transAxes, va="center", ha="right",
            fontsize=11, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))
    ax.text(0.50, 0.02, lbl["bottom"], transform=ax.transAxes, va="bottom", ha="center",
            fontsize=11, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))
    ax.text(0.50, 0.98, lbl["top"],   transform=ax.transAxes, va="top", ha="center",
            fontsize=11, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))

def _to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    m = np.nanmax(x)
    if m > 0: x = x / m
    return (x*255.0 + 0.5).clip(0,255).astype(np.uint8)

def _save_gif(frames_paths, out_gif, fps=12):
    imgs = [imageio.imread(p) for p in frames_paths if os.path.exists(p)]
    if imgs:
        imageio.mimsave(out_gif, imgs, fps=fps, loop=0)

def _scale_fixed01(x, vmin, vmax):
    x = np.asarray(x, np.float32)
    return np.clip((x - vmin) / (max(vmax - vmin, 1e-8)), 0, 1)


# --- cache de atlas ---
_ATLAS_VOL = {}
_ATLAS_RS  = {}  # clave: (atlas_path, subj_shape)

def _get_atlas_vol(atlas_path):
    if atlas_path not in _ATLAS_VOL:
        import nibabel as nib
        _ATLAS_VOL[atlas_path] = nib.load(atlas_path).get_fdata().astype(np.int16)
    return _ATLAS_VOL[atlas_path]

def _get_atlas_resampled(atlas_path, subj_shape):
    key = (atlas_path, tuple(map(int, subj_shape)))
    if key not in _ATLAS_RS:
        from skimage.transform import resize
        vol = _get_atlas_vol(atlas_path)
        Hs,Ws,Ds = map(int, subj_shape)
        _ATLAS_RS[key] = resize(vol, (Hs,Ws,Ds), order=0,
                                preserve_range=True, anti_aliasing=False).astype(np.int16)
    return _ATLAS_RS[key]

# -------------------- Bases --------------------
def _pick_bases_from_multichannel(img_1xCxHxW, mode="middle"):
    assert img_1xCxHxW.ndim == 4 and img_1xCxHxW.shape[0] == 1
    _, C, H, W = img_1xCxHxW.shape
    x = img_1xCxHxW.detach().float().cpu()
    def norm01(t):
        t = t - t.min()
        if t.max() > 0: t = t / t.max()
        return t
    if mode == "first":
        bases = [norm01(x[0,0:1]).numpy().squeeze(0)]; labels = ["first"]
    elif mode == "middle":
        m = C//2; bases = [norm01(x[0,m:m+1]).numpy().squeeze(0)]; labels = ["middle"]
    elif mode == "all":
        bases = [norm01(x[0,i:i+1]).numpy().squeeze(0) for i in range(C)]
        labels = [f"{i:02d}" for i in range(C)]
    elif mode == "mean":
        bases = [norm01(x.mean(dim=1, keepdim=True)).numpy().squeeze(0)]; labels=["mean"]
    elif mode == "max":
        bases = [norm01(x.max(dim=1, keepdim=True).values).numpy().squeeze(0)]; labels=["max"]
    else:
        raise ValueError(f"slice-mode desconocido: {mode}")
    return bases, labels

# -------------------- Overlay --------------------
def _overlay_and_save(img_1xHxW, heat_2d, path_png, alpha=0.45, title=None,
                      show_colorbar=True, cmap="seismic", vmin=-1, vmax=1,
                      mask_2d=None, apply_mask=True):
    # base
    if torch.is_tensor(img_1xHxW): base = _to_numpy01(img_1xHxW.squeeze(0))
    else: base = np.squeeze(np.asarray(img_1xHxW))
    if base.ndim != 2: raise ValueError(f"Base debe ser 2D, obtuve {base.shape}")
    # heat
    heat = np.squeeze(np.asarray(heat_2d, dtype=np.float32))
    if heat.ndim != 2: raise ValueError(f"Heat debe ser 2D, obtuve {heat.shape}")

    # orientación coherente
    base = _apply_display_orient(base, title.split(" (")[1].split(")")[0] if title else "axial")
    heat = _apply_display_orient(heat, title.split(" (")[1].split(")")[0] if title else "axial")
    

    # máscara opcional (por defecto, proyección máx del volumen)
    if apply_mask and mask_2d is not None:
        m = _apply_display_orient(mask_2d, title.split(" (")[1].split(")")[0] if title else "axial")
        m = (m > 0).astype(np.float32)
        base = base * m
        heat = np.where(m > 0, heat, np.nan)


    fig = plt.figure(figsize=(6,6)); ax = plt.gca()
    ax.imshow(base, cmap="gray", interpolation="bicubic")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 else mcolors.Normalize(vmin=vmin, vmax=vmax)
    im2 = ax.imshow(heat, cmap=cmap, interpolation="bilinear", alpha=alpha, norm=norm)
    _add_orient_labels(ax, title.split(" (")[1].split(")")[0] if title else "axial")
    if title: ax.set_title(title, fontsize=17, pad=4)
    ax.axis("off")
    if show_colorbar:
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=im2.get_cmap()),
                            ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
        ticks = [-1,-0.5,0,0.5,1] if vmin < 0 else [0,0.25,0.5,0.75,1]
        cbar.set_ticks(ticks); cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
        cbar.set_label("Contribución (− reduce edad, + aumenta edad)" if vmin<0 else "Nivel de relevancia", fontsize=9)
        cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=8); cbar.solids.set_alpha(1)
    fig.savefig(path_png, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_per_slice_overlays(image_1xCxHxW, heat_2d, outdir, prefix, alpha, title, signed=True,
                             mask_2d=None, apply_mask=True):
    x = image_1xCxHxW.detach().float().cpu()[0]
    C, H, W = x.shape
    outdir_slices = os.path.join(outdir, "per_slice"); os.makedirs(outdir_slices, exist_ok=True)
    for i in range(C):
        base_1xHxW = x[i:i+1].numpy()
        _overlay_and_save(
            base_1xHxW, heat_2d,
            os.path.join(outdir_slices, f"{prefix}_slice_{i:02d}.png"),
            alpha=alpha, title=f"{title} | slice {i:02d}",
            cmap=("seismic" if signed else "jet"),
            vmin=(-1 if signed else 0), vmax=1,
            mask_2d=mask_2d, apply_mask=apply_mask
        )

# -------------------- Montaje simple (sin colorbar) ------------
def _grid_save_no_colorbar(images_paths, out_path, title=None):
    paths = [p for p in images_paths if p is not None and os.path.exists(p)]
    if not paths: return
    fig_w = 12; fig_h = 4.2
    fig, axs = plt.subplots(1, len(paths), figsize=(fig_w, fig_h))
    if len(paths) == 1: axs = [axs]
    for ax, p in zip(axs, paths):
        img = mpimg.imread(p); ax.imshow(img); ax.axis("off")
    if title: fig.suptitle(title, fontsize=18, color=(0,0,0), y=0.98)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.08, wspace=0.02)
    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -------------------- Autoinferencia de checkpoints --------------------
def _open_state_dict(path, maploc):
    ckpt = torch.load(path, map_location=maploc)
    if hasattr(ckpt, "state_dict"): return ckpt.state_dict()
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt

def _infer_backbone_norm_inplace_nblock(path):
    sd = _open_state_dict(path, DEVICE)
    keys = list(sd.keys())
    is_resnet = any(("global_feat.stem" in k) or ("global_feat.layer1.0" in k) for k in keys)
    is_vgg16  = any(".conv33." in k or ".conv43." in k or ".conv53." in k for k in keys)
    backbone = "resnet18" if is_resnet else ("vgg16" if is_vgg16 else "vgg8")
    has_running = any(k.endswith("running_mean") or k.endswith("running_var") for k in keys)
    norm = "bn" if has_running else "gn"
    conv_in = []
    for k, v in sd.items():
        if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 4:
            conv_in.append(int(v.shape[1]))
    if not conv_in: raise RuntimeError("No conv2d weights found to infer inplace.")
    inplace = int(min(conv_in))
    idxs = {int(m.group(1)) for k in keys for m in [re.search(r"attnlist\.(\d+)\.", k)] if m}
    nblock = (max(idxs) + 1) if idxs else 0
    patch_size = step = None
    ckpt = torch.load(path, map_location="cpu")
    for key in ["hparams","config","meta","args","hyperparams"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            d = ckpt[key]
            patch_size = patch_size or d.get("patch_size")
            step = step or d.get("step")
    return {"backbone": backbone, "norm": norm, "inplace": inplace, "nblock": nblock,
            "patch_size": patch_size or 64, "step": step or 32, "state_dict": sd}

def _build_and_load_model(hp, path):
    model = GlobalLocalBrainAge(
        inplace=hp["inplace"], patch_size=hp["patch_size"], step=hp["step"],
        nblock=hp["nblock"], backbone=hp["backbone"], backbone_norm=hp["norm"],
        backbone_pretrained=False, backbone_freeze_bn=False,
    ).to(DEVICE)
    mu = model.load_state_dict(hp["state_dict"], strict=False)
    print(f"\nCargando: [{os.path.basename(path)}] norm={hp['norm']} missing={len(getattr(mu,'missing_keys',[]))} unexpected={len(getattr(mu,'unexpected_keys',[]))}")
    model.eval()
    return model

# -------------------- Grad-CAM firmado (global) --------------------
def _cnn_branch(model, which="global"):
    return model.global_feat if which=="global" else model.local_feat

def _find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _score_scalar(model, img, which="global"):
    outs = model(img)
    if which == "global" or len(outs) <= 1:
        return outs[0].view([])
    else:
        return torch.stack([o.view([]) for o in outs[1:]]).mean()

def gradcam_signed(model, image, which="global"):
    img = image.clone().detach().requires_grad_(True)
    target = _find_last_conv(_cnn_branch(model, which))
    if target is None:
        H, W = img.shape[-2:]
        return np.zeros((H, W), dtype=np.float32)
    acts, grads = [], []
    h1 = target.register_forward_hook(lambda m,i,o: acts.append(o))
    h2 = target.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    try:
        score = _score_scalar(model, img, which)
        model.zero_grad(set_to_none=True)
        score.backward()
        A = acts[-1][0]      # (C,h,w)
        G = grads[-1][0]     # (C,h,w)
        w = G.mean(dim=(1,2))
        cam = torch.einsum("c,chw->hw", w, A)  # firmado
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=img.shape[-2:], mode="bilinear", align_corners=False)[0,0]
        m = cam.abs().max()
        if m > 0: cam = cam / m  # [-1,1]
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()

# -------------------- IG firmado --------------------
def integrated_gradients_signed(model, image, steps=50, which="global", baseline="mean",
                                return_frames=False, base_for_overlay=None, mask_2d=None, plane="axial", alpha_overlay=0.45):
    model.eval()
    img = image.detach()
    if baseline == "zero":
        base = torch.zeros_like(img)
    elif baseline == "mean":
        base = img.mean(dim=(-2,-1), keepdim=True).expand_as(img)
    else:
        base = torch.zeros_like(img)

    
    alphas = torch.linspace(0, 1, steps, device=img.device)
    total = torch.zeros_like(img)
    frames = []  # cada entrada: dict(alpha, x_np, ig_map_np)

    for i, a in enumerate(alphas):
        x = (base + a * (img - base)).clone().detach().requires_grad_(True)
        score = _score_scalar(model, x, which=("global" if which!="local" else "local"))
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        if x.grad is not None:
            total += x.grad

        if return_frames:
            # IG acumulado hasta el step i
            avg_grad = total / float(i+1)
            ig_now = ((img - base) * avg_grad).sum(dim=1)[0]  # (H,W)
            m = ig_now.abs().max()
            if m > 0: ig_now = ig_now / m
            ig_np = ig_now.detach().cpu().numpy()

            # base para overlay: toma el canal central si no se pasó uno
            if base_for_overlay is None:
                base_for_overlay_np = _to_numpy01(img[0, img.shape[1]//2:img.shape[1]//2+1].detach().cpu())[0]
            else:
                base_for_overlay_np = base_for_overlay

            # máscara opcional
            if mask_2d is not None:
                msk = (mask_2d > 0).astype(np.float32)
                base_show = base_for_overlay_np * msk
            else:
                base_show = base_for_overlay_np

            frames.append({
            "alpha": float(a.item()),
            "x_raw": x.detach().cpu()[0, x.shape[1]//2].numpy(), 
            "ig_map": ig_np,
            "base_np": base_show
        })


    avg_grad = total / max(1, len(alphas))
    ig = (img - base) * avg_grad
    ig_map = ig.sum(dim=1)[0]
    m = ig_map.abs().max()
    if m > 0: ig_map = ig_map / m
    ig_np = ig_map.detach().cpu().squeeze().numpy()

    if return_frames:
        return ig_np, frames
    return ig_np


# -------------------- Occlusion firmado --------------------
@torch.no_grad()
def occlusion_signed(model, image, occ_patch=32, occ_stride=16, baseline="mean"):
    base_pred = _score_scalar(model, image, which="global").item()
    _, _, H, W = image.shape
    heat = torch.zeros((H, W), device=image.device)
    cover = torch.zeros((H, W), device=image.device)
    fill_value = 0.0 if baseline == "zero" else float(image.mean().item())
    for y in range(0, H - occ_patch + 1, occ_stride):
        for x in range(0, W - occ_patch + 1, occ_stride):
            img2 = image.clone()
            img2[..., y:y+occ_patch, x:x+occ_patch] = fill_value
            p = _score_scalar(model, img2, which="global").item()
            delta = p - base_pred  # firmado
            heat[y:y+occ_patch, x:x+occ_patch] += delta
            cover[y:y+occ_patch, x:x+occ_patch] += 1
    cover = torch.clamp(cover, min=1)
    heat = heat / cover
    m = heat.abs().max()
    if m > 0: heat = heat / m
    return heat.detach().cpu().squeeze().numpy()

# -------------------- Montajes con colorbar compartida ------------
def _grid_save_shared_colorbar_nobar(images_paths, out_path, title=None, signed=True):
    paths = [p for p in images_paths if p is not None and os.path.exists(p)]
    if not paths: return
    fig_w = 12; fig_h = 4.2; fig, axs = plt.subplots(1, len(paths), figsize=(fig_w, fig_h))
    if len(paths) == 1: axs = [axs]
    for ax, p in zip(axs, paths):
        img = mpimg.imread(p); ax.imshow(img); ax.axis("off")
    if title: fig.suptitle(title, fontsize=18, color=(0,0,0), y=0.98)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.20, wspace=-0.5)
    if signed:
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1); cmap="seismic"; ticks = [-1,-0.5,0,0.5,1]
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1); cmap="jet"; ticks = [0,0.25,0.5,0.75,1]
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.14, 0.12, 0.72, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks); cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
    cbar.set_label("Contribución (− reduce edad, + aumenta edad)" if signed else "Nivel de relevancia", fontsize=9)
    cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=8); cbar.solids.set_alpha(1)
    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _collect_paths(prefix, plane_outdirs):
    paths = []
    for pdir in plane_outdirs:
        picked = None
        for name in (f"{prefix}_middle_nobar.png", f"{prefix}_mean_nobar.png",
                     f"{prefix}_first_nobar.png", f"{prefix}_max_nobar.png"):
            cand = os.path.join(pdir, name)
            if os.path.exists(cand): picked = cand; break
        if picked is None:
            for name in (f"{prefix}_middle.png", f"{prefix}_mean.png",
                         f"{prefix}_first.png", f"{prefix}_max.png"):
                cand = os.path.join(pdir, name)
                if os.path.exists(cand): picked = cand; break
        if picked is None:
            per_slice = os.path.join(pdir, "per_slice")
            if os.path.isdir(per_slice):
                cands = sorted([f for f in os.listdir(per_slice) if f.startswith(prefix) and f.endswith(".png")])
                if cands:
                    mid = cands[len(cands)//2]; picked = os.path.join(per_slice, mid)
        paths.append(picked)
    return paths

def _find_labels_csv(base_dir):
    cands = [
        os.path.join(base_dir, "../../data/atlases/combined_labels.csv"),
        os.path.join(base_dir, "../../data/aux_global/combined_labels.csv"),
        os.path.join(base_dir, "../../data/combined_labels.csv"),
    ]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError("combined_labels.csv no encontrado en data/{atlases,aux_global,.}")

def _load_labels_map(labels_csv_path) -> Dict[int,str]:
    m = {}
    with open(labels_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2: continue
            rid = int(row[0]); lab = str(row[1]).strip()
            m[rid] = lab
    return m


def _roi_metrics_from_heatmap(heat_hw: np.ndarray, atlas_5x: np.ndarray) -> Dict[int, Dict[str, float]]:
    assert heat_hw.ndim == 2 and atlas_5x.ndim == 3
    H, W = heat_hw.shape
    if atlas_5x.shape[1:] != (H, W):
        raise ValueError(f"Dim mismatch heat {heat_hw.shape} vs atlas {atlas_5x.shape}")

    stats = defaultdict(lambda: {
        "sum_pos": 0.0, "cnt_pos": 0,     # suma de valores > 0  (positiva)
        "sum_neg": 0.0, "cnt_neg": 0,     # suma de valores < 0  (negativa, es decir <= 0)
        "sum_abs": 0.0, "area_px": 0
    })

    # Acumular por ROI (en las 5 rebanadas del atlas)
    for s in range(atlas_5x.shape[0]):
        sl = atlas_5x[s]
        for rid in np.unique(sl):
            if rid == 0:
                continue
            mask = (sl == rid)
            vals = heat_hw[mask]
            if vals.size == 0:
                continue
            pos = vals[vals > 0]
            neg = vals[vals < 0]
            stats[rid]["sum_pos"] += float(pos.sum()) if pos.size else 0.0         # >= 0
            stats[rid]["cnt_pos"] += int(pos.size)
            stats[rid]["sum_neg"] += float(neg.sum()) if neg.size else 0.0         # <= 0 (negativo)
            stats[rid]["cnt_neg"] += int(neg.size)
            stats[rid]["sum_abs"] += float(np.abs(vals).sum())
            stats[rid]["area_px"] += int(vals.size)

    # Métricas finales por ROI
    out: Dict[int, Dict[str, float]] = {}
    for rid, d in stats.items():
        mean_pos = (d["sum_pos"] / d["cnt_pos"]) if d["cnt_pos"] > 0 else 0.0
        mean_neg = (d["sum_neg"] / d["cnt_neg"]) if d["cnt_neg"] > 0 else 0.0
        density  = (d["sum_abs"] / d["area_px"]) if d["area_px"] > 0 else 0.0

        # NUEVO: predominancia de signo
        mean_net = mean_pos + mean_neg
        pos_abs  = d["sum_pos"]                     # ya es positivo
        neg_abs  = -d["sum_neg"]                    # volver positivo
        denom    = pos_abs + neg_abs
        ratio_pos = (pos_abs / denom) if denom > 0 else 0.0

        out[rid] = {
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "mean_net": mean_net,                   # NUEVO
            "sum_abs": d["sum_abs"],
            "area_px": d["area_px"],
            "density_abs": density,
            "ratio_pos": ratio_pos,                 # NUEVO
            # para agregación global direccional:
            "sum_pos": d["sum_pos"],
            "cnt_pos": d["cnt_pos"],
            "sum_neg": d["sum_neg"],
            "cnt_neg": d["cnt_neg"],
        }
    return out


def _write_roi_csv(path_csv, metrics: Dict[int, Dict[str,float]], id2label: Dict[int,str], topk=None):
    rows = []
    for rid, d in metrics.items():
        mean_net  = d.get("mean_net", d.get("mean_pos", 0.0) + d.get("mean_neg", 0.0))
        density   = d.get("density_abs", (d["sum_abs"]/d["area_px"]) if d["area_px"]>0 else 0.0)
        ratio_pos = d.get("ratio_pos", 0.0)
        rows.append((
            rid,
            id2label.get(rid, f"ROI_{rid}"),
            d["mean_pos"],
            d["mean_neg"],
            mean_net,          # NUEVO
            d["sum_abs"],
            d["area_px"],
            density,
            ratio_pos          # NUEVO
        ))
    rows.sort(key=lambda t: t[7], reverse=True)  # ordenar por density_abs
    if topk is not None:
        rows = rows[:topk]

    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "roi_id","label",
            "mean_pos","mean_neg","mean_net",
            "sum_abs","area_px","density_abs","ratio_pos"
        ])
        for r in rows:
            w.writerow(r)
    return rows



def _draw_topk_contours(base_2d, heat_2d, atlas_middle_2d, ordered_rows, out_png,
                        k=10, alpha=0.45, signed=True, mask_2d=None, apply_mask=True, plane="axial"):
    rows = ordered_rows[:k]
    roi_ids = [r[0] for r in rows]

    base  = _apply_display_orient(base_2d,       plane)
    heat  = _apply_display_orient(heat_2d,       plane)
    atlas = _apply_display_orient(atlas_middle_2d, plane)

    if apply_mask and mask_2d is not None:
        m = _apply_display_orient(mask_2d, plane)
        m = (m > 0).astype(np.float32)
        base = base * m
        heat = np.where(m > 0, heat, np.nan)
    

    fig = plt.figure(figsize=(6,6)); ax = plt.gca()
    ax.imshow(base, cmap="gray", interpolation="bicubic")
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) if signed else mcolors.Normalize(vmin=0, vmax=1)
    ax.imshow(heat, cmap=("seismic" if signed else "jet"), alpha=alpha, norm=norm, interpolation="bilinear")
    _add_orient_labels(ax, plane)

    tab20 = cmaps.get_cmap("tab20")
    for i, rid in enumerate(roi_ids):
        mask = (atlas == rid).astype(np.uint8)
        if mask.sum() == 0: continue
        for c in measure.find_contours(mask, level=0.5):
            ax.plot(c[:,1], c[:,0], linewidth=1.5, color=tab20(i % 20), alpha=0.6)

    ax.axis("off")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -------------------- Carga sujeto / paths --------------------
def _load_subject_age(base_dir, subject_id, planes=("axial","coronal","sagittal")):
    for pl in planes:
        p = os.path.join(base_dir, f"../../data/processed/{pl}", f"{subject_id}.pt")
        if os.path.exists(p):
            try:
                sample = torch.load(p, map_location="cpu"); age = sample.get("age", None)
                if torch.is_tensor(age): return age.detach().cpu().item() if age.numel()==1 else None
                if isinstance(age, (int,float,np.number)): return float(age)
            except Exception:
                pass
    return None

def _resolve_model_path(plane, args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    explicit = getattr(args, f"{plane}_model", None)
    if explicit: return explicit, f"{plane}_custompath"
    pid = getattr(args, f"{plane}_id", None) or args.id
    if pid is None: raise ValueError(f"Debes especificar --{plane}-model o --{plane}-id o --id")
    return os.path.join(base_dir, f"../models/model_{plane}_{pid}.pt"), str(pid)

# -------------------- Ejecución por plano --------------------
def _fmt_title(kind, plane, age_txt, pred_txt, signed=True, params_txt=None, for_ensemble=False):
    base = f"{kind} ({plane}) | Pred {pred_txt}" if for_ensemble else f"{kind} ({plane}) | Real {age_txt} | Pred {pred_txt}"
    if params_txt: base += f" | {params_txt}"
    return base

def run_for_plane(plane, args, subject_id, ensemble_title=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, f"../../data/processed/{plane}")
    model_path, model_id = _resolve_model_path(plane, args)
    sample_path = os.path.join(data_dir, f"{subject_id}.pt")
    if not os.path.exists(model_path): raise FileNotFoundError(model_path)
    if not os.path.exists(sample_path): raise FileNotFoundError(sample_path)

    sample = torch.load(sample_path, map_location=DEVICE)
    image = sample["image"].unsqueeze(0).to(DEVICE)  # 1xCxHxW
    age = float(sample.get("age", -1))

    hp = _infer_backbone_norm_inplace_nblock(model_path)
    model = _build_and_load_model(hp, model_path)

    with torch.no_grad(): pred = model(image)[0].item()
    outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}"); os.makedirs(outdir, exist_ok=True)

    bases, labels = _pick_bases_from_multichannel(image, mode=args.slice_mode)
    age_txt, pred_txt = f"{age:.1f}", f"{pred:.1f}"

    # --------- Máscara tipo 'max' desde la propia imagen (sin atlas) ----------
    vol = sample["image"].detach().float().cpu()        # CxHxW
    mask_mid = (vol.max(dim=0).values > 0).numpy().astype(np.uint8)  # (H,W)
    use_mask = (not args.no_mask)

    # --------- IG firmado ----------
    base_mid = image.detach().float().cpu()[0, image.shape[1]//2].numpy()

    if args.ig_record_steps:
        ig_dir = os.path.join(outdir, "ig_steps"); os.makedirs(ig_dir, exist_ok=True)
        ig_map, frames = integrated_gradients_signed(
            model, image, steps=args.ig_steps, which="global", baseline=args.ig_baseline,
            return_frames=True, base_for_overlay=base_mid, mask_2d=mask_mid, plane=plane, alpha_overlay=args.alpha
        )
        
        # escala fija tomada del slice medio de la entrada final
        mid = image.shape[1]//2
        ref = image.detach().cpu()[0, mid].numpy()
        vmin, vmax = np.percentile(ref, [1, 99])  # o usa ref.min(), ref.max()


        input_frames_paths, overlay_frames_paths = [], []
        for k, f in enumerate(frames):
            p_in = os.path.join(ig_dir, f"input_step_{k:03d}.png")
            x_vis = _scale_fixed01(f["x_raw"], vmin, vmax)          # ya está en [0,1] con escala fija
            arr = (_apply_display_orient(x_vis, plane) * 255.0 + 0.5).clip(0,255).astype(np.uint8)
            Image.fromarray(arr).save(p_in)
            input_frames_paths.append(p_in)

            p_ov = os.path.join(ig_dir, f"overlay_step_{k:03d}.png")
            _overlay_and_save(f["base_np"], f["ig_map"], p_ov, alpha=args.alpha,
                              title=None, show_colorbar=False, cmap="seismic", vmin=-1, vmax=1,
                              mask_2d=mask_mid, apply_mask=(not args.no_mask))
            overlay_frames_paths.append(p_ov)

        if args.ig_gif_kind in ("input","both"):
            _save_gif(input_frames_paths, os.path.join(ig_dir, f"ig_input_{args.ig_steps}steps.gif"), fps=args.ig_gif_fps)
        if args.ig_gif_kind in ("overlay","both"):
            _save_gif(overlay_frames_paths, os.path.join(ig_dir, f"ig_overlay_{args.ig_steps}steps.gif"), fps=args.ig_gif_fps)

        ig = ig_map
    else:
        ig = integrated_gradients_signed(model, image, steps=args.ig_steps, which="global", baseline=args.ig_baseline)


    np.save(os.path.join(outdir, "ig.npy"), ig)
    for base_i, lab in zip(bases, labels):
        _overlay_and_save(base_i, ig, os.path.join(outdir, f"ig_{lab}.png"), alpha=args.alpha,
                          title=_fmt_title("Integrated Gradients", plane, age_txt, pred_txt,
                                           params_txt=f"steps={args.ig_steps}, base={args.ig_baseline}"),
                          cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)
        _overlay_and_save(base_i, ig, os.path.join(outdir, f"ig_{lab}_nobar.png"), alpha=args.alpha,
                          title=_fmt_title("IG", plane, age_txt, pred_txt, for_ensemble=True),
                          show_colorbar=False, cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)
    _save_per_slice_overlays(
        image, ig, outdir, prefix="ig", alpha=args.alpha,
        title=f"Integrated Gradients ({plane}) | Real {age_txt} | Pred {pred_txt}",
        signed=True, mask_2d=mask_mid, apply_mask=use_mask
    )

    # --------- Occlusion firmado ----------
    occ = occlusion_signed(model, image, occ_patch=args.occ_patch, occ_stride=args.occ_stride,
                           baseline=args.occ_baseline)
    np.save(os.path.join(outdir, "occlusion.npy"), occ)
    for base_i, lab in zip(bases, labels):
        _overlay_and_save(base_i, occ, os.path.join(outdir, f"occlusion_{lab}.png"), alpha=args.alpha,
                          title=_fmt_title("Occlusion", plane, age_txt, pred_txt,
                                           params_txt=f"p={args.occ_patch}, s={args.occ_stride}, base={args.occ_baseline}"),
                          cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)
        _overlay_and_save(base_i, occ, os.path.join(outdir, f"occlusion_{lab}_nobar.png"), alpha=args.alpha,
                          title=_fmt_title("Occlusion", plane, age_txt, pred_txt, for_ensemble=True),
                          show_colorbar=False, cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)
    _save_per_slice_overlays(
       image, occ, outdir, prefix="occlusion", alpha=args.alpha,
       title=f"Occlusion ({plane}) | Real {age_txt} | Pred {pred_txt}",
       signed=True, mask_2d=mask_mid, apply_mask=use_mask
    )

    # ----- Atlas + métricas por ROI (solo para CSV/contornos, NO para la máscara de overlays) -----
    labels_csv = _find_labels_csv(base_dir)
    id2label = _load_labels_map(labels_csv)
    meta = sample.get("meta", {})
    if not meta or "orig_shape" not in meta or "slice_indices" not in meta or plane not in meta["slice_indices"]:
        raise RuntimeError(f"{subject_id}: falta meta.orig_shape/slice_indices[{plane}] en el .pt")
    subj_shape = tuple(meta["orig_shape"])
    slice_idxs = meta["slice_indices"][plane]

    atlas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../data/atlases/combined_subcortical_cerebellum_1mm.nii.gz")
    atlas_5  = atlas_slices_matched(atlas_path, subj_shape, plane, slice_idxs)
    mid_idx  = len(atlas_5)//2
    atlas_mid = atlas_5[mid_idx]
    base_mid  = image.detach().float().cpu()[0, image.shape[1]//2].numpy()

    # --- DEBUG de co-registro (coloca aquí) ---
    dbg_base  = _apply_display_orient(base_mid,  plane)
    dbg_atlas = _apply_display_orient(atlas_mid, plane)
    
    plt.imsave(os.path.join(outdir, f"DEBUG_base_{plane}.png"), dbg_base, cmap="gray")
    
    from skimage import measure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(dbg_base, cmap="gray"); ax.axis("off")
    _add_orient_labels(ax, plane)
    cont = (dbg_atlas > 0).astype(np.uint8)
    for c in measure.find_contours(cont, 0.5):
        ax.plot(c[:,1], c[:,0], linewidth=1.2)
    fig.savefig(os.path.join(outdir, f"DEBUG_atlas_outline_{plane}.png"),
                dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # --- FIN DEBUG ---
    

    # IG
    metrics_ig  = _roi_metrics_from_heatmap(ig,  atlas_mid[None, ...])
    ordered_ig = _write_roi_csv(os.path.join(outdir, "roi_metrics_ig.csv"), metrics_ig, id2label, topk=None)
    _draw_topk_contours(base_mid, ig,  atlas_mid, ordered_ig,
        os.path.join(outdir, f"contours_ig_top{args.topk}.png"),
        k=args.topk, alpha=args.alpha, signed=True,
        mask_2d=mask_mid, apply_mask=use_mask, plane=plane)

    # Occlusion
    metrics_occ = _roi_metrics_from_heatmap(occ, atlas_mid[None, ...])
    ordered_occ = _write_roi_csv(os.path.join(outdir, "roi_metrics_occlusion.csv"), metrics_occ, id2label, topk=None)
    _draw_topk_contours(base_mid, occ, atlas_mid, ordered_occ,
        os.path.join(outdir, f"contours_occlusion_top{args.topk}.png"),
        k=args.topk, alpha=args.alpha, signed=True,
        mask_2d=mask_mid, apply_mask=use_mask, plane=plane)

    plane_metrics = {"ig": metrics_ig, "occlusion": metrics_occ}

    # --------- Grad-CAM firmado (GLOBAL) ----------
    if args.gradcam == "global":
        gcg = gradcam_signed(model, image, which="global")
        np.save(os.path.join(outdir, "gradcam_global.npy"), gcg)
        for base_i, lab in zip(bases, labels):
            _overlay_and_save(base_i, gcg, os.path.join(outdir, f"gradcam_global_{lab}.png"),
                              alpha=args.alpha, title=_fmt_title("Grad-CAM GLOBAL", plane, age_txt, pred_txt),
                              cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)
            _overlay_and_save(base_i, gcg, os.path.join(outdir, f"gradcam_global_{lab}_nobar.png"),
                              alpha=args.alpha, title=_fmt_title("Grad-CAM GLOBAL", plane, age_txt, pred_txt, for_ensemble=True),
                              show_colorbar=False, cmap="seismic", vmin=-1, vmax=1, mask_2d=mask_mid, apply_mask=use_mask)

    # --------- report.json mínimo ----------
    report = {
        "subject_id": subject_id,
        "plane": plane,
        "model_id": model_id,
        "timestamp": int(time.time()),
        "prediction": float(pred),
        "age": float(age),
        "methods": {
            "ig_signed": {"steps": int(args.ig_steps), "baseline": args.ig_baseline},
            "occlusion_signed": {"patch": int(args.occ_patch), "stride": int(args.occ_stride), "baseline": args.occ_baseline},
            "gradcam": args.gradcam
        },
        "viz": {
            "ig_png": [f"ig_{lab}.png" for lab in labels],
            "occ_png": [f"occlusion_{lab}.png" for lab in labels],
            "gradcam_global_png": ([f"gradcam_global_{lab}.png" for lab in labels] if args.gradcam == "global" else [])
        },
        "hparams_inferidos": {k: _py(v) for k,v in hp.items() if k != "state_dict"}
    }
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --------- Montaje por plano ----------
    for kind, signed in [("ig", True), ("occlusion", True)]:
        paths = _collect_paths(kind, [outdir])
        if any(p is not None for p in paths):
            _grid_save_shared_colorbar_nobar(paths, os.path.join(outdir, f"{kind}_montage.png"),
                                             title=ensemble_title, signed=signed)

    print(f"[{plane}] {subject_id} | Real: {age_txt} - Predicha: {pred_txt}")
    print(f"   Guardado en: {outdir}\n")
    return outdir, plane_metrics

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Explicabilidad clínica mínima: IG firmado + Occlusion firmado + Grad-CAM global")
    # ids / rutas
    ap.add_argument("--id", help="ID común (opcional)")
    ap.add_argument("--axial-id"); ap.add_argument("--coronal-id"); ap.add_argument("--sagittal-id")
    ap.add_argument("--axial-model"); ap.add_argument("--coronal-model"); ap.add_argument("--sagittal-model")

    # datos / filtro
    ap.add_argument("--planes", nargs="+", default=["axial","coronal","sagittal"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--age-min", type=float, default=None)
    ap.add_argument("--age-max", type=float, default=None)

    # visual
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--slice-mode", choices=["first","middle","all","mean","max"], default="middle")
    ap.add_argument("--outdir", default="./explanations_clinical")

    # IG
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--ig-baseline", choices=["zero","mean"], default="mean")
    ap.add_argument("--ig-record-steps", action="store_true",
                    help="Guarda frames por step desde el baseline hasta la entrada y genera GIF")
    ap.add_argument("--ig-gif-fps", type=int, default=12, help="FPS del GIF de IG")
    ap.add_argument("--ig-gif-kind", choices=["input","overlay","both"], default="overlay",
                    help="input: solo interpolación; overlay: IG acumulado sobre base; both: dos GIFs")

    # Occlusion
    ap.add_argument("--occ-patch", type=int, default=32)
    ap.add_argument("--occ-stride", type=int, default=16)
    ap.add_argument("--occ-baseline", choices=["zero","mean"], default="mean")

    # Grad-CAM
    ap.add_argument("--gradcam", choices=["global","none"], default="global")

    # Ensemble titles
    ap.add_argument("--ensemble-preds-dir", default="./predictions", help="Carpeta con val_true_ensemble.txt y val_pred_ensemble.txt")
    ap.add_argument("--prefer-perfect-ensemble", action="store_true")
    ap.add_argument("--perfect-tol", type=float, default=0.0)

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no-mask", action="store_true", help="Desactiva el enmascarado de PNGs finales")



    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    val_file = os.path.join(base_dir, "../IDs/val_ids.txt")
    with open(val_file, "r") as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]
    with open(val_file, "r") as f:
        all_val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # cargar ensemble para título (si existe)
    ens_true_arr = ens_pred_arr = None
    try:
        ens_true_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_true_ensemble.txt"))
        ens_pred_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_pred_ensemble.txt"))
        if np.isscalar(ens_true_arr): ens_true_arr = np.array([float(ens_true_arr)])
        if np.isscalar(ens_pred_arr): ens_pred_arr = np.array([float(ens_pred_arr)])
    except Exception as e:
        print(f"[WARN] No pude cargar ensemble .txt desde {args.ensemble_preds_dir}: {e}")

    # filtro por edad
    if args.age_min is not None or args.age_max is not None:
        filtered=[]
        for sid in val_ids:
            age_val = _load_subject_age(base_dir, sid)
            if age_val is None: continue
            if args.age_min is not None and age_val < args.age_min: continue
            if args.age_max is not None and age_val > args.age_max: continue
            filtered.append(sid)
        val_ids = filtered

    # filtro opcional por "perfect ensemble"
    if args.prefer_perfect_ensemble and ens_true_arr is not None and ens_pred_arr is not None:
        candidates = []
        for sid, true, pred in zip(all_val_ids, ens_true_arr, ens_pred_arr):
            if abs(true - pred) <= args.perfect_tol:
                candidates.append(sid)
        if candidates:
            val_ids = [sid for sid in val_ids if sid in candidates]

    # selección final
    ids_to_use = val_ids[:args.n]
    print("\nPlanos:", ", ".join(args.planes))
    print("Sujetos:", ", ".join(ids_to_use) if ids_to_use else "(ninguno)")
    print(f"IG: steps={args.ig_steps}, baseline={args.ig_baseline}")
    print(f"Occlusion: patch={args.occ_patch}, stride={args.occ_stride}, baseline={args.occ_baseline}")
    print(f"Grad-CAM: {args.gradcam}")

    if not (args.id or args.axial_id or args.coronal_id or args.sagittal_id or
            args.axial_model or args.coronal_model or args.sagittal_model):
        raise SystemExit("Debes especificar al menos --id o --<plane>-id o --<plane>-model.")

    for subject_id in ids_to_use:
        title = None
        if ens_true_arr is not None and ens_pred_arr is not None and subject_id in all_val_ids:
            try:
                idx_all = all_val_ids.index(subject_id)
                if 0 <= idx_all < len(ens_true_arr) and 0 <= idx_all < len(ens_pred_arr):
                    ens_true = float(ens_true_arr[idx_all]); ens_pred = float(ens_pred_arr[idx_all])
                    title = f"Ensemble Pred: {ens_pred:.1f} | Real {ens_true:.1f} | BAG {ens_pred-ens_true:+.1f}"
            except ValueError:
                pass

        plane_outdirs = []
        for plane in args.planes:
            outdir, plane_metrics = run_for_plane(plane, args, subject_id, ensemble_title=title)
            plane_outdirs.append(outdir)
            if "global_acc" not in locals():
                global_acc = {
                    "ig": defaultdict(lambda: {"sum_abs":0.0,"sum_pos":0.0,"cnt_pos":0,"sum_neg":0.0,"cnt_neg":0,"area_px":0}),
                    "occlusion": defaultdict(lambda: {"sum_abs":0.0,"sum_pos":0.0,"cnt_pos":0,"sum_neg":0.0,"cnt_neg":0,"area_px":0})
                }
            for method in ("ig","occlusion"):
                for rid, d in plane_metrics[method].items():
                    g = global_acc[method][rid]
                    # agregados globales (incluye direccionalidad)
                    g["sum_abs"] += d["sum_abs"]
                    g["area_px"] += d["area_px"]
                    # nuevos: acumular sumas/contadores para medias globales
                    if "sum_pos" in d: g["sum_pos"] += d["sum_pos"]
                    if "cnt_pos" in d: g["cnt_pos"] += d["cnt_pos"]
                    if "sum_neg" in d: g["sum_neg"] += d["sum_neg"]
                    if "cnt_neg" in d: g["cnt_neg"] += d["cnt_neg"]

        # Montajes ensemble (junta los 3 planos)
        ens_dir = os.path.join(args.outdir, f"ensemble_{subject_id}"); os.makedirs(ens_dir, exist_ok=True)
        for kind, signed in [("ig", True), ("occlusion", True)]:
            paths = _collect_paths(kind, plane_outdirs)
            if any(p is not None for p in paths):
                _grid_save_shared_colorbar_nobar(paths, os.path.join(ens_dir, f"ensemble_{kind}.png"),
                                                 title=title, signed=signed)

        if "global" == args.gradcam:
            paths = _collect_paths("gradcam_global", plane_outdirs)
            if any(p is not None for p in paths):
                _grid_save_shared_colorbar_nobar(
                    paths, os.path.join(ens_dir, "ensemble_gradcam_global.png"),
                    title=title, signed=True
                )

        # ----- Global: combinar planos por sum_abs y calcular medias direccionales -----
        if plane_outdirs:
            labels_csv = _find_labels_csv(base_dir)
            id2label = _load_labels_map(labels_csv)

            def _finalize_global(method_name):
                rows = []
                for rid, d in global_acc[method_name].items():
                    mean_pos_g = (d["sum_pos"]/d["cnt_pos"]) if d["cnt_pos"]>0 else 0.0
                    mean_neg_g = (d["sum_neg"]/d["cnt_neg"]) if d["cnt_neg"]>0 else 0.0
                    mean_net_g = mean_pos_g + mean_neg_g                      # NUEVO
                    density_g  = (d["sum_abs"]/d["area_px"]) if d["area_px"]>0 else 0.0
                    pos_abs_g  = d["sum_pos"]
                    neg_abs_g  = -d["sum_neg"]
                    denom_g    = pos_abs_g + neg_abs_g
                    ratio_pos_g = (pos_abs_g/denom_g) if denom_g>0 else 0.0   # NUEVO

                    rows.append((
                        rid, id2label.get(rid, f"ROI_{rid}"),
                        mean_pos_g, mean_neg_g, mean_net_g,        # NUEVO
                        d["sum_abs"], d["area_px"], density_g, ratio_pos_g   # NUEVO
                    ))

                rows.sort(key=lambda t: t[7], reverse=True)  # por density_abs
                os.makedirs(ens_dir, exist_ok=True)

                # preparar dict para _write_roi_csv
                out_dict = {
                    r[0]: {
                        "mean_pos": r[2], "mean_neg": r[3], "mean_net": r[4],   # NUEVO
                        "sum_abs": r[5], "area_px": r[6],
                        "density_abs": r[7], "ratio_pos": r[8]                  # NUEVO
                    }
                    for r in rows
                }
                _ = _write_roi_csv(os.path.join(ens_dir, f"roi_metrics_{method_name}_global.csv"),
                                   out_dict, id2label, topk=None)


                # --- Contours del ensemble en un SOLO PNG con los 3 planos ---
                tmp_paths = []
                for plane, p_outdir in zip(args.planes, plane_outdirs):
                    sample = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     f"../../data/processed/{plane}", f"{subject_id}.pt"),
                                        map_location="cpu")
                    meta = sample["meta"]
                    subj_shape = tuple(meta["orig_shape"])
                    slice_idxs = meta["slice_indices"][plane]

                    # atlas matcheado por sujeto y mismo slice
                    atlas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "../../data/atlases/combined_subcortical_cerebellum_1mm.nii.gz")
                    atlas_5   = atlas_slices_matched(atlas_path, subj_shape, plane, slice_idxs)
                    atlas_mid = atlas_5[len(atlas_5)//2]

                    base_mid = sample["image"][sample["image"].shape[0]//2].numpy()
                    heat = np.load(os.path.join(p_outdir, f"{method_name}.npy"))
                    mask_mid = (sample["image"].float().max(dim=0).values > 0).numpy().astype(np.uint8)

                    tmp_png = os.path.join(ens_dir, f".tmp_contours_{method_name}_{plane}.png")
                    _draw_topk_contours(base_mid, heat, atlas_mid, rows, tmp_png,
                                        k=args.topk, alpha=args.alpha, signed=True,
                                        mask_2d=mask_mid, apply_mask=(not args.no_mask), plane=plane)
                    tmp_paths.append(tmp_png)


                _grid_save_no_colorbar(
                    tmp_paths,
                    os.path.join(ens_dir, f"contours_{method_name}_global_top{args.topk}.png"),
                    title=title
                )
                # (opcional) mantener temporales por depuración.

            _finalize_global("ig")
            _finalize_global("occlusion")

    print("\nListo. Resultados en:", args.outdir)

if __name__ == "__main__":
    main()
