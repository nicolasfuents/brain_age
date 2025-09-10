#!/usr/bin/env python3
# ---------------------------------------------------------
# explain_ensemble_planes.py
# Explica un ensemble por plano (IDs o rutas distintas)
#   - Patch heatmap
#   - Grad-CAM (local y/o global)
#   - Occlusion
# Autoinfiere hparams y normalización desde cada checkpoint.
# Títulos del ENSEMBLE leídos de: val_true_ensemble.txt y val_pred_ensemble.txt
# Ejemplo:
# python explain_ensemble_planes.py --axial-id 20250822-1437 --coronal-id 20250814-0853 --sagittal-id 20250822-1437 --planes axial coronal sagittal --n 3 --age-min 60 --gradcam ambos --slice-mode mean --occ-patch 32 --occ-stride 16 --alpha 0.45 --outdir ./explanations_gradcam_oclussion_patchheatmap
# ---------------------------------------------------------

import os, sys, re, argparse, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# Importa el modelo desde el root del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = mcolors.Normalize(vmin=0, vmax=1)

# -------------------- Utils pequeños --------------------
def _py(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def _to_numpy01(t):
    t = t.detach().float().cpu()
    t = t - t.min()
    if t.max() > 0:
        t = t / t.max()
    return t.numpy()

def custom_fmt(x, pos):
    if abs(x - 0) < 1e-8 or abs(x - 1) < 1e-8:
        return f"{int(x)}"
    return f"{x:.2f}"

# -------------------- Base/máscara cerebro --------------------
def _pick_bases_from_multichannel(img_1xCxHxW, mode="middle"):
    assert img_1xCxHxW.ndim == 4 and img_1xCxHxW.shape[0] == 1
    _, C, H, W = img_1xCxHxW.shape
    x = img_1xCxHxW.detach().float().cpu()

    def norm01(t):
        t = t - t.min()
        if t.max() > 0:
            t = t / t.max()
        return t

    if mode == "first":
        bases = [norm01(x[0, 0:1]).numpy().squeeze(0)]; labels = ["first"]
    elif mode == "middle":
        m = C // 2
        bases = [norm01(x[0, m:m+1]).numpy().squeeze(0)]; labels = ["middle"]
    elif mode == "all":
        bases = [norm01(x[0, i:i+1]).numpy().squeeze(0) for i in range(C)]
        labels = [f"{i:02d}" for i in range(C)]
    elif mode == "mean":
        mean_base = norm01(x.mean(dim=1, keepdim=True)).numpy().squeeze(0)
        bases = [mean_base]; labels = ["mean"]
    elif mode == "max":
        max_base = norm01(x.max(dim=1, keepdim=True).values).numpy().squeeze(0)
        bases = [max_base]; labels = ["max"]
    else:
        raise ValueError(f"slice-mode desconocido: {mode}")
    return bases, labels

def _make_brain_mask_from_base(base_2d, thr=0.05):
    b = base_2d
    mask = (b > thr).astype(np.float32)
    return mask

def _overlay_and_save(img_1xHxW, heat_2d, path_png, alpha=0.45, title=None,
                      brain_mask=None, show_colorbar=True, bottom_pad_with_bar=0.18, bottom_pad_no_bar=0.02):
    # --- base en 2D ---
    if torch.is_tensor(img_1xHxW):
        base = _to_numpy01(img_1xHxW.squeeze(0))
    else:
        arr = np.asarray(img_1xHxW)
        base = np.squeeze(arr)  # (H,W)
    if base.ndim != 2:
        raise ValueError(f"Base debe ser 2D, obtuve {base.shape}")

    # --- heat en 2D ---
    heat = np.asarray(heat_2d, dtype=np.float32)
    heat = np.squeeze(heat)
    if heat.ndim != 2:
        raise ValueError(f"Heat debe ser 2D, obtuve {heat.shape}")

    # Rotaciones coherentes
    base = np.rot90(base, k=1)
    heat = np.rot90(heat, k=1)

    # --- máscara opcional ---
    if brain_mask is not None:
        m = np.asarray(brain_mask)
        m = np.squeeze(m)
        if m.ndim == 3:
            m = m[0] if m.shape[0] == 1 else m[..., 0]
        if m.ndim != 2:
            raise ValueError(f"Mask debe ser 2D, obtuve {m.shape}")
        m = np.rot90(m, k=1)
        m = m > 0

        if m.shape != heat.shape:
            H, W = heat.shape
            Mh, Mw = m.shape
            m_fixed = np.zeros_like(heat, dtype=bool)
            h = min(H, Mh); w = min(W, Mw)
            m_fixed[:h, :w] = m[:h, :w]
            m = m_fixed

        if np.any(m):
            vmax = np.nanmax(heat[m])
            if vmax > 0:
                heat = heat / vmax
            heat = np.where(m, heat, np.nan)
        else:
            if heat.max() > 0:
                heat = heat / heat.max()
    else:
        heat = heat - np.nanmin(heat)
        if np.nanmax(heat) > 0:
            heat = heat / np.nanmax(heat)

    # --- Plot ---
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im  = ax.imshow(base, cmap="gray", interpolation="bicubic")
    im2 = ax.imshow(np.ma.masked_invalid(heat), cmap="jet",
                    interpolation="bilinear", alpha=alpha, vmin=0, vmax=1)

    if title: ax.set_title(title, fontsize=17, pad=4)
    ax.axis("off")

    if show_colorbar:
        cax = inset_axes(
            ax, width="100%", height="3%", loc="lower center",
            bbox_to_anchor=(0, -0.10, 1, 1), bbox_transform=ax.transAxes, borderpad=0
        )
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=im2.get_cmap())
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
        cbar.set_label("Nivel de relevancia", fontsize=9)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8)
        cbar.solids.set_alpha(1)
        cbar.ax.set_facecolor((0,0,0,0))
        cbar.ax.patch.set_alpha(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_with_bar)
    else:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_no_bar)


    # 1) ocupa TODO el ancho de la figura
    # 2) deja solo el espacio vertical que necesita el colorbar
    # (left/right/top=0/1 quitan márgenes laterales y superiores)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0.18)

    # quita márgenes de datos dentro del Axes (por si acaso)
    ax.margins(x=0, y=0)

    # guardado sin padding extra alrededor del bbox total (contenido+colorbar)
    fig.savefig(path_png, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# === Guardado por-cada-slice (sin alterar nada más) ===
def _save_per_slice_overlays(image_1xCxHxW, heat_2d, outdir, prefix, alpha, title):
    x = image_1xCxHxW.detach().float().cpu()[0]  # CxHxW
    C, H, W = x.shape
    outdir_slices = os.path.join(outdir, "per_slice")
    os.makedirs(outdir_slices, exist_ok=True)
    for i in range(C):
        base_1xHxW = x[i:i+1].numpy()
        mask_i = _make_brain_mask_from_base(base_1xHxW.squeeze(0))
        _overlay_and_save(
            base_1xHxW, heat_2d,
            os.path.join(outdir_slices, f"{prefix}_slice_{i:02d}.png"),
            alpha=alpha, title=f"{title} | slice {i:02d}",
            brain_mask=mask_i
        )

# -------------------- Autoinferencia desde checkpoint --------------------
def _open_state_dict(path, maploc):
    ckpt = torch.load(path, map_location=maploc)
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
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
    if not conv_in:
        raise RuntimeError("No conv2d weights found to infer inplace.")
    inplace = int(min(conv_in))

    idxs = {int(m.group(1)) for k in keys for m in [re.search(r"attnlist\.(\d+)\.", k)] if m}
    nblock = (max(idxs) + 1) if idxs else 0

    # patch_size/step si hay metadatos
    patch_size = step = None
    ckpt = torch.load(path, map_location="cpu")
    for key in ["hparams", "config", "meta", "args", "hyperparams"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            d = ckpt[key]
            patch_size = patch_size or d.get("patch_size")
            step = step or d.get("step")

    return {
        "backbone": backbone, "norm": norm, "inplace": inplace, "nblock": nblock,
        "patch_size": patch_size or 64, "step": step or 32, "state_dict": sd,
    }

def _build_and_load_model(hp, path):
    model = GlobalLocalBrainAge(
        inplace=hp["inplace"], patch_size=hp["patch_size"], step=hp["step"],
        nblock=hp["nblock"], backbone=hp["backbone"], backbone_norm=hp["norm"],
        backbone_pretrained=False, backbone_freeze_bn=False,
    ).to(DEVICE)
    missing_unexpected = model.load_state_dict(hp["state_dict"], strict=False)
    mk = getattr(missing_unexpected, "missing_keys", [])
    uk = getattr(missing_unexpected, "unexpected_keys", [])
    print(f"  load [{os.path.basename(path)}] norm={hp['norm']} missing={len(mk)} unexpected={len(uk)}")
    model.eval()
    return model

# -------------------- Targets Grad-CAM robustos --------------------
def _find_last_conv(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def _pick_cam_target(model, which="local", backbone="vgg16"):
    if which == "local":
        if backbone == "vgg16" and hasattr(model.local_feat, "conv53"):
            return model.local_feat.conv53
        tgt = getattr(model.local_feat, "conv42", None)
        return tgt if isinstance(tgt, torch.nn.Module) else _find_last_conv(model.local_feat)
    else:  # global
        if backbone == "vgg16" and hasattr(model.global_feat, "conv53"):
            return model.global_feat.conv53
        tgt = getattr(model.global_feat, "conv42", None)
        return tgt if isinstance(tgt, torch.nn.Module) else _find_last_conv(model.global_feat)

# -------------------- Mapas de explicabilidad --------------------
def patch_heatmap(model, image, patch_size=64, step=32):
    model.eval()
    with torch.no_grad():
        outs = model(image)
    scores = outs[1:]  # locales
    _, _, H, W = image.shape
    heat = torch.zeros((H, W), device=image.device)
    patch_idx = 0
    for y in range(0, H - patch_size, step):
        for x in range(0, W - patch_size, step):
            if patch_idx >= len(scores):
                continue
            val = float(scores[patch_idx])
            heat[y:y+patch_size, x:x+patch_size] += val
            patch_idx += 1
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.detach().cpu().numpy()

def _gradcam_generic(model, image, target_module):
    model.eval()
    acts, grads = [], []
    h1 = target_module.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_module.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        img = image.clone().detach().requires_grad_(True)
        outs = model(img)
        score = outs[0].view([])
        model.zero_grad(set_to_none=True)
        score.backward()
        A = acts[-1][0]          # (C,H,W)
        G = grads[-1][0]         # (C,H,W)
        weights = G.mean(dim=(1, 2))
        cam2d = torch.einsum("c,chw->hw", weights, A).relu()
        cam4d = F.interpolate(
            cam2d.unsqueeze(0).unsqueeze(0),
            size=img.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam4d[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()

def gradcam_local(model, image, backbone):
    target = _pick_cam_target(model, which="local", backbone=backbone)
    if target is None:
        return np.zeros(image.shape[-2:], dtype=np.float32)
    acts, grads = [], []
    h1 = target.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        img = image.clone().detach().requires_grad_(True)
        outs = model(img)
        if len(outs) <= 1:
            return np.zeros(img.shape[-2:], dtype=np.float32)
        score = torch.stack([o.view([]) for o in outs[1:]]).mean()
        model.zero_grad(set_to_none=True); score.backward()
        A = acts[-1][0]; G = grads[-1][0]; weights = G.mean(dim=(1, 2))
        cam2d = torch.einsum("c,chw->hw", weights, A).relu()
        cam4d = F.interpolate(cam2d.unsqueeze(0).unsqueeze(0),
                              size=img.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam4d[0, 0]; cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()

def gradcam_global(model, image, backbone):
    target = _pick_cam_target(model, which="global", backbone=backbone)
    if target is None:
        return np.zeros(image.shape[-2:], dtype=np.float32)
    return _gradcam_generic(model, image, target)

def occlusion_map(model, image, occ_patch=32, occ_stride=16):
    model.eval()
    with torch.no_grad():
        base = model(image)[0].item()
    _, _, H, W = image.shape
    occ = torch.zeros((H, W), device=image.device)
    for y in range(0, H - occ_patch + 1, occ_stride):
        for x in range(0, W - occ_patch + 1, occ_stride):
            img2 = image.clone()
            img2[..., y:y+occ_patch, x:x+occ_patch] = 0
            with torch.no_grad():
                p = model(img2)[0].item()
            drop = abs(p - base)
            occ[y:y+occ_patch, x:x+occ_patch] += drop
    occ = occ - occ.min()
    if occ.max() > 0:
        occ = occ / occ.max()
    return occ.detach().cpu().numpy()

# -------------------- Carga sujeto y ejecución por plano --------------------
def _load_subject_age(base_dir, subject_id, planes=("axial", "coronal", "sagittal")):
    for pl in planes:
        p = os.path.join(base_dir, f"../../data/processed/{pl}", f"{subject_id}.pt")
        if os.path.exists(p):
            try:
                sample = torch.load(p, map_location="cpu")
                age = sample.get("age", None)
                if torch.is_tensor(age):
                    return age.detach().cpu().item() if age.numel() == 1 else None
                if isinstance(age, (int, float, np.number)):
                    return float(age)
            except Exception:
                pass
    return None

def _resolve_model_path(plane, args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    explicit = getattr(args, f"{plane}_model", None)
    if explicit:
        return explicit, f"{plane}_custompath"
    pid = getattr(args, f"{plane}_id", None) or args.id
    if pid is None:
        raise ValueError(f"Debes especificar --{plane}-model o --{plane}-id o --id")
    return os.path.join(base_dir, f"../models/model_{plane}_{pid}.pt"), str(pid)

def run_for_plane(plane, args, subject_id):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, f"../../data/processed/{plane}")
    model_path, model_id = _resolve_model_path(plane, args)
    sample_path = os.path.join(data_dir, f"{subject_id}.pt")
    if not os.path.exists(model_path): raise FileNotFoundError(model_path)
    if not os.path.exists(sample_path): raise FileNotFoundError(sample_path)

    sample = torch.load(sample_path, map_location=DEVICE)
    image = sample["image"].unsqueeze(0).to(DEVICE)
    age = float(sample.get("age", -1))

    hp = _infer_backbone_norm_inplace_nblock(model_path)
    model = _build_and_load_model(hp, model_path)

    with torch.no_grad():
        pred = model(image)[0].item()

    outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}")
    os.makedirs(outdir, exist_ok=True)

    bases, labels = _pick_bases_from_multichannel(image, mode=args.slice_mode)
    masks = [_make_brain_mask_from_base(b) for b in bases]
    age_txt, pred_txt = f"{age:.1f}", f"{pred:.1f}"

    # Patch heatmap
    ph = patch_heatmap(model, image, patch_size=hp["patch_size"], step=hp["step"])
    np.save(os.path.join(outdir, "patch_heatmap.npy"), ph)
    for base_i, mask_i, lab in zip(bases, masks, labels):
        # con colorbar (standalone): mantiene real + pred
        _overlay_and_save(
            base_i, ph, os.path.join(outdir, f"patch_heatmap_{lab}.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Patch heatmap", plane, age_txt, pred_txt, for_ensemble=False),
            brain_mask=mask_i, show_colorbar=True
        )

        # SIN colorbar (para ensemble): solo pred
        _overlay_and_save(
            base_i, ph, os.path.join(outdir, f"patch_heatmap_{lab}_nobar.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Patch heatmap", plane, age_txt, pred_txt, for_ensemble=True),
            brain_mask=mask_i, show_colorbar=False
        )

    _save_per_slice_overlays(
        image, ph, outdir, prefix="patch_heatmap",
        alpha=args.alpha, title=f"Patch heatmap ({plane}) | Real {age_txt} | Pred {pred_txt}"
    )

    # Grad-CAM local
    if args.gradcam in ("local", "ambos"):
        cam_local = gradcam_local(model, image, hp["backbone"])
        np.save(os.path.join(outdir, "gradcam_local.npy"), cam_local)
        for base_i, mask_i, lab in zip(bases, masks, labels):
            _overlay_and_save(
            base_i, cam_local, os.path.join(outdir, f"gradcam_local_{lab}.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Grad-CAM LOCAL", plane, age_txt, pred_txt, for_ensemble=False),
            brain_mask=mask_i, show_colorbar=True
        )

        _overlay_and_save(
            base_i, cam_local, os.path.join(outdir, f"gradcam_local_{lab}_nobar.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Grad-CAM LOCAL", plane, age_txt, pred_txt, for_ensemble=True),
            brain_mask=mask_i, show_colorbar=False
        )

        _save_per_slice_overlays(
            image, cam_local, outdir, prefix="gradcam_local",
            alpha=args.alpha, title=f"Grad-CAM LOCAL ({hp['backbone']}) | Real {age_txt} | Pred {pred_txt}"
        )

    # Grad-CAM global
    if args.gradcam in ("global", "ambos"):
        cam_global = gradcam_global(model, image, hp["backbone"])
        np.save(os.path.join(outdir, "gradcam_global.npy"), cam_global)
        for base_i, mask_i, lab in zip(bases, masks, labels):
            _overlay_and_save(
            base_i, cam_global, os.path.join(outdir, f"gradcam_global_{lab}.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Grad-CAM GLOBAL", plane, age_txt, pred_txt, for_ensemble=False),
            brain_mask=mask_i, show_colorbar=True
        )

        _overlay_and_save(
            base_i, cam_global, os.path.join(outdir, f"gradcam_global_{lab}_nobar.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Grad-CAM GLOBAL", plane, age_txt, pred_txt, for_ensemble=True),
            brain_mask=mask_i, show_colorbar=False
        )


        _save_per_slice_overlays(
            image, cam_global, outdir, prefix="gradcam_global",
            alpha=args.alpha, title=f"Grad-CAM GLOBAL ({hp['backbone']}) | Real {age_txt} | Pred {pred_txt}"
        )

    # Occlusion
    occ = occlusion_map(model, image, occ_patch=args.occ_patch, occ_stride=args.occ_stride)
    np.save(os.path.join(outdir, "occlusion.npy"), occ)
    for base_i, mask_i, lab in zip(bases, masks, labels):
        _overlay_and_save(
            base_i, occ, os.path.join(outdir, f"occlusion_{lab}.png"),
            alpha=args.alpha,
            title=_fmt_plane_title(f"Occlusion (p={args.occ_patch}, s={args.occ_stride})",
                                   plane, age_txt, pred_txt, for_ensemble=False),
            brain_mask=mask_i, show_colorbar=True
        )

        _overlay_and_save(
            base_i, occ, os.path.join(outdir, f"occlusion_{lab}_nobar.png"),
            alpha=args.alpha,
            title=_fmt_plane_title("Occlusion", plane, age_txt, pred_txt, for_ensemble=True),
            brain_mask=mask_i, show_colorbar=False
        )

    _save_per_slice_overlays(
        image, occ, outdir, prefix="occlusion",
        alpha=args.alpha, title=f"Occlusion (p={args.occ_patch}, s={args.occ_stride}) | Real {age_txt} | Pred {pred_txt}"
    )

    summary = {
        "plane": plane, "model_id": model_id, "subject_id": subject_id,
        "edad_real": _py(age), "edad_predicha": _py(pred),
        "model_path": os.path.abspath(model_path),
        "hparams_inferidos": {k: _py(v) for k, v in hp.items() if k != "state_dict"},
        "gradcam": args.gradcam, "occ_patch": int(args.occ_patch), "occ_stride": int(args.occ_stride),
    }
    with open(os.path.join(outdir, "explain_config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{plane}] {subject_id} | Real: {age_txt} - Predicha: {pred_txt}")
    print(f"   Guardado en: {outdir}")

# Config por defecto del título del ENSEMBLE (podés cambiarlos acá):
TITLE_FONT_SIZE = 18
TITLE_COLOR_RGB = (0, 0, 0)  # negro

def _fmt_plane_title(kind, plane, age_txt, pred_txt, *, for_ensemble=False):
    """
    kind: 'Patch heatmap', 'Grad-CAM LOCAL', 'Grad-CAM GLOBAL', 'Occlusion (p=.., s=..)'
    plane: 'axial' | 'coronal' | 'sagittal'
    Sin backbone. Para ensemble (nobar) omite la edad real.
    """
    if for_ensemble:
        return f"{kind} ({plane}) | Pred {pred_txt}"
    else:
        return f"{kind} ({plane}) | Real {age_txt} | Pred {pred_txt}"

def _grid_save_shared_colorbar_nobar(images_paths, out_path, title=None,
                                     title_size=TITLE_FONT_SIZE, title_color=TITLE_COLOR_RGB,
                                     wspace=0.03, cbar_height=0.02, cbar_pad=0.06):
    """
    Monta 3 imágenes (ya recortadas y SIN colorbar) en una figura Matplotlib
    y agrega un ÚNICO colorbar horizontal compartido (0–1, cmap 'jet').
    """

    paths = [p for p in images_paths if p is not None and os.path.exists(p)]
    if not paths:
        return

    # tamaño aproximado: ancho proporcional al número de columnas
    fig_w = 12
    fig_h = 4.2
    fig, axs = plt.subplots(1, len(paths), figsize=(fig_w, fig_h))

    if len(paths) == 1:
        axs = [axs]

    for ax, p in zip(axs, paths):
        img = mpimg.imread(p)  # RGBA
        ax.imshow(img)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=title_size, color=title_color, y=0.98)

    # espacio horizontal entre paneles
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.20, wspace=-0.5)
    # colorbar compartido (leyenda independiente con el mismo mapeo)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap="jet")
    sm.set_array([])

    # eje para el colorbar (en coordenadas de figura)
    fig_left, fig_right = 0.015, 0.985
    avail = fig_right - fig_left           # 0.97
    width = 0.72
    left  = fig_left + (avail - width)/2   # centrado dentro de [0.015, 0.985]
    cax = fig.add_axes([left, 0.15, width, cbar_height]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
    cbar.set_label("Nivel de relevancia", fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    cbar.solids.set_alpha(1)

    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# === ENSEMBLE: helpers (solo imágenes, sin predicciones) =====================
def _safe_load_png(path):
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None
    


def _load_font(size=TITLE_FONT_SIZE):
    """
    Usa la misma fuente default de Matplotlib, pero escalable.
    """
    try:
        # Usa la fuente por defecto de Matplotlib y respeta el tamaño
        font_path = font_manager.findfont(font_manager.FontProperties(family="sans-serif"))
        return ImageFont.truetype(font_path, size)
    except Exception:
        # fallback
        return ImageFont.load_default()


def _grid_save(images, out_path, cols=3, title=None,
               title_size=TITLE_FONT_SIZE, title_color=TITLE_COLOR_RGB,
               gap=8, title_pad_top=None, title_pad_bottom=44):
    """
    Crea un grid simple con fondo transparente (RGBA).
    - Sin barra superior: el título se dibuja directamente sobre transparencia.
    - Los gaps entre imágenes también quedan transparentes.
    """

    def _load_font_strict(size):
        candidates = [
            os.path.join(mpl.get_data_path(), "fonts/ttf/DejaVuSans.ttf"),
            os.path.join(mpl.get_data_path(), "fonts/ttf/DejaVuSansDisplay.ttf"),
        ]
        try:
            ff = fm.findfont("DejaVu Sans", fallback_to_default=False)
            if ff and os.path.exists(ff):
                candidates.insert(0, ff)
        except Exception:
            pass
        candidates += ["arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for path in candidates:
            try:
                if path and os.path.exists(path):
                    return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # Normalización de imágenes (y pasarlas a RGBA para preservar transparencia del canvas)
    images = [im for im in images if im is not None]
    if not images:
        return
    W, H = images[0].size
    norm = []
    for im in images:
        if im.size != (W, H):
            im = im.resize((W, H), Image.BICUBIC)
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        norm.append(im)

    rows = math.ceil(len(norm) / cols)
    gap = 8

    # Medimos el título (si existe) para reservar alto extra transparente
    title_h = 0
    if title:
        font = _load_font_strict(title_size)
        tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        tmpd = ImageDraw.Draw(tmp)
        try:
            bbox = tmpd.textbbox((0, 0), title, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = tmpd.textsize(title, font=font)

        # pads por defecto (si no te pasan nada)
        default_pad = max(12, title_size // 5)
        pad_top = default_pad if title_pad_top is None else int(title_pad_top)
        pad_bottom = default_pad if title_pad_bottom is None else int(title_pad_bottom)

        # alto total reservado para el bloque del título
        title_h = pad_top + text_h + pad_bottom
    else:
        font = None
        text_w = text_h = 0
        pad_top = pad_bottom = 0
        title_h = 0


    grid_w = cols * W + (cols - 1) * gap
    grid_h = rows * H + (rows - 1) * gap

    # Canvas RGBA totalmente transparente
    canvas = Image.new("RGBA", (grid_w, grid_h + title_h), (0, 0, 0, 0))

    # Dibujar el grid (con gaps transparentes)
    for idx, im in enumerate(norm):
        r = idx // cols
        c = idx % cols
        x = c * (W + gap)
        y = title_h + r * (H + gap)
        canvas.paste(im, (x, y), im)  # usa el propio alpha de la imagen

    # Dibujar título (solo texto, sin barra)
    if title:
        draw = ImageDraw.Draw(canvas)
        x = (grid_w - text_w) // 2
        y = (title_h - (text_h)) // 2
        draw.text((x, y), title, fill=title_color + (255,), font=font)

    # Guardar como PNG (preserva transparencia)
    canvas.save(out_path, format="PNG")



def _save_ensemble_montages_for_subject(subject_id, plane_outdirs, gradcam_mode, out_root, title_suffix=None):
    """
    Junta las imágenes por plano ya generadas y arma un montaje por cada tipo:
      - ensemble_patch_heatmap.png
      - ensemble_gradcam_local.png (si aplica)
      - ensemble_gradcam_global.png (si aplica)
      - ensemble_occlusion.png
    """
    ens_dir = os.path.join(out_root, f"ensemble_{subject_id}")
    os.makedirs(ens_dir, exist_ok=True)

    def collect_paths(prefix):
        paths = []
        for pdir in plane_outdirs:
            picked = None
            # preferir archivos _nobar
            for name in (f"{prefix}_middle_nobar.png", f"{prefix}_mean_nobar.png",
                         f"{prefix}_first_nobar.png", f"{prefix}_max_nobar.png"):
                cand = os.path.join(pdir, name)
                if os.path.exists(cand):
                    picked = cand; break
            if picked is None:
                # fallback a normales
                for name in (f"{prefix}_middle.png", f"{prefix}_mean.png",
                             f"{prefix}_first.png", f"{prefix}_max.png"):
                    cand = os.path.join(pdir, name)
                    if os.path.exists(cand):
                        picked = cand; break
            if picked is None:
                # per_slice fallback
                per_slice = os.path.join(pdir, "per_slice")
                if os.path.isdir(per_slice):
                    cands = sorted([f for f in os.listdir(per_slice)
                                    if f.startswith(prefix) and f.endswith(".png")])
                    if cands:
                        mid = cands[len(cands)//2]
                        picked = os.path.join(per_slice, mid)
            paths.append(picked)
        return paths

    # Patch heatmap
    patch_paths = collect_paths("patch_heatmap")
    _grid_save_shared_colorbar_nobar(
        patch_paths,
        os.path.join(ens_dir, "ensemble_patch_heatmap.png"),
        title=title_suffix
    )

    # Grad-CAM local
    if gradcam_mode in ("local", "ambos"):
        gcl_paths = collect_paths("gradcam_local")
        _grid_save_shared_colorbar_nobar(
            gcl_paths,
            os.path.join(ens_dir, "ensemble_gradcam_local.png"),
            title=title_suffix
        )

    # Grad-CAM global
    if gradcam_mode in ("global", "ambos"):
        gcg_paths = collect_paths("gradcam_global")
        _grid_save_shared_colorbar_nobar(
            gcg_paths,
            os.path.join(ens_dir, "ensemble_gradcam_global.png"),
            title=title_suffix
        )

    # Occlusion
    occ_paths = collect_paths("occlusion")
    _grid_save_shared_colorbar_nobar(
        occ_paths,
        os.path.join(ens_dir, "ensemble_occlusion.png"),
        title=title_suffix
    )
# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Explicabilidad multi-plano (IDs/rutas por plano) con autoconfiguración")
    # IDs genéricos o por plano
    ap.add_argument("--id", help="ID común para los tres planos (opcional)")
    ap.add_argument("--axial-id"); ap.add_argument("--coronal-id"); ap.add_argument("--sagittal-id")
    # Rutas directas (opcionales)
    ap.add_argument("--axial-model"); ap.add_argument("--coronal-model"); ap.add_argument("--sagittal-model")

    ap.add_argument("--planes", nargs="+", default=["axial", "coronal", "sagittal"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--age-min", type=float, default=None)
    ap.add_argument("--age-max", type=float, default=None)
    ap.add_argument("--occ-patch", type=int, default=32)
    ap.add_argument("--occ-stride", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--gradcam", choices=["local", "global", "ambos", "none"], default="ambos")
    ap.add_argument("--slice-mode", choices=["first", "middle", "all", "mean", "max"], default="middle")
    ap.add_argument("--outdir", default="./explanations")

    # ÚNICO flag para títulos del ensemble (archivos .txt)
    ap.add_argument("--ensemble-preds-dir", default="./predictions",
                    help="Carpeta con val_true_ensemble.txt y val_pred_ensemble.txt")

    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    val_file = os.path.join(base_dir, "../IDs/val_ids.txt")
    with open(val_file, "r") as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # Guardar también los IDs originales (sin filtrar) para mapear índices de los .txt
    with open(val_file, "r") as f:
        all_val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # Cargar ensemble .txt (mismo flujo que tu script de figuras)
    ens_true_arr = ens_pred_arr = None
    try:
        ens_true_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_true_ensemble.txt"))
        ens_pred_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_pred_ensemble.txt"))
        if np.isscalar(ens_true_arr): ens_true_arr = np.array([float(ens_true_arr)])
        if np.isscalar(ens_pred_arr): ens_pred_arr = np.array([float(ens_pred_arr)])
    except Exception as e:
        print(f"[WARN] No pude cargar ensemble .txt desde {args.ensemble_preds_dir}: {e}")

    # Filtro por edad si corresponde
    if args.age_min is not None or args.age_max is not None:
        filtered = []
        for sid in val_ids:
            age_val = _load_subject_age(base_dir, sid)
            if age_val is None:
                continue
            if args.age_min is not None and age_val < args.age_min:
                continue
            if args.age_max is not None and age_val > args.age_max:
                continue
            filtered.append(sid)
        val_ids = filtered

    ids_to_use = val_ids[:args.n]
    print("\n🧠 Explicando planos:", ", ".join(args.planes))
    print("📄 Sujetos:", ", ".join(ids_to_use) if ids_to_use else "(ninguno)")
    print("🗺️ Grad-CAM:", args.gradcam)
    if args.age_min is not None or args.age_max is not None:
        amin = f"{args.age_min}" if args.age_min is not None else "-inf"
        amax = f"{args.age_max}" if args.age_max is not None else "+inf"
        print(f"🎯 Filtro edad: [{amin}, {amax}]")
    print("📐 Slice mode:", args.slice_mode)

    # Chequeo mínimos: algún id o modelo por plano
    if not (args.id or args.axial_id or args.coronal_id or args.sagittal_id or
            args.axial_model or args.coronal_model or args.sagittal_model):
        raise SystemExit("Debes especificar al menos --id global o --<plane>-id o --<plane>-model.")

    for subject_id in ids_to_use:
        plane_outdirs = []
        for plane in args.planes:
            run_for_plane(plane, args, subject_id)
            model_path, model_id = _resolve_model_path(plane, args)
            outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}")
            plane_outdirs.append(outdir)

        # Título con pred ensemble y edad real (desde .txt)
        title = None
        if ens_true_arr is not None and ens_pred_arr is not None:
            try:
                idx_all = all_val_ids.index(subject_id)
                if 0 <= idx_all < len(ens_true_arr) and 0 <= idx_all < len(ens_pred_arr):
                    ens_true = float(ens_true_arr[idx_all])
                    ens_pred = float(ens_pred_arr[idx_all])
                    title = f"Ensemble Pred: {ens_pred:.1f} | Real {ens_true:.1f}"
            except ValueError:
                pass

        # Montajes del ensemble (solo imágenes)
        _save_ensemble_montages_for_subject(
            subject_id=subject_id,
            plane_outdirs=plane_outdirs,
            gradcam_mode=args.gradcam,
            out_root=args.outdir,
            title_suffix=title
        )

    print("\n✅ Listo. Revisá la carpeta de salida:", args.outdir)

if __name__ == "__main__":
    main()
