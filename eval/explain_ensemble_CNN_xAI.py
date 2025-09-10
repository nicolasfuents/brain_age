#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------
# explain_ensemble_CNN_xAI.py
# Métodos xAI alternativos a LRP para la parte CNN:
#   - gradcampp, scorecam, ig_smoothgrad, guidedbp, deeplift
# Mantiene tu mismo flujo de I/O, autoinferencia de checkpoints
# por plano, overlays y montajes por sujeto.
# Ejemplo de uso:
# python explain_ensemble_CNN_xAI.py --axial-id 20250822-1437 --coronal-id 20250814-0853 --sagittal-id 20250822-1437 --planes axial coronal sagittal --n 3 --age-min 60 --which both --slice-mode mean --alpha 0.45 --methods gradcampp scorecam ig_smoothgrad guidedbp deeplift --outdir ./explanations_xai --ensemble-preds-dir ./predictions
# ---------------------------------------------------------

import os, sys, re, json, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy 

# ===== importar tu modelo desde el repo raíz =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Utils compartidos --------------------
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
    if abs(x-0) < 1e-8 or abs(x-1) < 1e-8: return f"{int(x)}"
    return f"{x:.2f}"

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

def _make_brain_mask_from_base(base_2d, thr=0.05):
    return (np.asarray(base_2d) > thr).astype(np.float32)

def _overlay_and_save(img_1xHxW, heat_2d, path_png, alpha=0.45, title=None,
                      brain_mask=None, show_colorbar=True, bottom_pad_with_bar=0.18,
                      bottom_pad_no_bar=0.02, cmap="jet", vmin=0, vmax=1):
    # base
    if torch.is_tensor(img_1xHxW): base = _to_numpy01(img_1xHxW.squeeze(0))
    else: base = np.squeeze(np.asarray(img_1xHxW))
    if base.ndim != 2: raise ValueError(f"Base debe ser 2D, obtuve {base.shape}")
    # heat
    heat = np.squeeze(np.asarray(heat_2d, dtype=np.float32))
    if heat.ndim != 2: raise ValueError(f"Heat debe ser 2D, obtuve {heat.shape}")

    # orientar igual que tus otros scripts
    base = np.rot90(base, k=1)
    heat = np.rot90(heat, k=1)

    # Máscara + normalización
    if brain_mask is not None:
        m = np.rot90(np.squeeze(np.asarray(brain_mask)), k=1) > 0
        if m.shape != heat.shape:
            H,W = heat.shape; Mh,Mw = m.shape
            mf = np.zeros_like(heat, dtype=bool); h=min(H,Mh); w=min(W,Mw)
            mf[:h,:w] = m[:h,:w]; m = mf
        if np.any(m):
            hs = np.where(m, heat, np.nan)
            vmax_in = np.nanmax(hs)
            if vmax_in > 0: heat = heat / vmax_in
            heat = np.where(m, heat, np.nan)
    else:
        heat = heat - np.nanmin(heat)
        mx = np.nanmax(heat)
        if mx > 0: heat = heat / mx

    fig = plt.figure(figsize=(6,6)); ax = plt.gca()
    ax.imshow(base, cmap="gray", interpolation="bicubic")
    im2 = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, interpolation="bilinear",
                    alpha=alpha, vmin=vmin, vmax=vmax)
    if title: ax.set_title(title, fontsize=17, pad=4)
    ax.axis("off")
    if show_colorbar:
        cax = inset_axes(ax, width="100%", height="3%", loc="lower center",
                         bbox_to_anchor=(0, -0.10, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=im2.get_cmap()); sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([0,0.25,0.5,0.75,1]); cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
        cbar.set_label("Nivel de relevancia", fontsize=9)
        cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=8); cbar.solids.set_alpha(1)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_with_bar)
    else:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_no_bar)
    ax.margins(x=0, y=0)
    fig.savefig(path_png, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_per_slice_overlays(image_1xCxHxW, heat_2d, outdir, prefix, alpha, title):
    x = image_1xCxHxW.detach().float().cpu()[0]
    C, H, W = x.shape
    outdir_slices = os.path.join(outdir, "per_slice"); os.makedirs(outdir_slices, exist_ok=True)
    for i in range(C):
        base_1xHxW = x[i:i+1].numpy()
        mask_i = _make_brain_mask_from_base(base_1xHxW.squeeze(0))
        _overlay_and_save(
            base_1xHxW, heat_2d,
            os.path.join(outdir_slices, f"{prefix}_slice_{i:02d}.png"),
            alpha=alpha, title=f"{title} | slice {i:02d}",
            brain_mask=mask_i, cmap="jet", vmin=0, vmax=1
        )

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
    print(f"  load [{os.path.basename(path)}] norm={hp['norm']} missing={len(getattr(mu,'missing_keys',[]))} unexpected={len(getattr(mu,'unexpected_keys',[]))}")
    model.eval()
    return model

# -------------------- Localización de la última conv + score --------------------
def _cnn_branch(model, which="global"):
    return model.global_feat if which=="global" else model.local_feat

def _find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

@torch.no_grad()
def _forward_pred(model, img, which="global"):
    out = model(img)
    # tu modelo devuelve un tensor escalar en out[0] (pred global); mantenemos eso
    score = out[0].view([])
    return score

def _score_scalar(model, img, which="global"):
    out = model(img)
    if which == "global":
        score = out[0].view([])
    else:
        # usar contribuciones locales si existen; si no, caer al global
        score = (torch.stack([o.view([]) for o in out[1:]]).mean()
                 if isinstance(out, (list, tuple)) and len(out) > 1 else out[0].view([]))
    return score, out


# -------------------- Métodos xAI --------------------
def gradcampp(model, img_1xCxHxW, which="global"):
    img = img_1xCxHxW.clone().to(DEVICE).requires_grad_(True)
    target_layer = _find_last_conv(_cnn_branch(model, which))
    if target_layer is None:
        return np.zeros(img.shape[-2:], np.float32)

    activations, gradients = [], []

    def fwd_hook(m, i, o): activations.append(o)            # (B,C,h,w)
    def bwd_hook(m, gi, go): gradients.append(go[0])        # (B,C,h,w)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    score, _ = _score_scalar(model, img, which)
    score.backward(retain_graph=True)

    h1.remove(); h2.remove()
    A = activations[-1]  # (1,C,h,w)

    # Fallback si el backward hook no capturó nada
    if len(gradients) == 0:
        g = torch.autograd.grad(score, A, retain_graph=False, allow_unused=True)[0]
        if g is None:
            return np.zeros(img.shape[-2:], np.float32)
        G = g
        # Grad-CAM clásico
        w = G.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * A).sum(dim=1, keepdim=True))
    else:
        G = gradients[-1]
        B,C,h,w = G.shape
        G2, G3 = G*G, G*G*G
        eps = 1e-8
        sumA = (A.view(B,C,-1).sum(-1, keepdim=True)).view(B,C,1,1) + eps
        alpha = G2 / (2*G2 + (A*G3).sum(dim=(2,3), keepdim=True)/sumA + eps)
        w = (alpha * torch.relu(G)).sum(dim=(2,3), keepdim=True)
        cam = torch.relu((w * A).sum(dim=1, keepdim=True))

    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0,0]
    cam = cam - cam.min()
    mx = cam.max()
    if mx > 0: cam = cam / mx
    return cam.detach().cpu().numpy()

def scorecam(model, img_1xCxHxW, which="global", topk=64):
    img = img_1xCxHxW.clone().to(DEVICE)
    target_layer = _find_last_conv(_cnn_branch(model, which))
    if target_layer is None:
        return np.zeros(img.shape[-2:], np.float32)

    feats = []
    h = target_layer.register_forward_hook(lambda m,i,o: feats.append(o))
    with torch.no_grad():
        base_score = _forward_pred(model, img, which).item()
        _ = model(img)
    h.remove()
    A = feats[-1][0]                          # (C,h,w)

    energy = A.view(A.size(0), -1).abs().sum(-1)
    k = min(int(topk), A.size(0))
    idx = torch.topk(energy, k=k).indices
    up = F.interpolate(A[idx].unsqueeze(0), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
    up = up - up.amin(dim=(1,2), keepdim=True)
    den = up.amax(dim=(1,2), keepdim=True); den[den==0] = 1.0
    up = up / den

    weights = []
    with torch.no_grad():
        for c in range(k):
            masked = img * up[c].unsqueeze(0).unsqueeze(0)
            sc = _forward_pred(model, masked, which).item()
            weights.append(abs(sc - base_score))   # <<<<< clave en regresión
    w = torch.tensor(weights, device=up.device).view(k,1,1)

    if torch.all(w == 0):
        # fallback: usa energía de activaciones como pesos
        w = (energy[idx].view(k,1,1)).to(up.device)

    cam = torch.relu((w * up).sum(dim=0))
    cam = cam - cam.min()
    mx = cam.max()
    if mx > 0: cam = cam / mx
    return cam.detach().cpu().numpy()


def integrated_gradients_smooth(model, img_1xCxHxW, which="global", steps=32, samples=8, noise_std=0.1, baseline="zeros"):
    x = img_1xCxHxW.clone().to(DEVICE)
    B,C,H,W = x.shape
    if baseline == "zeros":
        x0 = torch.zeros_like(x)
    elif baseline == "blur":  # baseline simple: promedio espacial por canal
        x0 = x.mean(dim=(2,3), keepdim=True).expand_as(x)
    else:
        x0 = torch.zeros_like(x)

    grads_sum = torch.zeros_like(x)
    diff = (x - x0)

    for s in range(samples):
        noise = torch.randn_like(x) * noise_std if noise_std > 0 else 0.0
        x_noisy = (x + noise).clamp(min=0, max=1)
        path_grads = torch.zeros_like(x)

        for t in range(1, steps+1):
            xt = x0 + (float(t)/steps) * (x_noisy - x0)
            xt.requires_grad_(True)
            score, _ = _score_scalar(model, xt, which)
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)
            g = xt.grad.detach()
            path_grads += g

        avg_grad = path_grads / steps
        grads_sum += avg_grad

    ig = (diff * grads_sum / samples).abs().sum(dim=1)[0]  # (H,W) importancia positiva
    ig = ig - ig.min(); mx = ig.max()
    if mx > 0: ig = ig / mx
    return ig.detach().cpu().numpy()

# Guided Backprop: reemplazar ReLU por GuidedReLU en backward
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return torch.relu(inp)
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        grad_in = grad_out.clone()
        grad_in[inp <= 0] = 0
        grad_in[grad_out <= 0] = 0
        return grad_in

class GuidedReLU_Module(nn.Module):
    def forward(self, x): return GuidedReLU.apply(x)

def _replace_relu_with_guided(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedReLU_Module())
        else:
            _replace_relu_with_guided(child)

def guided_backprop(model, img_1xCxHxW, which="global"):
    # clonar modelo completo para no tocar el original
    model_gb = copy.deepcopy(model)
    _replace_relu_with_guided(_cnn_branch(model_gb, which))
    model_gb.eval()

    x = img_1xCxHxW.clone().to(DEVICE).requires_grad_(True)
    score, _ = _score_scalar(model_gb, x, which)
    model_gb.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    sal = x.grad.detach().abs().sum(dim=1)[0]   # (H,W)
    sal = sal - sal.min()
    mx = sal.max()
    if mx > 0:
        sal = sal / mx
    return sal.cpu().numpy()


def deeplift_rescale(model, img_1xCxHxW, which="global", baseline="zeros"):
    # Aproximación tipo DeepLIFT (Rescale): Gradient × (x - x0)
    x = img_1xCxHxW.clone().to(DEVICE)
    if baseline == "zeros":
        x0 = torch.zeros_like(x)
    elif baseline == "blur":
        x0 = x.mean(dim=(2,3), keepdim=True).expand_as(x)
    else:
        x0 = torch.zeros_like(x)

    x.requires_grad_(True)
    score, _ = _score_scalar(model, x, which)
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)
    g = x.grad.detach()
    contrib = (x - x0) * g
    sal = contrib.abs().sum(dim=1)[0]
    sal = sal - sal.min()
    mx = sal.max()
    if mx > 0:
        sal = sal / mx
    return sal.detach().cpu().numpy()

# -------------------- Carga sujeto / paths / guardado --------------------
def _load_subject_age(base_dir, subject_id, planes=("axial","coronal","sagittal")):
    for pl in planes:
        p = os.path.join(base_dir, f"../../data/processed/{pl}", f"{subject_id}.pt")
        if os.path.exists(p):
            try:
                sample = torch.load(p, map_location="cpu")
                age = sample.get("age", None)
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

def _fmt_plane_title(kind, plane, age_txt, pred_txt, *, for_ensemble=False):
    return (f"{kind} ({plane}) | Pred {pred_txt}" if for_ensemble
            else f"{kind} ({plane}) | Real {age_txt} | Pred {pred_txt}")

# -------------------- Ejecución por plano --------------------
METHOD_FUNCS = {
    "gradcampp": gradcampp,
    "scorecam":  scorecam,
    "ig_smoothgrad": integrated_gradients_smooth,
    "guidedbp": guided_backprop,
    "deeplift": deeplift_rescale,
}

def run_for_plane(plane, args, subject_id):
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

    outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}")
    os.makedirs(outdir, exist_ok=True)

    bases, labels = _pick_bases_from_multichannel(image, mode=args.slice_mode)
    masks = [_make_brain_mask_from_base(b) for b in bases]
    age_txt, pred_txt = f"{age:.1f}", f"{pred:.1f}"

    which_list = []
    if args.which in ("global","both"): which_list.append(("global", "global"))
    if args.which in ("local","both"):  which_list.append(("local",  "local"))

    # Ejecutar cada método pedido
    for which, tag in which_list:
        for mname in args.methods:
            fn = METHOD_FUNCS[mname]
            if mname == "scorecam":
                heat = fn(model, image, which=which, topk=args.scorecam_topk)
            elif mname == "ig_smoothgrad":
                heat = fn(model, image, which=which, steps=args.ig_steps,
                          samples=args.ig_samples, noise_std=args.ig_noise_std,
                          baseline=args.ig_baseline)
            elif mname == "deeplift":
                heat = fn(model, image, which=which, baseline=args.dl_baseline)
            else:
                heat = fn(model, image, which=which)

            prefix = f"{mname}_{tag}"
            np.save(os.path.join(outdir, f"{prefix}.npy"), heat)

            for base_i, mask_i, lab in zip(bases, masks, labels):
                _overlay_and_save(
                    base_i, heat, os.path.join(outdir, f"{prefix}_{lab}.png"),
                    alpha=args.alpha,
                    title=_fmt_plane_title(f"{mname.replace('_',' ').upper()} {which.upper()}",
                                           plane, age_txt, pred_txt, for_ensemble=False),
                    brain_mask=mask_i, show_colorbar=True, cmap=("jet"), vmin=0, vmax=1
                )
                _overlay_and_save(
                    base_i, heat, os.path.join(outdir, f"{prefix}_{lab}_nobar.png"),
                    alpha=args.alpha,
                    title=_fmt_plane_title(f"{mname.replace('_',' ').upper()} {which.upper()}",
                                           plane, age_txt, pred_txt, for_ensemble=True),
                    brain_mask=mask_i, show_colorbar=False, cmap=("jet"), vmin=0, vmax=1
                )

            _save_per_slice_overlays(
                image, heat, outdir, prefix=prefix, alpha=args.alpha,
                title=f"{mname.replace('_',' ').upper()} {which.upper()} ({plane}) | Real {age_txt} | Pred {pred_txt}",
            )

    # resumen json
    summary = {
        "plane": plane, "model_id": model_id, "subject_id": subject_id,
        "edad_real": _py(age), "edad_predicha": _py(pred),
        "model_path": os.path.abspath(model_path),
        "hparams_inferidos": {k: _py(v) for k,v in hp.items() if k != "state_dict"},
        "methods": args.methods, "which": args.which,
        "ig": {"steps": args.ig_steps, "samples": args.ig_samples, "noise_std": args.ig_noise_std, "baseline": args.ig_baseline},
        "scorecam_topk": int(args.scorecam_topk),
        "deeplift_baseline": args.dl_baseline,
    }
    with open(os.path.join(outdir, "explain_config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{plane}] {subject_id} | Real: {age_txt} - Predicha: {pred_txt}")
    print(f"   Guardado en: {outdir}")

# -------------------- Montajes ensemble (idénticos a tus scripts) ----------
TITLE_FONT_SIZE = 18
TITLE_COLOR_RGB = (0, 0, 0)

def _grid_save_shared_colorbar_nobar(images_paths, out_path, title=None,
                                     title_size=TITLE_FONT_SIZE, title_color=TITLE_COLOR_RGB,
                                     cbar_height=0.02):
    paths = [p for p in images_paths if p is not None and os.path.exists(p)]
    if not paths: return
    fig_w = 12; fig_h = 4.2; fig, axs = plt.subplots(1, len(paths), figsize=(fig_w, fig_h))
    if len(paths) == 1: axs = [axs]
    for ax, p in zip(axs, paths):
        img = mpimg.imread(p); ax.imshow(img); ax.axis("off")
    if title: fig.suptitle(title, fontsize=title_size, color=title_color, y=0.98)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.20, wspace=-0.5)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap="jet"); sm.set_array([])
    fig_left, fig_right = 0.015, 0.985
    width = 0.72; left  = fig_left + ((fig_right - fig_left) - width)/2
    cax = fig.add_axes([left, 0.15, width, cbar_height])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0,0.25,0.5,0.75,1]); cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
    cbar.set_label("Nivel de relevancia", fontsize=9)
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
                cands = sorted([f for f in os.listdir(per_slice)
                                if f.startswith(prefix) and f.endswith(".png")])
                if cands:
                    mid = cands[len(cands)//2]; picked = os.path.join(per_slice, mid)
        paths.append(picked)
    return paths

def _save_ensemble_montages_for_subject(subject_id, plane_outdirs, out_root, title_suffix=None, prefix="gradcampp_global"):
    ens_dir = os.path.join(out_root, f"ensemble_{subject_id}"); os.makedirs(ens_dir, exist_ok=True)
    paths = _collect_paths(prefix, plane_outdirs)
    if any(p is not None for p in paths):
        _grid_save_shared_colorbar_nobar(paths, os.path.join(ens_dir, f"ensemble_{prefix}.png"), title=title_suffix)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="xAI CNN multi-plano (Grad-CAM++, Score-CAM, IG-Smooth, GuidedBP, DeepLIFT-Rescale)")
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
    ap.add_argument("--outdir", default="./explanations_xai")
    ap.add_argument("--ensemble-preds-dir", default="./predictions")
    # métodos
    ap.add_argument("--methods", nargs="+",
                    default=["gradcampp","scorecam","ig_smoothgrad","guidedbp","deeplift"],
                    choices=list(METHOD_FUNCS.keys()))
    ap.add_argument("--which", choices=["global","local","both"], default="both")

    # hiperparámetros de métodos
    ap.add_argument("--scorecam-topk", type=int, default=64)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-samples", type=int, default=8)
    ap.add_argument("--ig-noise-std", type=float, default=0.1)
    ap.add_argument("--ig-baseline", choices=["zeros","blur"], default="zeros")
    ap.add_argument("--dl-baseline", choices=["zeros","blur"], default="zeros")

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

    ids_to_use = val_ids[:args.n]
    print("\n🧠 Explicando planos:", ", ".join(args.planes))
    print("📄 Sujetos:", ", ".join(ids_to_use) if ids_to_use else "(ninguno)")
    print("🔍 Métodos:", ", ".join(args.methods))
    print("🧩 Rama:", args.which)

    if not (args.id or args.axial_id or args.coronal_id or args.sagittal_id or
            args.axial_model or args.coronal_model or args.sagittal_model):
        raise SystemExit("Debes especificar al menos --id o --<plane>-id o --<plane>-model.")

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

        # genera un montaje por método 
        tags = ["global","local"] if args.which in ("both","local") else ["global"]
        for m in args.methods:
            for t in tags:
                _save_ensemble_montages_for_subject(
                    subject_id=subject_id,
                    plane_outdirs=plane_outdirs,
                    out_root=args.outdir,
                    title_suffix=title,
                    prefix=f"{m}_{t}"
                )

    
    
    print("\n✅ Listo. Revisá la carpeta de salida:", args.outdir)

if __name__ == "__main__":
    main()
