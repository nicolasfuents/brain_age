#!/usr/bin/env python3
# ---------------------------------------------------------
# explain_ensemble_IG_SHAP.py
# Explica un ENSEMBLE por plano con:
#   - Integrated Gradients (IG)
#   - Atención del transformer (si está disponible) o proxy por parches
#   - PatchSHAP (aprox. KernelSHAP por parches, sin dependencias externas)
# Compatibilidad con tu flujo: autoinfiere hparams/normalización desde cada
# checkpoint, soporta IDs o rutas por plano, slice-mode, filtros por edad,
# y genera montajes por sujeto y por ensemble (como tu script de Grad-CAM).
# ---------------------------------------------------------

import os, sys, re, math, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw, ImageFont

# === Importar modelo desde root del proyecto ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM01 = mcolors.Normalize(vmin=0, vmax=1)

# -------------------- Utilitarios --------------------
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
        t = t - t.min();  tmax = t.max();  t = t / tmax if tmax > 0 else t
        return t
    if mode == "first":
        bases = [norm01(x[0, 0:1]).numpy().squeeze(0)]; labels = ["first"]
    elif mode == "middle":
        m = C // 2; bases = [norm01(x[0, m:m+1]).numpy().squeeze(0)]; labels = ["middle"]
    elif mode == "all":
        bases = [norm01(x[0, i:i+1]).numpy().squeeze(0) for i in range(C)]; labels = [f"{i:02d}" for i in range(C)]
    elif mode == "mean":
        bases = [norm01(x.mean(dim=1, keepdim=True)).numpy().squeeze(0)]; labels = ["mean"]
    elif mode == "max":
        bases = [norm01(x.max(dim=1, keepdim=True).values).numpy().squeeze(0)]; labels = ["max"]
    else:
        raise ValueError(f"slice-mode desconocido: {mode}")
    return bases, labels

def _make_brain_mask_from_base(base_2d, thr=0.05):
    b = base_2d
    mask = (b > thr).astype(np.float32)
    return mask

def _overlay_and_save(img_1xHxW, heat_2d, path_png, alpha=0.45, title=None,
                      brain_mask=None, show_colorbar=True, bottom_pad_with_bar=0.18, bottom_pad_no_bar=0.02,
                      cmap="jet", vmin=0, vmax=1):
    if torch.is_tensor(img_1xHxW):
        base = _to_numpy01(img_1xHxW.squeeze(0))
    else:
        arr = np.asarray(img_1xHxW); base = np.squeeze(arr)
    if base.ndim != 2: raise ValueError(f"Base debe ser 2D, obtuve {base.shape}")

    heat = np.asarray(heat_2d, dtype=np.float32); heat = np.squeeze(heat)
    if heat.ndim != 2: raise ValueError(f"Heat debe ser 2D, obtuve {heat.shape}")

    # Rotaciones coherentes con tu flujo
    base = np.rot90(base, k=1); heat = np.rot90(heat, k=1)

    if brain_mask is not None:
        m = np.asarray(brain_mask); m = np.squeeze(m); m = np.rot90(m, k=1); m = m > 0
        if m.shape != heat.shape:
            H, W = heat.shape; Mh, Mw = m.shape
            mf = np.zeros_like(heat, dtype=bool); h = min(H, Mh); w = min(W, Mw); mf[:h,:w] = m[:h,:w]; m = mf
        if np.any(m):
            # Normalizar dentro del cerebro
            hs = np.where(m, heat, np.nan)
            vmax_in = np.nanmax(np.abs(hs)) if (vmin < 0) else np.nanmax(hs)
            if vmax_in > 0:
                heat = heat / vmax_in
            heat = np.where(m, heat, np.nan)
    else:
        # Normalización global si no hay máscara
        if vmin < 0:
            mx = np.max(np.abs(heat));  heat = heat / mx if mx > 0 else heat
        else:
            heat = heat - np.nanmin(heat);  mx = np.nanmax(heat);  heat = heat / mx if mx > 0 else heat

    fig = plt.figure(figsize=(6, 6)); ax = plt.gca()
    ax.imshow(base, cmap="gray", interpolation="bicubic")
    im2 = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, interpolation="bilinear", alpha=alpha, vmin=vmin, vmax=vmax)
    if title: ax.set_title(title, fontsize=17, pad=4)
    ax.axis("off")

    if show_colorbar:
        cax = inset_axes(ax, width="100%", height="3%", loc="lower center",
                         bbox_to_anchor=(0, -0.10, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=im2.get_cmap()); sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([vmin, (vmin+vmax)/2, vmax] if vmin < 0 else [0, 0.25, 0.5, 0.75, 1])
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
        label = "Atribución (IG/SHAP/Attn)"
        cbar.set_label(label, fontsize=9); cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=8)
        cbar.solids.set_alpha(1); cbar.ax.set_facecolor((0,0,0,0)); cbar.ax.patch.set_alpha(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_with_bar)
    else:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_pad_no_bar)

    ax.margins(x=0, y=0)
    fig.savefig(path_png, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# Guardado por-cada-slice
def _save_per_slice_overlays(image_1xCxHxW, heat_2d, outdir, prefix, alpha, title, signed=False):
    x = image_1xCxHxW.detach().float().cpu()[0]
    C, H, W = x.shape; outdir_slices = os.path.join(outdir, "per_slice"); os.makedirs(outdir_slices, exist_ok=True)
    for i in range(C):
        base_1xHxW = x[i:i+1].numpy(); mask_i = _make_brain_mask_from_base(base_1xHxW.squeeze(0))
        _overlay_and_save(base_1xHxW, heat_2d, os.path.join(outdir_slices, f"{prefix}_slice_{i:02d}.png"),
                          alpha=alpha, title=f"{title} | slice {i:02d}", brain_mask=mask_i,
                          cmap=("seismic" if signed else "jet"), vmin=(-1 if signed else 0), vmax=1)

# -------------------- Carga de checkpoints --------------------
def _open_state_dict(path, maploc):
    ckpt = torch.load(path, map_location=maploc)
    if hasattr(ckpt, "state_dict"): return ckpt.state_dict()
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt

def _infer_backbone_norm_inplace_nblock(path):
    sd = _open_state_dict(path, DEVICE); keys = list(sd.keys())
    is_resnet = any(("global_feat.stem" in k) or ("global_feat.layer1.0" in k) for k in keys)
    is_vgg16  = any(".conv33." in k or ".conv43." in k or ".conv53." in k for k in keys)
    backbone = "resnet18" if is_resnet else ("vgg16" if is_vgg16 else "vgg8")
    has_running = any(k.endswith("running_mean") or k.endswith("running_var") for k in keys)
    norm = "bn" if has_running else "gn"
    conv_in = []
    for k, v in sd.items():
        if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 4: conv_in.append(int(v.shape[1]))
    if not conv_in: raise RuntimeError("No conv2d weights found to infer inplace.")
    inplace = int(min(conv_in))
    idxs = {int(m.group(1)) for k in keys for m in [re.search(r"attnlist\.(\d+)\.", k)] if m}
    nblock = (max(idxs) + 1) if idxs else 0
    patch_size = step = None; ckpt = torch.load(path, map_location="cpu")
    for key in ["hparams", "config", "meta", "args", "hyperparams"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            d = ckpt[key]; patch_size = patch_size or d.get("patch_size"); step = step or d.get("step")
    return {"backbone": backbone, "norm": norm, "inplace": inplace, "nblock": nblock,
            "patch_size": patch_size or 64, "step": step or 32, "state_dict": sd}

def _build_and_load_model(hp, path):
    model = GlobalLocalBrainAge(inplace=hp["inplace"], patch_size=hp["patch_size"], step=hp["step"],
                                nblock=hp["nblock"], backbone=hp["backbone"], backbone_norm=hp["norm"],
                                backbone_pretrained=False, backbone_freeze_bn=False).to(DEVICE)
    missing_unexpected = model.load_state_dict(hp["state_dict"], strict=False)
    mk = getattr(missing_unexpected, "missing_keys", []); uk = getattr(missing_unexpected, "unexpected_keys", [])
    print(f"  load [{os.path.basename(path)}] norm={hp['norm']} missing={len(mk)} unexpected={len(uk)}")
    model.eval(); return model

# -------------------- Integrated Gradients --------------------
@torch.no_grad()
def _forward_scalar(model, img, which="global"):
    # which: 'global' -> outs[0]; 'local' -> mean(outs[1:])
    outs = model(img)
    if which == "global" or len(outs) <= 1:
        return outs[0].view([])
    else:
        return torch.stack([o.view([]) for o in outs[1:]]).mean()

def integrated_gradients(model, image, steps=50, which="global", baseline="zero"):
    model.eval()
    img = image.detach()
    if baseline == "zero":
        base = torch.zeros_like(img)
    elif baseline == "mean":
        # baseline plana con el promedio por canal de la propia imagen
        base = img.mean(dim=(-2, -1), keepdim=True).expand_as(img)
    else:
        base = torch.zeros_like(img)

    # No-grad para línea de puntos, pero necesitamos grad dentro del loop
    alphas = torch.linspace(0, 1, steps, device=img.device)
    total = torch.zeros_like(img)
    for a in alphas:
        x = (base + a * (img - base)).clone().detach().requires_grad_(True)
        outs = model(x)
        score = outs[0].view([]) if which == "global" or len(outs) <= 1 else torch.stack([o.view([]) for o in outs[1:]]).mean()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        if x.grad is None:
            continue
        total += x.grad
    avg_grad = total / max(1, len(alphas))
    ig = (img - base) * avg_grad
    # Agregar sobre canales para heatmap 2D
    ig_map = ig.abs().sum(dim=1)[0]  # (H,W)
    # Normalizar a [-1,1] si querés signo: aquí usamos magnitud (0..1)
    ig_map = ig_map - ig_map.min();  mx = ig_map.max();  ig_map = ig_map / mx if mx > 0 else ig_map
    return ig_map.detach().cpu().numpy()

# -------------------- Atención del transformer --------------------
class _AttnCatcher:
    def __init__(self):
        self.weights = []
    def hook(self, module, inp, out):
        # Intentar capturar pesos si vienen como (out, attn) o atributo
        if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
            self.weights.append(out[1].detach().cpu())
        elif hasattr(module, 'last_attn') and isinstance(module.last_attn, torch.Tensor):
            self.weights.append(module.last_attn.detach().cpu())

@torch.no_grad()
def transformer_attention_map(model, image, patch_size=64, step=32, layer_index=-1):
    """
    Intenta capturar el mapa de atención de la última capa de atención.
    Si no se logra, cae en un proxy basado en los scores locales (outs[1:])
    para construir un heatmap de parches.
    """
    model.eval()
    catcher = _AttnCatcher()

    # heurística: enganchar módulos que contengan 'attn' en su nombre
    handles = []
    for name, m in model.named_modules():
        if 'attn' in name.lower() or 'attention' in name.lower():
            try:
                h = m.register_forward_hook(catcher.hook); handles.append(h)
            except Exception:
                pass

    outs = model(image)
    for h in handles: h.remove()

    H, W = image.shape[-2:]
    attn_heat = None
    if catcher.weights:
        A = catcher.weights[layer_index]
        # A: (n_heads, N, N) o (B,n_heads,N,N) -> tomamos media por heads y sobre tokens destino
        A = A.mean(dim=0) if A.dim() == 3 else A.mean(dim=(0,1))  # (N,N)
        # Suponemos que los primeros Np tokens son parches; proyectamos importancia por token
        token_importance = A.mean(dim=0)  # (N,)
        # Reconstruir grid de parches por desplazamiento (aprox a partir de step/patch)
        gy = (H - patch_size) // step; gx = (W - patch_size) // step
        grid = torch.zeros((H, W))
        idx = 0
        for y in range(0, H - patch_size, step):
            for x in range(0, W - patch_size, step):
                if idx >= len(token_importance):
                    continue
                val = float(token_importance[idx].item())
                grid[y:y+patch_size, x:x+patch_size] += val
                idx += 1
        g = grid - grid.min();  mx = grid.max();  grid = g / mx if mx > 0 else g
        attn_heat = grid.numpy()
    else:
        # Proxy: usar los outs locales como importancia de parche
        scores = outs[1:]  # lista de tensores scalars
        grid = torch.zeros((H, W), device=image.device)
        idx = 0
        for y in range(0, H - patch_size, step):
            for x in range(0, W - patch_size, step):
                if idx >= len(scores):
                    continue
                val = float(scores[idx])
                grid[y:y+patch_size, x:x+patch_size] += val
                idx += 1
        grid = grid - grid.min(); mx = grid.max(); grid = grid / mx if mx > 0 else grid
        attn_heat = grid.detach().cpu().numpy()

    return attn_heat

# -------------------- PatchSHAP (aprox. sin SHAP lib) --------------------
@torch.no_grad()
def patch_shap(model, image, patch_size=64, step=32, samples=256, baseline="zero"):
    """
    Aproximación tipo KernelSHAP sobre parches: genera máscaras binarias sobre la grilla
    de parches, evalúa el modelo con cada máscara aplicada (baseline en celdas apagadas)
    y resuelve una regresión ridge para estimar valores de Shapley por parche.
    Devuelve un heatmap 2D en escala 0..1.
    """
    model.eval()
    _, _, H, W = image.shape
    gy = (H - patch_size) // step
    gx = (W - patch_size) // step
    n_patches = max(1, gy * gx)

    if baseline == "zero":
        base_val = 0.0
    else:  # mean por canal
        base_val = float(image.mean().item())

    # Construir matriz de diseño X (muestras x n_patches) y vector y (pred)
    X = np.zeros((samples, n_patches), dtype=np.float32)
    y = np.zeros((samples,), dtype=np.float32)

    for s in range(samples):
        mask = (np.random.rand(n_patches) < 0.5).astype(np.float32)
        X[s] = mask
        img2 = image.clone()
        # aplicar baseline en parches apagados
        idx = 0
        for yy in range(0, H - patch_size, step):
            for xx in range(0, W - patch_size, step):
                if mask[idx] < 0.5:
                    img2[..., yy:yy+patch_size, xx:xx+patch_size] = base_val
                idx += 1
        y[s] = _forward_scalar(model, img2, which="global").item()

    # Ridge pequeña para estabilidad
    lam = 1e-3
    XtX = X.T @ X + lam * np.eye(n_patches, dtype=np.float32)
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)  # (n_patches,)

    # Proyectar a mapa 2D
    heat = torch.zeros((H, W), device=image.device)
    idx = 0
    for yy in range(0, H - patch_size, step):
        for xx in range(0, W - patch_size, step):
            val = float(w[idx])
            heat[yy:yy+patch_size, xx:xx+patch_size] += val
            idx += 1
    # Normalizar a 0..1 con absoluto (importancia)
    heat = heat.abs(); heat = heat - heat.min(); mx = heat.max(); heat = heat / mx if mx > 0 else heat
    return heat.detach().cpu().numpy()

# -------------------- Resolución de paths y data --------------------
def _load_subject_age(base_dir, subject_id, planes=("axial","coronal","sagittal")):
    for pl in planes:
        p = os.path.join(base_dir, f"../../data/processed/{pl}", f"{subject_id}.pt")
        if os.path.exists(p):
            try:
                sample = torch.load(p, map_location="cpu"); age = sample.get("age", None)
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
    if explicit: return explicit, f"{plane}_custompath"
    pid = getattr(args, f"{plane}_id", None) or args.id
    if pid is None: raise ValueError(f"Debes especificar --{plane}-model o --{plane}-id o --id")
    return os.path.join(base_dir, f"../models/model_{plane}_{pid}.pt"), str(pid)

# -------------------- Ejecución por plano --------------------
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

    outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}"); os.makedirs(outdir, exist_ok=True)
    bases, labels = _pick_bases_from_multichannel(image, mode=args.slice_mode)
    masks = [_make_brain_mask_from_base(b) for b in bases]
    age_txt, pred_txt = f"{age:.1f}", f"{pred:.1f}"

    # === Integrated Gradients ===
    ig_map = integrated_gradients(model, image, steps=args.ig_steps, which=("local" if args.ig_local else "global"), baseline=args.ig_baseline)
    np.save(os.path.join(outdir, "integrated_gradients.npy"), ig_map)
    for base_i, mask_i, lab in zip(bases, masks, labels):
        _overlay_and_save(base_i, ig_map, os.path.join(outdir, f"ig_{lab}.png"), alpha=args.alpha,
                          title=f"Integrated Gradients ({plane}) | Real {age_txt} | Pred {pred_txt}",
                          brain_mask=mask_i, show_colorbar=True, cmap=("jet" if not args.ig_signed else "seismic"), vmin=(0 if not args.ig_signed else -1), vmax=1)
        _overlay_and_save(base_i, ig_map, os.path.join(outdir, f"ig_{lab}_nobar.png"), alpha=args.alpha,
                          title=f"IG ({plane}) | Pred {pred_txt}", brain_mask=mask_i, show_colorbar=False,
                          cmap=("jet" if not args.ig_signed else "seismic"), vmin=(0 if not args.ig_signed else -1), vmax=1)
    _save_per_slice_overlays(image, ig_map, outdir, prefix="ig", alpha=args.alpha,
                             title=f"Integrated Gradients ({plane}) | Real {age_txt} | Pred {pred_txt}", signed=args.ig_signed)

    # === Atención del transformer (o proxy) ===
    attn_map = transformer_attention_map(model, image, patch_size=hp["patch_size"], step=hp["step"], layer_index=args.attn_layer)
    np.save(os.path.join(outdir, "attention.npy"), attn_map)
    for base_i, mask_i, lab in zip(bases, masks, labels):
        _overlay_and_save(base_i, attn_map, os.path.join(outdir, f"attn_{lab}.png"), alpha=args.alpha,
                          title=f"Atención/Proxy ({plane}) | Real {age_txt} | Pred {pred_txt}", brain_mask=mask_i, show_colorbar=True)
        _overlay_and_save(base_i, attn_map, os.path.join(outdir, f"attn_{lab}_nobar.png"), alpha=args.alpha,
                          title=f"Atención ({plane}) | Pred {pred_txt}", brain_mask=mask_i, show_colorbar=False)
    _save_per_slice_overlays(image, attn_map, outdir, prefix="attn", alpha=args.alpha,
                             title=f"Atención/Proxy ({plane}) | Real {age_txt} | Pred {pred_txt}")

    # === PatchSHAP ===
    shap_map = patch_shap(model, image, patch_size=hp["patch_size"], step=hp["step"], samples=args.shap_samples, baseline=args.shap_baseline)
    np.save(os.path.join(outdir, "patchshap.npy"), shap_map)
    for base_i, mask_i, lab in zip(bases, masks, labels):
        _overlay_and_save(base_i, shap_map, os.path.join(outdir, f"shap_{lab}.png"), alpha=args.alpha,
                          title=f"PatchSHAP ({plane}) | Real {age_txt} | Pred {pred_txt}", brain_mask=mask_i, show_colorbar=True)
        _overlay_and_save(base_i, shap_map, os.path.join(outdir, f"shap_{lab}_nobar.png"), alpha=args.alpha,
                          title=f"PatchSHAP ({plane}) | Pred {pred_txt}", brain_mask=mask_i, show_colorbar=False)
    _save_per_slice_overlays(image, shap_map, outdir, prefix="shap", alpha=args.alpha,
                             title=f"PatchSHAP ({plane}) | Real {age_txt} | Pred {pred_txt}")

    # Resumen JSON
    summary = {
        "plane": plane, "model_id": model_id, "subject_id": subject_id,
        "edad_real": _py(age), "edad_predicha": _py(pred),
        "model_path": os.path.abspath(model_path),
        "hparams_inferidos": {k: _py(v) for k, v in hp.items() if k != "state_dict"},
        "ig_steps": int(args.ig_steps), "ig_baseline": args.ig_baseline, "ig_local": bool(args.ig_local),
        "attn_layer": int(args.attn_layer), "shap_samples": int(args.shap_samples), "shap_baseline": args.shap_baseline
    }
    with open(os.path.join(outdir, "explain_config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{plane}] {subject_id} | Real: {age_txt} - Predicha: {pred_txt}")
    print(f"   Guardado en: {outdir}")

# -------------------- Montajes Ensemble (comparten colorbar) --------------
TITLE_FONT_SIZE = 18
TITLE_COLOR_RGB = (0, 0, 0)

def _grid_save_shared_colorbar_nobar(images_paths, out_path, title=None,
                                     title_size=TITLE_FONT_SIZE, title_color=TITLE_COLOR_RGB,
                                     wspace=0.03, cbar_height=0.02, cbar_pad=0.06,
                                     signed=False):
    paths = [p for p in images_paths if p is not None and os.path.exists(p)]
    if not paths: return
    fig_w = 12; fig_h = 4.2; fig, axs = plt.subplots(1, len(paths), figsize=(fig_w, fig_h))
    if len(paths) == 1: axs = [axs]
    for ax, p in zip(axs, paths):
        img = mpimg.imread(p); ax.imshow(img); ax.axis("off")
    if title: fig.suptitle(title, fontsize=title_size, color=title_color, y=0.98)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.20, wspace=-0.5)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=(-1 if signed else 0), vmax=1), cmap=("seismic" if signed else "jet"))
    sm.set_array([])
    fig_left, fig_right = 0.015, 0.985; avail = fig_right - fig_left; width = 0.72; left  = fig_left + (avail - width)/2
    cax = fig.add_axes([left, 0.15, width, cbar_height])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [-1, 0, 1] if signed else [0, 0.25, 0.5, 0.75, 1]
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))
    cbar.set_label("Nivel de relevancia", fontsize=9)
    cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=8); cbar.solids.set_alpha(1)
    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _collect_paths(prefix, plane_outdirs):
    paths = []
    for pdir in plane_outdirs:
        picked = None
        for name in (f"{prefix}_middle_nobar.png", f"{prefix}_mean_nobar.png", f"{prefix}_first_nobar.png", f"{prefix}_max_nobar.png"):
            cand = os.path.join(pdir, name)
            if os.path.exists(cand): picked = cand; break
        if picked is None:
            for name in (f"{prefix}_middle.png", f"{prefix}_mean.png", f"{prefix}_first.png", f"{prefix}_max.png"):
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


def _save_ensemble_montages_for_subject(subject_id, plane_outdirs, out_root, title_suffix=None, signed_ig=False):
    ens_dir = os.path.join(out_root, f"ensemble_{subject_id}"); os.makedirs(ens_dir, exist_ok=True)
    for kind, signed in [("ig", signed_ig), ("attn", False), ("shap", False)]:
        paths = _collect_paths(kind, plane_outdirs)
        _grid_save_shared_colorbar_nobar(paths, os.path.join(ens_dir, f"ensemble_{kind}.png"), title=title_suffix, signed=signed)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Explicabilidad (IG, Atención, PatchSHAP) multi-plano con autoconfiguración")
    # IDs genéricos o por plano
    ap.add_argument("--id", help="ID común para los tres planos (opcional)")
    ap.add_argument("--axial-id"); ap.add_argument("--coronal-id"); ap.add_argument("--sagittal-id")
    # Rutas directas
    ap.add_argument("--axial-model"); ap.add_argument("--coronal-model"); ap.add_argument("--sagittal-model")

    ap.add_argument("--planes", nargs="+", default=["axial","coronal","sagittal"]) 
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--age-min", type=float, default=None)
    ap.add_argument("--age-max", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--slice-mode", choices=["first","middle","all","mean","max"], default="middle")
    ap.add_argument("--outdir", default="./explanations_ig_shap")

    # Ensemble titles (opcionales)
    ap.add_argument("--ensemble-preds-dir", default="./predictions", help="Carpeta con val_true_ensemble.txt y val_pred_ensemble.txt")

    # IG params
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--ig-baseline", choices=["zero","mean"], default="zero")
    ap.add_argument("--ig-local", action="store_true", help="IG sobre score local (media de outs[1:])")
    ap.add_argument("--ig-signed", action="store_true", help="Mapa IG con colormap divergente (-1..1)")

    # Atención
    ap.add_argument("--attn-layer", type=int, default=-1, help="Índice de capa de atención a usar si hay múltiples capturas")

    # SHAP
    ap.add_argument("--shap-samples", type=int, default=256)
    ap.add_argument("--shap-baseline", choices=["zero","mean"], default="zero")

    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    val_file = os.path.join(base_dir, "../IDs/val_ids.txt")
    with open(val_file, "r") as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]
    with open(val_file, "r") as f:
        all_val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # Cargar ensemble .txt para títulos
    ens_true_arr = ens_pred_arr = None
    try:
        ens_true_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_true_ensemble.txt"))
        ens_pred_arr = np.loadtxt(os.path.join(args.ensemble_preds_dir, "val_pred_ensemble.txt"))
        if np.isscalar(ens_true_arr): ens_true_arr = np.array([float(ens_true_arr)])
        if np.isscalar(ens_pred_arr): ens_pred_arr = np.array([float(ens_pred_arr)])
    except Exception as e:
        print(f"[WARN] No pude cargar ensemble .txt desde {args.ensemble_preds_dir}: {e}")

    # Filtro por edad
    if args.age_min is not None or args.age_max is not None:
        filtered = []
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
    print("⚙️ IG: steps=", args.ig_steps, "baseline=", args.ig_baseline, "local=", args.ig_local)
    print("⚙️ Atención: layer=", args.attn_layer)
    print("⚙️ PatchSHAP: samples=", args.shap_samples, "baseline=", args.shap_baseline)

    if not (args.id or args.axial_id or args.coronal_id or args.sagittal_id or args.axial_model or args.coronal_model or args.sagittal_model):
        raise SystemExit("Debes especificar al menos --id o --<plane>-id o --<plane>-model.")

    for subject_id in ids_to_use:
        plane_outdirs = []
        for plane in args.planes:
            run_for_plane(plane, args, subject_id)
            model_path, model_id = _resolve_model_path(plane, args)
            outdir = os.path.join(args.outdir, f"{plane}_{model_id}_{subject_id}"); plane_outdirs.append(outdir)

        # Título con pred ensemble y edad real (desde .txt)
        title = None
        if ens_true_arr is not None and ens_pred_arr is not None:
            try:
                idx_all = all_val_ids.index(subject_id)
                if 0 <= idx_all < len(ens_true_arr) and 0 <= idx_all < len(ens_pred_arr):
                    ens_true = float(ens_true_arr[idx_all]); ens_pred = float(ens_pred_arr[idx_all])
                    title = f"Ensemble Pred: {ens_pred:.1f} | Real {ens_true:.1f}"
            except ValueError:
                pass

        _save_ensemble_montages_for_subject(subject_id=subject_id, plane_outdirs=plane_outdirs,
                                             out_root=args.outdir, title_suffix=title, signed_ig=args.ig_signed)

    print("\n✅ Listo. Revisá la carpeta de salida:", args.outdir)

if __name__ == "__main__":
    main()
