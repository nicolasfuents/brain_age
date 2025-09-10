# evaluate_hybrid_ensemble.py

# Mejor ensemble:
# python evaluate_hybrid_ensemble.py --axial-model ../models/model_axial_20250822-1437.pt --coronal-model ../models/model_coronal_20250814-0853.pt --sagittal-model ../models/model_sagittal_20250822-1437.pt --patch-size 64 --step 32 --save-summary ../eval/val_summary_winner.txt --save-preds-csv ../eval/val_preds_winner.csv --global-only --ridge-stack --ridge-alpha 1.0 --ridge-use-calibrated

import os
import sys
import argparse
from datetime import datetime
import re
import numpy as np
import torch

# Importa el modelo desde el root del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GlobalLocalTransformer import GlobalLocalBrainAge  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Utilitarios
# --------------------------
def load_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_pred - y_true)))

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean((y_pred - y_true) ** 2))

def fit_affine(y_true, y_pred):
    """ Ajusta y' = a*y + b por mínimos cuadrados. """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    X = np.vstack([y_pred, np.ones_like(y_pred)]).T
    a, b = np.linalg.lstsq(X, y_true, rcond=None)[0]
    return float(a), float(b)

def apply_affine(y, a, b):
    y = np.asarray(y, dtype=np.float32)
    return a * y + b

# ---------- Autodetección desde checkpoint ----------
def _open_state_dict(path, maploc):
    ckpt = torch.load(path, map_location=maploc)
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt

def infer_backbone_inplace_nblock_from_ckpt(state_dict):
    """
    - BACKBONE:
        * ResNet*: si aparecen 'global_feat.layer1.0' o 'global_feat.stem'
        * VGG16: si aparecen conv33 / conv43 / conv53
        * VGG8  : fallback
    - INPLACE: menor in_channels de los pesos conv2d
    - NBLOCK : 1 + max índice en 'attnlist.{i}.'
    """
    keys = list(state_dict.keys())
    is_resnet = any(("global_feat.stem" in k) or ("global_feat.layer1.0" in k) for k in keys)
    is_vgg16 = any(".conv33." in k or ".conv43." in k or ".conv53." in k for k in keys)

    if is_resnet:
        backbone = "resnet18"
    elif is_vgg16:
        backbone = "vgg16"
    else:
        backbone = "vgg8"

    conv_in_ch = []
    for k, v in state_dict.items():
        if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 4:
            conv_in_ch.append(int(v.shape[1]))
    if not conv_in_ch:
        raise RuntimeError("No se encontraron pesos conv2d para inferir INPLACE.")
    inplace = int(min(conv_in_ch))

    idxs = {int(m.group(1)) for k in keys for m in [re.search(r"^attnlist\.(\d+)\.", k)] if m}
    nblock = (max(idxs) + 1) if idxs else 0

    return backbone, inplace, nblock

def infer_norm_from_state_dict(state_dict):
    """ BN si hay running_mean/var; si no, asumimos GN. """
    keys = state_dict.keys()
    has_running = any(k.endswith("running_mean") or k.endswith("running_var") for k in keys)
    return "bn" if has_running else "gn"

def build_model(backbone, inplace, patch_size, step, nblock,
                backbone_norm="bn", backbone_pretrained=False, backbone_freeze_bn=False):
    model = GlobalLocalBrainAge(inplace=inplace, patch_size=patch_size, step=step,
                                nblock=nblock, backbone=backbone,
                                backbone_norm=backbone_norm,
                                backbone_pretrained=backbone_pretrained,
                                backbone_freeze_bn=backbone_freeze_bn).to(DEVICE)
    model.eval()
    return model

def forward_to_scalar(out, global_only=False):
    """ Convierte salida a escalar (soporta listas [glo, loc1, ...]). """
    if isinstance(out, (list, tuple)):
        if global_only:
            out = out[0]
        else:
            out = torch.cat(out, dim=1).mean(dim=1, keepdim=True)
    return float(out.squeeze().detach().cpu().item())

def predict_plane_auto(plane, ids, model_path, data_root, default_ps, default_step, global_only=False):
    """
    - Autodetecta BACKBONE/INPLACE/NBLOCK y NORMA (BN/GN)
    - Carga con strict=False (tolerante a BN/GN/EMA)
    - Recalibra BN con unos forwards (si aplica)
    - Predice y devuelve métricas + params
    """
    data_dir = os.path.join(data_root, plane)

    state = _open_state_dict(model_path, DEVICE)
    backbone, inplace, nblock = infer_backbone_inplace_nblock_from_ckpt(state)
    backbone_norm = infer_norm_from_state_dict(state)
    patch_size = default_ps
    step = default_step

    model = build_model(backbone, inplace, patch_size, step, nblock,
                        backbone_norm=backbone_norm,
                        backbone_pretrained=False,
                        backbone_freeze_bn=False)

    _missing_unexpected = model.load_state_dict(state, strict=False)
    mk = getattr(_missing_unexpected, "missing_keys", [])
    uk = getattr(_missing_unexpected, "unexpected_keys", [])
    print(f"[{plane}] norm={backbone_norm} missing={len(mk)} unexpected={len(uk)}")
    if mk:
        print(f"[{plane}] sample missing:", mk[:10])

    # Recalibración liviana de BN (si existiera)
    model.train()
    with torch.no_grad():
        count = 0
        for _id in ids:
            pth = os.path.join(data_dir, f"{_id}.pt")
            if not os.path.exists(pth):
                continue
            x = torch.load(pth, map_location=DEVICE)["image"].unsqueeze(0).to(DEVICE)
            _ = model(x)
            count += 1
            if count >= 32:
                break
    model.eval()

    ages, preds = [], []
    with torch.no_grad():
        for _id in ids:
            pth = os.path.join(data_dir, f"{_id}.pt")
            sample = torch.load(pth, map_location=DEVICE)
            image = sample["image"].unsqueeze(0).to(DEVICE)
            age = float(sample["age"])
            out = model(image)
            pred = forward_to_scalar(out, global_only=global_only)
            ages.append(age)
            preds.append(pred)

    return (ages, preds, mae(ages, preds), mse(ages, preds),
            {"backbone": backbone, "inplace": inplace, "nblock": nblock,
             "patch_size": patch_size, "step": step, "norm": backbone_norm})

# ---------- Ridge stacking (sin sklearn) ----------
def ridge_fit_predict(y_true, feats, alpha=1.0):
    """
    y_true: (N,)
    feats: dict con claves 'axial','coronal','sagittal' -> arrays (N,)
    alpha: L2 para coeficientes (NO regulariza el sesgo)
    Devuelve: y_pred, (intercept, w_ax, w_cor, w_sag)
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    xa = np.asarray(feats["axial"], dtype=np.float64).reshape(-1, 1)
    xc = np.asarray(feats["coronal"], dtype=np.float64).reshape(-1, 1)
    xs = np.asarray(feats["sagittal"], dtype=np.float64).reshape(-1, 1)

    # Diseño: [1, xa, xc, xs]
    X = np.concatenate([np.ones_like(xa), xa, xc, xs], axis=1)  # (N,4)

    # Matriz de regularización: no regularizar bias
    I = np.eye(4, dtype=np.float64)
    I[0, 0] = 0.0

    XtX = X.T @ X
    XtX_reg = XtX + alpha * I
    Xty = X.T @ y
    theta = np.linalg.solve(XtX_reg, Xty)  # (4,1)

    y_pred = (X @ theta).reshape(-1)
    intercept = float(theta[0, 0])
    w_ax = float(theta[1, 0])
    w_cor = float(theta[2, 0])
    w_sag = float(theta[3, 0])
    return y_pred.astype(np.float32), (intercept, w_ax, w_cor, w_sag)

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid per-plane ensemble (mix backbones) con autodetección por checkpoint")

    # Archivos de IDs y datos
    parser.add_argument("--ids-file", default=os.path.join(os.path.dirname(__file__), "../IDs/val_ids.txt"))
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "../../data/processed"))

    # Hiperparámetros del framework (PS/STEP)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--step", type=int, default=32)

    # Modelos por plano
    parser.add_argument("--axial-model", required=True)
    parser.add_argument("--coronal-model", required=True)
    parser.add_argument("--sagittal-model", required=True)

    # Salidas
    parser.add_argument("--save-summary", default=None)
    parser.add_argument("--save-preds-csv", default=None)

    # Opciones de predicción
    parser.add_argument("--global-only", action="store_true",
                        help="Usar solo la salida global (out[0]) si el modelo devuelve lista")

    # Blend manual y búsqueda automática
    parser.add_argument("--manual-weights", nargs=3, type=float, metavar=("WA", "WC", "WS"),
                        help="Pesos manuales para (axial, coronal, sagittal). Suman ~1.")
    parser.add_argument("--use-calibrated", action="store_true",
                        help="El blend manual usa predicciones calibradas (axial_cal,...).")
    parser.add_argument("--auto-search", action="store_true",
                        help="Barrido automático de pesos favoreciendo coronal: wa=ws=(1-wc)/2, wc∈[wc-min,wc-max].")
    parser.add_argument("--wc-min", type=float, default=0.40)
    parser.add_argument("--wc-max", type=float, default=0.80)
    parser.add_argument("--wc-steps", type=int, default=21, help="Número de puntos en el grid de wc.")

    # Ridge stacking
    parser.add_argument("--ridge-stack", action="store_true",
                        help="Activa stacking lineal con Ridge sobre (axial, coronal, sagittal).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Coeficiente L2 para Ridge (no regulariza el bias).")
    parser.add_argument("--ridge-use-calibrated", action="store_true",
                        help="Si está activo, el stacking usa predicciones calibradas; si no, crudas.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_summary = os.path.join(base_dir, f"../models/val_summary_hybrid_{timestamp}.txt")
    summary_path = args.save_summary or default_summary

    ids = load_ids(args.ids_file)
    planes = ["axial", "coronal", "sagittal"]
    model_paths = {"axial": args.axial_model, "coronal": args.coronal_model, "sagittal": args.sagittal_model}

    # --- Predicciones por plano ---
    all_age = None
    preds_raw, plane_mae, plane_mse, plane_params = {}, {}, {}, {}

    for p in planes:
        ages, preds, p_mae, p_mse, params = predict_plane_auto(
            plane=p,
            ids=ids,
            model_path=model_paths[p],
            data_root=args.data_root,
            default_ps=args.patch_size,
            default_step=args.step,
            global_only=args.global_only
        )
        if all_age is None:
            all_age = ages
        preds_raw[p] = np.array(preds, dtype=np.float32)
        plane_mae[p] = p_mae
        plane_mse[p] = p_mse
        plane_params[p] = params

    ages_np = np.array(all_age, dtype=np.float32)

    # --- Calibración por plano: y' = a*y + b ---
    calib_params, preds_cal = {}, {}
    for p in planes:
        a, b = fit_affine(ages_np, preds_raw[p])
        calib_params[p] = (a, b)
        preds_cal[p] = apply_affine(preds_raw[p], a, b)

    # --- Ensembles base ---
    ens_mean = np.mean([preds_raw[p] for p in planes], axis=0)
    mae_mean = mae(ages_np, ens_mean)
    mse_mean = mse(ages_np, ens_mean)

    eps = 1e-12
    w = np.array([1.0 / (plane_mse[p] + eps) for p in planes], dtype=np.float64)
    w = w / w.sum()
    ens_weighted = (w[0] * preds_raw["axial"] + w[1] * preds_raw["coronal"] + w[2] * preds_raw["sagittal"])
    mae_weighted = mae(ages_np, ens_weighted)
    mse_weighted = mse(ages_np, ens_weighted)

    ens_cal_mean = np.mean([preds_cal[p] for p in planes], axis=0)
    mae_cal_mean = mae(ages_np, ens_cal_mean)
    mse_cal_mean = mse(ages_np, ens_cal_mean)

    plane_mse_cal = {p: mse(ages_np, preds_cal[p]) for p in planes}
    w_cal = np.array([1.0 / (plane_mse_cal[p] + eps) for p in planes], dtype=np.float64)
    w_cal = w_cal / w_cal.sum()
    ens_cal_weighted = (w_cal[0] * preds_cal["axial"] + w_cal[1] * preds_cal["coronal"] + w_cal[2] * preds_cal["sagittal"])
    mae_cal_weighted = mae(ages_np, ens_cal_weighted)
    mse_cal_weighted = mse(ages_np, ens_cal_weighted)

    # --- Blend manual (opcional) ---
    ens_manual = None
    mae_manual = mse_manual = None
    wa = wc = ws = None
    if args.manual_weights is not None:
        wa, wc, ws = args.manual_weights
        s = wa + wc + ws
        if s <= 0:
            raise ValueError("Los pesos manuales deben ser > 0.")
        wa, wc, ws = wa/s, wc/s, ws/s  # normalizamos
        src = preds_cal if args.use_calibrated else preds_raw
        ens_manual = (wa * src["axial"] + wc * src["coronal"] + ws * src["sagittal"])
        mae_manual = mae(ages_np, ens_manual)
        mse_manual = mse(ages_np, ens_manual)

    # --- Auto-search a favor de coronal (opcional) ---
    ens_auto = None
    mae_auto = mse_auto = None
    wa_auto = wc_auto = ws_auto = None
    if args.auto_search:
        candidates = [("cal", preds_cal), ("raw", preds_raw)]
        best = None
        grid = np.linspace(args.wc_min, args.wc_max, max(2, args.wc_steps))
        for label, src in candidates:
            for wc_try in grid:
                wa_try = ws_try = (1.0 - wc_try) / 2.0
                ens_try = wa_try * src["axial"] + wc_try * src["coronal"] + ws_try * src["sagittal"]
                mae_try = mae(ages_np, ens_try)
                mse_try = mse(ages_np, ens_try)
                key = (-mae_try, label, wa_try, wc_try, ws_try, mse_try, ens_try)  # max por -MAE
                if (best is None) or (key > best):
                    best = key
        _, label_best, wa_auto, wc_auto, ws_auto, mse_auto, ens_auto = best
        mae_auto = mae(ages_np, ens_auto)
        auto_uses_cal = (label_best == "cal")

    # --- Ridge stacking (opcional) ---
    ens_ridge = None
    mae_ridge = mse_ridge = None
    ridge_params = None  # (bias, w_ax, w_cor, w_sag)
    if args.ridge_stack:
        src = preds_cal if args.ridge_use_calibrated else preds_raw
        ens_ridge, ridge_params = ridge_fit_predict(ages_np, src, alpha=args.ridge_alpha)
        mae_ridge = mae(ages_np, ens_ridge)
        mse_ridge = mse(ages_np, ens_ridge)

    # --- Resumen ---
    lines = []
    lines.append("Parámetros inferidos por plano:")
    for p in planes:
        pr = plane_params[p]
        lines.append(f"  - {p:8s} | backbone={pr['backbone']:<8s}  norm={pr['norm']:<2s}  "
                     f"in={pr['inplace']}  nblock={pr['nblock']}  patch={pr['patch_size']}  step={pr['step']}")

    lines.append("\nMAE/MSE por plano (sin calibrar):")
    for p in planes:
        lines.append(f"  - {p:8s} | MAE: {plane_mae[p]:.3f} | MSE: {plane_mse[p]:.3f}")

    lines.append("\nCalibración por plano (y' = a·y + b):")
    for p in planes:
        a, b = calib_params[p]
        lines.append(f"  - {p:8s} | a={a:.4f}  b={b:.4f}  | MSE_cal={mse(ages_np, preds_cal[p]):.3f}")

    lines.append("\nEnsembles:")
    lines.append(f"  1) mean                      | MAE: {mae_mean:.3f} | MSE: {mse_mean:.3f}")
    lines.append(f"  2) weighted(1/MSE)           | MAE: {mae_weighted:.3f} | MSE: {mse_weighted:.3f}")
    lines.append(f"  3) cal + mean                | MAE: {mae_cal_mean:.3f} | MSE: {mse_cal_mean:.3f}")
    lines.append(f"  4) cal + weighted(1/MSE_cal) | MAE: {mae_cal_weighted:.3f} | MSE: {mse_cal_weighted:.3f}")

    if ens_manual is not None:
        tag = "[cal]" if args.use_calibrated else "[raw]"
        lines.append(f"  5) manual weights ({wa:.3f},{wc:.3f},{ws:.3f}) {tag} | MAE: {mae_manual:.3f} | MSE: {mse_manual:.3f}")

    if ens_auto is not None:
        tag = "[auto cal]" if auto_uses_cal else "[auto raw]"
        lines.append(f"  6) auto-search ({wa_auto:.3f},{wc_auto:.3f},{ws_auto:.3f}) {tag} | MAE: {mae_auto:.3f} | MSE: {mse_auto:.3f}")

    if ens_ridge is not None and ridge_params is not None:
        b0, w_ax, w_cor, w_sag = ridge_params
        tag = "[cal]" if args.ridge_use_calibrated else "[raw]"
        lines.append(f"  7) ridge-stacking {tag}  alpha={args.ridge_alpha:.3f} | "
                     f"MAE: {mae_ridge:.3f} | MSE: {mse_ridge:.3f} | "
                     f"weights: bias={b0:.3f}, ax={w_ax:.3f}, cor={w_cor:.3f}, sag={w_sag:.3f}")

    out = "\n".join(lines) + "\n"
    print("\n" + out)

    # Guardar resumen
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(out)

    # CSV opcional con predicciones
    if args.save_preds_csv:
        import csv
        with open(args.save_preds_csv, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            header = [
                "id", "age",
                "axial", "coronal", "sagittal",
                "axial_cal", "coronal_cal", "sagittal_cal",
                "ens_mean", "ens_weighted",
                "ens_cal_mean", "ens_cal_weighted",
                "ens_manual", "ens_auto", "ens_ridge"
            ]
            writer.writerow(header)
            for i, _id in enumerate(ids):
                row = [
                    _id, f"{ages_np[i]:.6f}",
                    f"{preds_raw['axial'][i]:.6f}",
                    f"{preds_raw['coronal'][i]:.6f}",
                    f"{preds_raw['sagittal'][i]:.6f}",
                    f"{preds_cal['axial'][i]:.6f}",
                    f"{preds_cal['coronal'][i]:.6f}",
                    f"{preds_cal['sagittal'][i]:.6f}",
                    f"{ens_mean[i]:.6f}",
                    f"{ens_weighted[i]:.6f}",
                    f"{ens_cal_mean[i]:.6f}",
                    f"{ens_cal_weighted[i]:.6f}",
                    f"{(ens_manual[i] if ens_manual is not None else '')}",
                    f"{(ens_auto[i] if ens_auto is not None else '')}",
                    f"{(ens_ridge[i] if ens_ridge is not None else '')}",
                ]
                writer.writerow(row)

    print(f"Resumen guardado en: {summary_path}")
    if args.save_preds_csv:
        print(f"Predicciones guardadas en: {args.save_preds_csv}")

if __name__ == "__main__":
    main()
