#!/usr/bin/env python3
# grid_search_all_combos.py
import os
import re
import csv
import itertools
import subprocess
from datetime import datetime
from pathlib import Path

# --------------------------
# Configuración
# --------------------------
EVAL_SCRIPT = "evaluate_hybrid_ensemble.py"
MODELS_DIR = Path("../models")
EVAL_DIR = Path("../eval")
PATCH_SIZE = 64
STEP = 32

# Candidatos por plano: (nombre_amistoso, filename)
CANDIDATES = {
    "axial": [
        ("vgg16_20250822-1437",  "model_axial_20250822-1437.pt"),
        ("vgg16_20250814-0853",  "model_axial_20250814-0853.pt"),
        ("resnet18_20250827-1025","model_axial_20250827-1025.pt"),
    ],
    "coronal": [
        ("vgg16_20250822-1437",  "model_coronal_20250822-1437.pt"),
        ("vgg16_20250814-0853",  "model_coronal_20250814-0853.pt"),
        ("resnet18_20250827-1025","model_coronal_20250827-1025.pt"),
    ],
    "sagittal": [
        ("vgg16_20250822-1437",  "model_sagittal_20250822-1437.pt"),
        ("vgg16_20250814-0853",  "model_sagittal_20250814-0853.pt"),
        ("resnet18_20250827-1025","model_sagittal_20250827-1025.pt"),
    ],
}

# Granularidad del auto-search (0..1)
WC_MIN, WC_MAX, WC_STEPS = 0.0, 1.0, 101
RIDGE_ALPHAS = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6]

# Regex para parsear métricas/weights del stdout
RE_AUTO = re.compile(r"auto-search.*\|\s*MAE:\s*([0-9.]+)\s*\|\s*MSE:\s*([0-9.]+)", re.I)
RE_RIDGE = re.compile(r"ridge-stacking\s*\[.*?\].*?MAE:\s*([0-9.]+)\s*\|\s*MSE:\s*([0-9.]+)\s*\|\s*weights:\s*bias=([-\d.]+),\s*ax=([-\d.]+),\s*cor=([-\d.]+),\s*sag=([-\d.]+)", re.I)
RE_AUTO_W = re.compile(r"auto-search\s*\(([^)]+)\)", re.I)  # para capturar (wa,wc,ws)

def run_combo(ax, co, sa, timestamp, out_dir: Path):
    outputs = []
    for alpha in RIDGE_ALPHAS:
        tag = f"{timestamp}_ax-{ax[0]}_co-{co[0]}_sa-{sa[0]}_ridgeA{alpha}"
        summary_path = out_dir / f"val_summary_{tag}.txt"
        preds_path   = out_dir / f"val_preds_{tag}.csv"

        cmd = [
            "python", EVAL_SCRIPT,
            "--axial-model",    str(MODELS_DIR / ax[1]),
            "--coronal-model",  str(MODELS_DIR / co[1]),
            "--sagittal-model", str(MODELS_DIR / sa[1]),
            "--patch-size", str(PATCH_SIZE),
            "--step", str(STEP),
            "--save-summary", str(summary_path),
            "--save-preds-csv", str(preds_path),
            "--global-only",
            "--auto-search", "--wc-min", str(WC_MIN), "--wc-max", str(WC_MAX), "--wc-steps", str(WC_STEPS),
            "--ridge-stack", "--ridge-alpha", str(alpha), "--ridge-use-calibrated",
        ]

        print("\n>>> Ejecutando:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout
        if proc.returncode != 0:
            print("   [ERROR] código", proc.returncode)
            print(proc.stderr)
            continue

        # Parseo iguales que antes...
        auto_mae = auto_mse = None
        auto_weights = None
        m = RE_AUTO.search(stdout)
        if m:
            auto_mae, auto_mse = float(m.group(1)), float(m.group(2))
            mw = RE_AUTO_W.search(stdout)
            auto_weights = mw.group(1).strip() if mw else None

        ridge_mae = ridge_mse = None
        ridge_bias = ridge_ax = ridge_co = ridge_sa = None
        m = RE_RIDGE.search(stdout)
        if m:
            ridge_mae, ridge_mse = float(m.group(1)), float(m.group(2))
            ridge_bias, ridge_ax, ridge_co, ridge_sa = map(float, m.groups()[2:])

        outputs.append({
            "tag": tag,
            "axial": ax[0], "coronal": co[0], "sagittal": sa[0],
            "alpha": alpha,
            "summary": str(summary_path),
            "preds": str(preds_path),
            "auto_mae": auto_mae, "auto_mse": auto_mse, "auto_weights": auto_weights,
            "ridge_mae": ridge_mae, "ridge_mse": ridge_mse,
            "ridge_bias": ridge_bias, "ridge_ax": ridge_ax, "ridge_cor": ridge_co, "ridge_sag": ridge_sa,
            "stdout_tail": "\n".join(stdout.strip().splitlines()[-20:]),
        })
    return outputs


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    # --- subcarpeta única para esta corrida (todo .csv/.txt va acá) ---
    RUN_DIR = EVAL_DIR / f"grid_{timestamp}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Guardando todos los .csv y .txt en: {RUN_DIR.resolve()}")

    # Chequeo existencia de modelos
    missing = []
    for plane, items in CANDIDATES.items():
        for name, fname in items:
            path = MODELS_DIR / fname
            if not path.exists():
                missing.append(str(path))
    if missing:
        print("ERROR: faltan estos checkpoints:\n  - " + "\n  - ".join(missing))
        return

    results = []
    combos = itertools.product(CANDIDATES["axial"], CANDIDATES["coronal"], CANDIDATES["sagittal"])
    for ax, co, sa in combos:
        res_list = run_combo(ax, co, sa, timestamp, RUN_DIR) or []
        for res in res_list:
            # elegir mejor método para ESTA (combo, alpha)
            best_method = None
            best_mae = float("inf")
            best_info = {}

            if res["auto_mae"] is not None and res["auto_mae"] < best_mae:
                best_method = "auto-search(cal)"
                best_mae = res["auto_mae"]
                best_info = {"mse": res["auto_mse"], "weights": res["auto_weights"]}

            if res["ridge_mae"] is not None and res["ridge_mae"] < best_mae:
                best_method = f"ridge(cal) α={res['alpha']}"
                best_mae = res["ridge_mae"]
                best_info = {
                    "mse": res["ridge_mse"],
                    "weights": f"bias={res['ridge_bias']:.3f}, ax={res['ridge_ax']:.3f}, cor={res['ridge_cor']:.3f}, sag={res['ridge_sag']:.3f}"
                }

            res["best_method"] = best_method
            res["best_mae"] = best_mae
            res["best_info"] = best_info
            results.append(res)
            print(f"   → Mejor método: {best_method} | MAE={best_mae:.3f} | info={best_info}")

    # Guardar ranking CSV (también en RUN_DIR)
    rank_csv = RUN_DIR / f"grid_rank_{timestamp}.csv"
    with open(rank_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag","alpha","axial","coronal","sagittal","best_method","best_mae","best_mse_or_none",
            "best_weights_or_none","summary_path","preds_path"
        ])
        for r in sorted(results, key=lambda x: x["best_mae"]):
            w.writerow([
                r["tag"], r.get("alpha"), r["axial"], r["coronal"], r["sagittal"],
                r.get("best_method"), f"{r.get('best_mae'):.3f}",
                f"{r['best_info'].get('mse'):.3f}" if r["best_info"].get("mse") is not None else "",
                r["best_info"].get("weights",""),
                r["summary"], r["preds"]
            ])

    # Reporte final en consola
    if results:
        best_overall = min(results, key=lambda x: x["best_mae"])
        print("\n================  TOP-1 GLOBAL  ================")
        print(f"Combo: ax={best_overall['axial']} | co={best_overall['coronal']} | sa={best_overall['sagittal']}")
        print(f"Mejor: {best_overall['best_method']} | MAE={best_overall['best_mae']:.3f} | info={best_overall['best_info']}")
        print(f"Resumen: {best_overall['summary']}")
        print(f"Preds  : {best_overall['preds']}")
        print(f"Ranking CSV: {rank_csv}")
    else:
        print("No se obtuvieron resultados.")

if __name__ == "__main__":
    main()
