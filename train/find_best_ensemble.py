#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
import argparse

# ==============================================================================
# 0. SETUP E IMPORTACIONES DEL ENTORNO DE ENTRENAMIENTO
# ==============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Enmascaramiento del sys.argv para evitar que el parser global 
    # de train_improved intercepte los argumentos de consola del ensamble.
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    from train_improved import BrainAgeDataset, build_dataloader, NUM_CLASSES, INPLACE, PATCH_SIZE, STEP
    from GlobalLocalTransformer_soft_labels import GlobalLocalBrainAge
    
    # Restauración topológica del vector de argumentos
    sys.argv = original_argv
except ImportError:
    sys.exit("CRITICAL: Ejecutá este script desde el directorio donde reside train_improved.py")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="both", choices=["extract_only", "search_only", "both"])
parser.add_argument("--n_jobs", type=int, default=-1)
parser.add_argument("--tag", type=str, required=True, help="Identificador topológico para aislar caché I/O y reportes")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_TXT = os.path.join(os.path.dirname(__file__), "../IDs/final_combined/v2_saneada/val_ids.txt")
MODELS_DIR = "/scratch/nfuentes/brain_age_project/openBHB_dataset/scripts/models"

# --- CONFIGURACIÓN GLOBAL DINÁMICA ---
ENSEMBLE_TAG = args.tag

# --- WHITELIST ESTRICTA DE CARPETAS DE INFERENCIA ---
# TARGET_BATCHES = [
#     "BATCH1_JUNIO_ipw_583162", "BATCH2_JUNIO_ipw_583163",
#     "BATCH3_JUNIO_ipw_583164", "BATCH4_JUNIO_ipw_583165",
#     "BATCH5_JUNIO_ipw_583167", "BATCH6_JUNIO_ipw_583168",
#     "BATCH7_JUNIO_ipw_583169", "BATCH8_JUNIO_ipw_583170",
#     "BATCH9_JUNIO_ipw_583171", "BATCH10_JUNIO_ipw_583172",
#     "BATCH11_JUNIO_ipw_583173", "BATCH12_JUNIO_ipw_583174",
#     "BATCH13_JUNIO_ipw_583175", "BATCH14_JUNIO_ipw_583176",
#     "BATCH15_JUNIO_ipw_583177", "BATCH16_JUNIO_ipw_583178",
#     "BATCH17_JUNIO_ipw_583179", "BATCH18_JUNIO_ipw_583180",
#     "BATCH19_JUNIO_ipw_583181", "BATCH20_JUNIO_ipw_583182",
#     "BATCH21_JUNIO_ipw_583183", "BATCH22_JUNIO_ipw_583184",
#     "BATCH23_JUNIO_ipw_583185", "BATCH24_JUNIO_ipw_583186"
# ]
TARGET_BATCHES = [
    "BATCH1_JUNIO_ipw_opt_624618"  
]
CACHE_FILE = os.path.join(MODELS_DIR, f"validation_predictions_cache_{ENSEMBLE_TAG}.npz")
RESULTS_DIR = os.path.join(MODELS_DIR, "ensemble_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. UTILIDADES DE INFERENCIA
# ==============================================================================
def decode_age(logits):
    probs = F.softmax(logits, dim=1)
    x = torch.arange(probs.shape[1], device=probs.device).float()
    return (probs * x).sum(dim=1)

def parse_model_metadata(filename):
    """Extrae hiperparámetros de la topología del nombre del archivo."""
    # Ejemplo: best_model_axial_axial_resnet18_soft_n8.pt
    basename = os.path.basename(filename).replace("best_model_", "").replace(".pt", "")
    parts = basename.split("_")
    
    # Manejo de la duplicación del plano en el naming convention
    plane = parts[0]
    backbone = parts[2]
    loss_type = parts[3]
    nblock = int(parts[4].replace("n", ""))
    
    return plane, backbone, loss_type, nblock

def extract_predictions(model_path, dataloader, metadata):
    """Realiza el forward pass sobre el manifold de validación."""
    plane, backbone, loss_type, nblock = metadata
    
    n_classes = NUM_CLASSES if loss_type == 'soft' else 1
    
    model = GlobalLocalBrainAge(
        inplace=INPLACE, patch_size=PATCH_SIZE, step=STEP, nblock=nblock,
        backbone=backbone, num_classes=n_classes, drop_rate=0.0
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    preds_orig_list = []
    preds_tta_list = []
    targets = []
    
    with torch.no_grad():
        for imgs, _, ages_real in dataloader:
            imgs = imgs.to(DEVICE)
            
            # 1. Forward Pass Original f(x)
            outputs = model(imgs)
            batch_preds_orig = []
            for out_head in outputs:
                if loss_type == 'soft':
                    batch_preds_orig.append(decode_age(out_head))
                else:
                    batch_preds_orig.append(out_head.view(-1))
            avg_preds_orig = torch.stack(batch_preds_orig).mean(dim=0)
            
            # 2. TTA (Test-Time Augmentation)
            if plane in ['axial', 'coronal']:
                imgs_flip = torch.flip(imgs, dims=[3])
                outputs_flip = model(imgs_flip)
                
                batch_preds_flip = []
                for out_head in outputs_flip:
                    if loss_type == 'soft':
                        batch_preds_flip.append(decode_age(out_head))
                    else:
                        batch_preds_flip.append(out_head.view(-1))
                avg_preds_flip = torch.stack(batch_preds_flip).mean(dim=0)
                
                final_preds_tta = (avg_preds_orig + avg_preds_flip) / 2.0
            else:
                final_preds_tta = avg_preds_orig
                
            preds_orig_list.extend(avg_preds_orig.cpu().numpy())
            preds_tta_list.extend(final_preds_tta.cpu().numpy())
            targets.extend(ages_real.numpy())
            
    return np.array(preds_orig_list), np.array(preds_tta_list), np.array(targets)

# ==============================================================================
# 2. GENERACIÓN DE MATRIZ DE PREDICCIONES
# ==============================================================================
def get_prediction_matrix():
    if os.path.exists(CACHE_FILE):
        print(f"[*] Cargando matriz de inferencias desde caché: {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        return data['Y_hat'][()], data['Y_hat_orig'][()], data['y']
        
    print(f"[*] Generando matriz de inferencias global. Mapeando estrictamente las 24 carpetas whitelist...")
    
    model_files = []
    for batch_folder in TARGET_BATCHES:
        folder_path = os.path.join(MODELS_DIR, batch_folder)
        
        if not os.path.exists(folder_path):
            print(f"[!] Advertencia I/O: La carpeta {batch_folder} no existe en {MODELS_DIR}.")
            continue
            
        for f in os.listdir(folder_path):
            if f.startswith("best_model_") and f.endswith(".pt"):
                model_files.append(os.path.join(folder_path, f))
    
    if not model_files:
        sys.exit("[!] ERROR: No se aislaron modelos válidos en las carpetas proporcionadas.")
        
    print(f"[*] Indexación completada: {len(model_files)} modelos aislados exitosamente para inferencia.")
        
    Y_hat_dict = {}
    Y_hat_orig_dict = {}
    y_true = None
    
    dataloaders = {
        "axial": build_dataloader(BrainAgeDataset(VAL_TXT, "axial", 'val'), "val", 0),
        "coronal": build_dataloader(BrainAgeDataset(VAL_TXT, "coronal", 'val'), "val", 0),
        "sagittal": build_dataloader(BrainAgeDataset(VAL_TXT, "sagittal", 'val'), "val", 0)
    }
    
    for m_path in tqdm(model_files, desc="Inferiendo Modelos"):
        metadata = parse_model_metadata(m_path)
        plane = metadata[0]
        
        preds_orig, preds_tta, targets = extract_predictions(m_path, dataloaders[plane], metadata)
        Y_hat_dict[m_path] = preds_tta
        Y_hat_orig_dict[m_path] = preds_orig
        
        if y_true is None:
            y_true = targets
            
    np.savez(CACHE_FILE, Y_hat=Y_hat_dict, Y_hat_orig=Y_hat_orig_dict, y=y_true)
    return Y_hat_dict, Y_hat_orig_dict, y_true

# ==============================================================================
# 3. RIDGE STACKING & EVALUACIÓN COMBINATORIA
# ==============================================================================
def evaluate_combo(combo, Y_hat_dict, y_true, cv_splits):
    """Resuelve el sistema Tikhonov para un subset de modelos."""
    if len(combo) == 1:
        mae = mean_absolute_error(y_true, Y_hat_dict[combo[0]])
        return combo, mae, None, None
        
    # Construcción de la matriz Y_hat para el combo actual
    X = np.column_stack([Y_hat_dict[m] for m in combo])
    
    # Búsqueda de hiperparámetros en el espacio logarítmico
    alphas = np.logspace(-3, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv_splits, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X, y_true)
    
    # CV MAE inverso (RidgeCV retorna valores negativos para la métrica)
    best_mae = -ridge_cv.best_score_ 
    best_alpha = ridge_cv.alpha_
    
    return combo, best_mae, best_alpha, ridge_cv

def process_combinations(combinations_list, Y_hat_dict, y_true):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(evaluate_combo)(c, Y_hat_dict, y_true, cv) for c in tqdm(combinations_list, desc="Evaluando")
    )
    # Ordenamiento topológico por mínimo MAE
    results.sort(key=lambda x: x[1])
    return results[0] # Retorna el óptimo (combo, mae, alpha, model)

def main():
    print(f"--- INICIANDO RIDGE STACKING (Modo: {args.mode}) ---")
    Y_hat_dict, Y_hat_orig_dict, y_true = get_prediction_matrix()
    
    if args.mode == "extract_only":
        print("\n[OK] Extracción finalizada y persistida en caché. Saliendo (extract_only).")
        return
        
    all_models = list(Y_hat_dict.keys())
    
    # Precalculamos el MAE individual para inyectarlo en el reporte
    individual_maes = {}
    for m in all_models:
        individual_maes[m] = {
            "tta": mean_absolute_error(y_true, Y_hat_dict[m]),
            "orig": mean_absolute_error(y_true, Y_hat_orig_dict[m])
        }
    
    # Subdivisión por planos para combinatoria triplanar
    models_by_plane = {"axial": [], "coronal": [], "sagittal": []}
    for m in all_models:
        p = parse_model_metadata(m)[0]
        models_by_plane[p].append(m)
        
    categories = {}
    
    print("\n[*] Categoría: Mejor Modelo Individual (N=1)")
    combos_1 = [(m,) for m in all_models]
    categories['Best_1_Model'] = process_combinations(combos_1, Y_hat_dict, y_true)
    
    print("\n[*] Categoría: Mejor Par (N=2)")
    combos_2 = list(combinations(all_models, 2))
    categories['Best_2_Models'] = process_combinations(combos_2, Y_hat_dict, y_true)
    
    print("\n[*] Categoría: Mejor Triplanar Estricto Global (1Ax, 1Cor, 1Sag)")
    combos_3_tri = list(product(models_by_plane["axial"], models_by_plane["coronal"], models_by_plane["sagittal"]))
    categories['Best_3_Triplanar_Global'] = process_combinations(combos_3_tri, Y_hat_dict, y_true)
    
    print("\n[*] Categoría: Mejores 3 Modelos (Cualquier Plano)")
    combos_3_any = list(combinations(all_models, 3))
    categories['Best_3_Any'] = process_combinations(combos_3_any, Y_hat_dict, y_true)
    
    # --- NUEVA SECCIÓN: TRIPLANAR AISLADO POR BACKBONE ---
    print("\n[*] Categoría: Mejor Triplanar por Backbone (Arquitectura Homogénea)")
    unique_backbones = set(parse_model_metadata(m)[1] for m in all_models)
    
    for bb in unique_backbones:
        bb_models_axial = [m for m in models_by_plane["axial"] if parse_model_metadata(m)[1] == bb]
        bb_models_coronal = [m for m in models_by_plane["coronal"] if parse_model_metadata(m)[1] == bb]
        bb_models_sagittal = [m for m in models_by_plane["sagittal"] if parse_model_metadata(m)[1] == bb]
        
        if bb_models_axial and bb_models_coronal and bb_models_sagittal:
            bb_combos_tri = list(product(bb_models_axial, bb_models_coronal, bb_models_sagittal))
            cat_name = f'Best_3_Triplanar_{bb.upper()}'
            print(f"  -> Evaluando {bb.upper()} ({len(bb_combos_tri)} combinaciones)...")
            categories[cat_name] = process_combinations(bb_combos_tri, Y_hat_dict, y_true)
        else:
            print(f"  -> Omitiendo {bb.upper()}: No hay modelos suficientes para formar un ensamble triplanar.")

    # --- NUEVA SECCIÓN: TRIPLANAR AISLADO POR LOSS TYPE ---
    print("\n[*] Categoría: Mejor Triplanar por Función de Costo (Loss Homogénea)")
    unique_losses = set(parse_model_metadata(m)[2] for m in all_models)
    
    for lt in unique_losses:
        lt_models_axial = [m for m in models_by_plane["axial"] if parse_model_metadata(m)[2] == lt]
        lt_models_coronal = [m for m in models_by_plane["coronal"] if parse_model_metadata(m)[2] == lt]
        lt_models_sagittal = [m for m in models_by_plane["sagittal"] if parse_model_metadata(m)[2] == lt]
        
        if lt_models_axial and lt_models_coronal and lt_models_sagittal:
            lt_combos_tri = list(product(lt_models_axial, lt_models_coronal, lt_models_sagittal))
            cat_name = f'Best_3_Triplanar_{lt.upper()}'
            print(f"  -> Evaluando {lt.upper()} ({len(lt_combos_tri)} combinaciones)...")
            categories[cat_name] = process_combinations(lt_combos_tri, Y_hat_dict, y_true)
        else:
            print(f"  -> Omitiendo {lt.upper()}: No hay modelos suficientes para formar un ensamble triplanar.")

    # ==============================================================================
    # 4. FUNCION DE EXPORTACIÓN MODULARIZADA
    # ==============================================================================
    def export_current_results(cats_dict, suffix_label=""):
        records = []
        print("\n" + "="*80)
        print(f"REPORTE DE VECTORES ÓPTIMOS {suffix_label}")
        print("="*80)
        
        for cat_name, (combo, mae, alpha, ridge_model) in cats_dict.items():
            clean_names = []
            sizes_mb = []
            total_size_mb = 0.0
            
            for m in combo:
                clean_names.append(os.path.basename(m).replace("best_model_", "").replace(".pt", ""))
                size = os.path.getsize(m) / (1024 * 1024)
                sizes_mb.append(f"{size:.2f}")
                total_size_mb += size
                
            print(f"\n> CATEGORÍA: {cat_name}")
            print(f"  MAE Validación : {mae:.4f} años")
            if alpha is not None:
                print(f"  Alpha Óptimo   : {alpha:.4f}")
                
            print(f"  Peso Total Ens.: {total_size_mb:.2f} MB")
            for i, (m_path, size_str) in enumerate(zip(combo, sizes_mb)):
                mae_tta = individual_maes[m_path]["tta"]
                mae_orig = individual_maes[m_path]["orig"]
                print(f"  M{i+1}: {m_path} ({size_str} MB | MAE w/TTA: {mae_tta:.4f} | MAE w/o TTA: {mae_orig:.4f})")
                
            if ridge_model is not None:
                joblib_path = os.path.join(RESULTS_DIR, f"{cat_name}_{ENSEMBLE_TAG}_ridge.joblib")
                joblib.dump(ridge_model, joblib_path)
                print(f"  Stacker guardado en : {joblib_path}")
                
            records.append({
                "Categoria": cat_name,
                "MAE": mae,
                "Alpha": alpha,
                "Peso_Total_MB": round(total_size_mb, 2),
                "Pesos_Individuales_MB": " | ".join(sizes_mb),
                "Modelos": " | ".join(clean_names)
            })

        df = pd.DataFrame(records)
        csv_path = os.path.join(RESULTS_DIR, f"ensemble_metrics_{ENSEMBLE_TAG}_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[*] Tabla de resultados exportada/actualizada en: {csv_path}")

    # --- PRIMERA EXPORTACIÓN DE SEGURIDAD (N<=3) ---
    export_current_results(categories, suffix_label="(N<=3 PREDOMINANTE)")

    # ==============================================================================
    # 5. EXPLOSIÓN COMBINATORIA (N=4) AL FINAL
    # ==============================================================================
    print("\n" + "="*80)
    print("[*] INICIANDO CATEGORÍA DE OVERHEAD ALTO: Mejores 4 Modelos (Cualquier Plano)")
    print("="*80)
    combos_4_any = list(combinations(all_models, 4))
    categories['Best_4_Any'] = process_combinations(combos_4_any, Y_hat_dict, y_true)

    # --- EXPORTACIÓN FINAL DEFINITIVA ---
    export_current_results(categories, suffix_label="(COMPLETO)")

if __name__ == "__main__":
    main()
