import os
import sys
import torch
import numpy as np
import joblib
import itertools
import pandas as pd
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

# --- IMPORTAR ARQUITECTURA ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from GlobalLocalTransformer_soft_labels import GlobalLocalBrainAge
except ImportError:
    from GlobalLocalTransformer import GlobalLocalBrainAge

# ==============================================================================
# 1. DEFINICIÓN DE CANDIDATOS (POOL COMPLETO 8 MODELOS)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models/final_8k")

# Diccionario completo con tus 8 modelos
CANDIDATES = {
    # AXIAL
    "axial_res18_211403":    {"file": "model_axial_resnet18_JOB_MIXED_211403.pt", "backbone": "resnet18", "plane": "axial"},

    # CORONAL
    "cor_soft_1328":      {"file": "model_coronal_soft_20260106-1328.pt",        "backbone": "resnet18", "plane": "coronal"},
    "cor_res18_201389":   {"file": "model_coronal_resnet18_JOB_201389.pt",       "backbone": "resnet18", "plane": "coronal"},
    "cor_vgg16_1225":     {"file": "model_coronal_20251230-1225.pt",             "backbone": "vgg16",    "plane": "coronal"},
    "cor_res18_211406":   {"file": "model_coronal_resnet18_JOB_MIXED_211406.pt", "backbone": "resnet18", "plane": "coronal"},
    "cor_vgg16_201399":   {"file": "model_coronal_vgg16_old_JOB_201399.pt",      "backbone": "vgg16",    "plane": "coronal"},

    # SAGITTAL
    "sag_res18_1206":     {"file": "model_sagittal_20260105-1206.pt",            "backbone": "resnet18", "plane": "sagittal"},
    "sag_vgg16_211402":   {"file": "model_sagittal_vgg16_JOB_MIXED_211402_MSE.pt", "backbone": "vgg16",    "plane": "sagittal"},
    #"sag_vgg16_250689":   {"file": "model_sagittal_vgg16_JOB_MIXED_250689_5_MSE.pt", "backbone": "vgg16",    "plane": "sagittal"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
DATA_DIR = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset"
DATA_OPENBHB = os.path.join(DATA_DIR, "data/DB_Lautaro_quasiraw/processed_p99")
DATA_OASIS = os.path.join(DATA_DIR, "data/processed_training_olders_T1_p99")
VAL_FILE = os.path.join(DATA_DIR, "scripts/IDs/final_combined/val_ids.txt")

# ==============================================================================
# UTILS
# ==============================================================================
class InferenceDataset(Dataset):
    def __init__(self, ids_file, plane):
        self.plane = plane
        with open(ids_file, "r") as f: self.ids = [line.strip() for line in f if line.strip()]
    def __len__(self): return len(self.ids)
    def _find_file(self, sid):
        p1 = os.path.join(DATA_OPENBHB, self.plane, f"{sid}.pt")
        if os.path.exists(p1): return p1
        return os.path.join(DATA_OASIS, sid, "t1_tensors", f"{self.plane}.pt")
    def __getitem__(self, idx):
        sid = self.ids[idx]
        try:
            path = self._find_file(sid)
            if not os.path.exists(path): return None, None, sid
            d = torch.load(path)
            return d["image"].float(), float(d["age"]), sid
        except: return None, None, sid

def collate_fn(b):
    b = [x for x in b if x[0] is not None]
    if not b: return torch.tensor([]), torch.tensor([]), []
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]), [x[2] for x in b]

def detect_config(path):
    sd = torch.load(path, map_location="cpu")
    keys = sd.keys()
    indices = [int(k.split('.')[1]) for k in keys if "attnlist" in k and ".query" in k]
    nblock = max(indices) + 1 if indices else 8
    out_dim = 1
    if 'gloout.weight' in keys: out_dim = sd['gloout.weight'].shape[0]
    elif 'classifier.weight' in keys: out_dim = sd['classifier.weight'].shape[0]
    return nblock, (out_dim > 1), sd

def get_preds(key, cfg):
    path = os.path.join(MODELS_DIR, cfg['file'])
    if not os.path.exists(path): 
        print(f" [SKIP] No encontrado: {cfg['file']}")
        return None, None, 0
        
    print(f" -> Inferencia: {key} ({cfg['backbone']})...", end=" ", flush=True)
    nblock, is_dist, sd = detect_config(path)

    model = GlobalLocalBrainAge(inplace=5, patch_size=64, step=32, 
                               nblock=nblock, backbone=cfg['backbone']).to(DEVICE)
    if is_dist:
        model.gloout = nn.Linear(model.gloout.in_features, 100).to(DEVICE)
        model.locout = nn.Linear(model.locout.in_features, 100).to(DEVICE)
    
    model.load_state_dict(sd)
    model.eval()

    # Cálculo de parámetros
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Params: {n_params:,}] ", end="", flush=True)

    dl = DataLoader(InferenceDataset(VAL_FILE, cfg['plane']), batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate_fn)
    preds, gts = {}, {}
    bins = torch.arange(100, device=DEVICE).float()
    
    with torch.no_grad():
        for img, age, ids in dl:
            img = img.to(DEVICE)
            out = model(img)[0]
            p = (torch.softmax(out, 1) * bins).sum(1) if is_dist else out.flatten()
            
            # TTA
            if cfg['plane'] in ['axial', 'coronal']:
                out2 = model(torch.flip(img, [-1]))[0]
                p2 = (torch.softmax(out2, 1) * bins).sum(1) if is_dist else out2.flatten()
                p = (p + p2) / 2
            
            for i, sid in enumerate(ids):
                preds[sid] = p[i].item()
                gts[sid] = age[i].item()
    
    print("OK.")
    return preds, gts, n_params

def evaluate_ensemble(model_keys, cache, y_true, common_ids):
    try:
        X = np.column_stack([[cache[k][sid] for sid in common_ids] for k in model_keys])
    except KeyError:
        return 999.9, 999.9, None
    
    # RidgeCV
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5).fit(X, y_true)
    y_pred = ridge.predict(X)
    mae_ridge = mean_absolute_error(y_true, y_pred)

    # Promedio Simple
    y_pred_simple = np.mean(X, axis=1)
    mae_simple = mean_absolute_error(y_true, y_pred_simple)

    return mae_ridge, mae_simple, ridge

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    print(f"--- BÚSQUEDA EXHAUSTIVA DE ENSEMBLES (CON GUARDADO) --- {ts}")
    
    # 1. CACHEO DE PREDICCIONES Y PARÁMETROS
    CACHE = {}
    GT = {}
    PARAMS_MAP = {} # Diccionario para guardar el conteo de params
    COMMON_IDS = None
    
    print("\n>>> CARGANDO MODELOS Y GENERANDO PREDICCIONES...")
    for k, cfg in CANDIDATES.items():
        p, t, num_p = get_preds(k, cfg)
        if p:
            CACHE[k] = p
            PARAMS_MAP[k] = num_p
            if COMMON_IDS is None:
                COMMON_IDS = set(p.keys()); GT = t
            else:
                COMMON_IDS = COMMON_IDS.intersection(set(p.keys()))
    
    final_ids = sorted(list(COMMON_IDS))
    y_true = np.array([GT[sid] for sid in final_ids])
    print(f"\nSujetos comunes: {len(final_ids)}")
    print("-" * 60)

    available_keys = list(CACHE.keys())
    results_2 = []
    results_3 = []
    results_4 = []

    # 1.5 EVALUANDO N=2
    print("\n>>> EVALUANDO ENSEMBLES DE 2 MODELOS...")
    combos_2 = list(itertools.combinations(available_keys, 2))
    for combo in combos_2:
        planes = [CANDIDATES[k]['plane'] for k in combo]
        if 'sagittal' not in planes: continue
        
        mae, mae_simple, model = evaluate_ensemble(combo, CACHE, y_true, final_ids)
        
        results_2.append({
            "Type": "N=2", 
            "Models": combo, 
            "MAE": mae, 
            "MAE_Simple": mae_simple, 
            "Obj": model
        })

    # 2. EVALUANDO N=3
    print("\n>>> EVALUANDO ENSEMBLES DE 3 MODELOS...")
    combos_3 = list(itertools.combinations(available_keys, 3))
    for combo in combos_3:
        planes = [CANDIDATES[k]['plane'] for k in combo]
        if 'sagittal' not in planes: continue
        
        mae, mae_simple, model = evaluate_ensemble(combo, CACHE, y_true, final_ids)
        
        results_3.append({
            "Type": "N=3", 
            "Models": combo, 
            "MAE": mae, 
            "MAE_Simple": mae_simple, 
            "Obj": model
        })

    # 3. EVALUANDO N=4
    print("\n>>> EVALUANDO ENSEMBLES DE 4 MODELOS...")
    combos_4 = list(itertools.combinations(available_keys, 4))
    for combo in combos_4:
        planes = [CANDIDATES[k]['plane'] for k in combo]
        if 'sagittal' not in planes: continue
        
        mae, mae_simple, model = evaluate_ensemble(combo, CACHE, y_true, final_ids)
        
        results_4.append({
            "Type": "N=4", 
            "Models": combo, 
            "MAE": mae, 
            "MAE_Simple": mae_simple, 
            "Obj": model
        })

    # ==========================================================================
    # 4. PROCESAMIENTO DE RESULTADOS
    # ==========================================================================
    df_2 = pd.DataFrame(results_2).sort_values("MAE")
    df_3 = pd.DataFrame(results_3).sort_values("MAE")
    df_4 = pd.DataFrame(results_4).sort_values("MAE")
    
    best_2 = df_2.iloc[0]
    best_3 = df_3.iloc[0]
    best_4 = df_4.iloc[0]

    # Buscar el mejor N=3 con 1 de cada plano
    best_tri_plane = None
    for _, row in df_3.iterrows():
        current_planes = {CANDIDATES[m]['plane'] for m in row['Models']}
        if len(current_planes) == 3:
            best_tri_plane = row
            break
    
    winner = best_2
    if best_3["MAE"] < winner["MAE"]: winner = best_3
    if best_4["MAE"] < winner["MAE"]: winner = best_4

    # ==========================================================================
    # 5. GUARDADO DE ARCHIVOS
    # ==========================================================================
    
    # A) CSV
    csv_path = os.path.join(MODELS_DIR, f"results_grid_search_{ts}.csv")
    df_all = pd.concat([df_2, df_3, df_4])
    df_export = df_all.drop(columns=["Obj"]).copy()
    df_export["Models"] = df_export["Models"].apply(lambda x: " + ".join(x))
    df_export.to_csv(csv_path, index=False)
    print(f"\n📄 CSV guardado en: {csv_path}")

    # B) TXT REPORT + JOBLIBS
    txt_path = os.path.join(MODELS_DIR, f"report_grid_search_{ts}.txt")
    
    print("\n--- GUARDANDO LOS MEJORES MODELOS ---")
    
    with open(txt_path, "w") as f:
        f.write(f"--- REPORTE GRID SEARCH ENSEMBLES ---\n")
        f.write(f"Script: {os.path.basename(__file__)}\n")
        f.write(f"Timestamp: {ts}\n\n")

        # --- SECCIÓN DE PARÁMETROS SOLICITADA ---
        f.write("=== REPORTE DE PARÁMETROS ===\n")
        
        # 1. Los 3 modelos específicos solicitados
        target_keys = ["axial_res18_211403", "cor_vgg16_1225", "sag_vgg16_211402"]
        f.write("--- Modelos de Interés ---\n")
        for tk in target_keys:
            if tk in PARAMS_MAP:
                f.write(f"{tk}: {PARAMS_MAP[tk]:,} parámetros\n")
            else:
                f.write(f"{tk}: NO ENCONTRADO/CARGADO\n")
        
        # 2. El resto (opcional, pero útil)
        f.write("\n--- Resto de Modelos ---\n")
        for k, v in PARAMS_MAP.items():
            if k not in target_keys:
                f.write(f"{k}: {v:,} parámetros\n")
        f.write("\n" + "="*30 + "\n\n")
        # ----------------------------------------

        f.write("=== GANADOR ABSOLUTO (RIDGE) ===\n")
        f.write(f"Tipo: {winner['Type']}\n")
        f.write(f"MAE Ridge:  {winner['MAE']:.6f}\n")
        f.write(f"MAE Simple: {winner['MAE_Simple']:.6f}\n")
        f.write(f"Equipo: {winner['Models']}\n\n")

        items_to_save = [
            ("N=2", best_2, ""),
            ("N=3", best_3, ""),
            ("N=4", best_4, "")
        ]

        if best_tri_plane is not None:
            if best_tri_plane['Models'] != best_3['Models']:
                items_to_save.append(("N=3 (Tri-Plane)", best_tri_plane, "_TRIPLANE"))
            else:
                f.write("NOTA: El mejor N=3 ya es un Tri-Plane (Ax+Cor+Sag).\n\n")

        for n_type, row_data, suffix in items_to_save:
            mae_r = row_data['MAE']
            mae_s = row_data['MAE_Simple']
            models_list = row_data['Models']
            
            joblib_name = f"best_ensemble_{n_type.split()[0]}{suffix}_{mae_r:.4f}_{ts}.joblib"
            joblib_path = os.path.join(MODELS_DIR, joblib_name)
            
            artifact = {
                "model": row_data["Obj"],
                "input_keys": models_list,
                "mae_ridge": mae_r,
                "mae_simple": mae_s,
                "timestamp": ts,
                "model_paths": [CANDIDATES[k]['file'] for k in models_list]
            }
            joblib.dump(artifact, joblib_path)
            print(f"💾 Guardado {n_type}: {joblib_name}")

            f.write(f"=== MEJOR COMBINACIÓN {n_type} ===\n")
            f.write(f"Modelos: {models_list}\n")
            f.write(f"MAE Ridge CV:    {mae_r:.6f}\n")
            f.write(f"MAE Prom Simple: {mae_s:.6f}\n")
            f.write(f"Path Joblib: {joblib_path}\n\n")

        f.write("=== TOP 10 (N=2) ===\n")
        for i, row in df_2.head(10).iterrows():
            f.write(f"{i+1}. Ridge: {row['MAE']:.4f} | {row['Models']}\n")

        f.write("\n=== TOP 10 (N=3) ===\n")
        for i, row in df_3.head(10).iterrows():
            f.write(f"{i+1}. Ridge: {row['MAE']:.4f} | {row['Models']}\n")
            
        f.write("\n=== TOP 10 (N=4) ===\n")
        for i, row in df_4.head(10).iterrows():
            f.write(f"{i+1}. Ridge: {row['MAE']:.4f} | {row['Models']}\n")

    print(f"📝 Reporte TXT guardado en: {txt_path}")
    print("="*80)

    # C) JOBLIB FINAL
    artifact = {
        "model": winner["Obj"],
        "input_keys": winner["Models"],
        "mae": winner["MAE"],
        "timestamp": ts,
        "model_paths": [CANDIDATES[k]['file'] for k in winner["Models"]]
    }
    
    filename = f"best_ensemble_FINAL_{winner['MAE']:.4f}_{ts}.joblib"
    out_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(artifact, out_path)
    
    print("\n" + "="*80)
    print(f"🥇 GANADOR FINAL: MAE {winner['MAE']:.4f}")
    print(f"💾 Modelo guardado en: {os.path.abspath(out_path)}")
    print("="*80)