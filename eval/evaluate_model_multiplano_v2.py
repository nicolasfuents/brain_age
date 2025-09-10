# evaluate_model_multiplano_v2.py
# Evalúa 3 modelos + ensemble sobre val_ids.txt

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import sys
import os
import re  # <-- añadido
# Agregar el directorio padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GlobalLocalTransformer import GlobalLocalBrainAge
from datetime import datetime
import argparse

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#INPLACE = 5
PATCH_SIZE = 64
STEP = 32
#NBLOCK = 10
#BACKBONE = "vgg16"
IDS_FILE = "val_ids.txt"

# Nombres manuales de modelos entrenados
#MODEL_FILES = {
#    "axial": "model_axial_20250827-1025.pt",
#    "coronal": "model_coronal_20250827-1025.pt",
#    "sagittal": "model_sagittal_20250827-1025.pt",
#}

# -----------------------------
# DATASET
# -----------------------------
class BrainAgeDataset(Dataset):
    def __init__(self, ids_file, data_dirs):
        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.data_dirs = data_dirs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        samples = [torch.load(os.path.join(data_dir, f"{id}.pt"))["image"] for data_dir in self.data_dirs]
        age = torch.load(os.path.join(self.data_dirs[0], f"{id}.pt"))["age"]
        return samples, age, id

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="ID del experimento (ej: 20250827-1025)")
    args = parser.parse_args()

    MODEL_FILES = {
        "axial": f"model_axial_{args.id}.pt",
        "coronal": f"model_coronal_{args.id}.pt",
        "sagittal": f"model_sagittal_{args.id}.pt",
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ID_PATH = os.path.join(BASE_DIR, f"../IDs/{IDS_FILE}")
    DATA_PATHS = {
        "axial": os.path.join(BASE_DIR, "../../data/processed/axial"),
        "coronal": os.path.join(BASE_DIR, "../../data/processed/coronal"),
        "sagittal": os.path.join(BASE_DIR, "../../data/processed/sagittal"),
    }
    
    OUTPUT_PATH = os.path.join(BASE_DIR, "val_summary.txt")

    # ====== AUTODETECCIÓN DE BACKBONE, INPLACE y NBLOCK ======
    sample_model_path = os.path.join(BASE_DIR, f"../models/{MODEL_FILES['axial']}")
    ckpt = torch.load(sample_model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Detección robusta de backbone (ResNet vs VGG)
    if any(k.startswith("global_feat.stem") or "global_feat.layer1.0" in k for k in state_dict.keys()):
        BACKBONE = "resnet18"   # tus checkpoints muestran patrón 2-2-2-2 (ResNet18)
    elif any(".conv33." in k for k in state_dict.keys()):
        BACKBONE = "vgg16"
    else:
        BACKBONE = "vgg8"
        # Inferir INPLACE de forma robusta: tomar el menor in_channels entre todas las conv2d
    conv_weight_shapes = [v.shape for k, v in state_dict.items()
                          if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 4]
    if not conv_weight_shapes:
        raise RuntimeError("No se encontraron pesos de convolución para inferir INPLACE.")
    INPLACE = int(min(s[1] for s in conv_weight_shapes))

    idxs = {int(m.group(1)) for k in state_dict.keys()
            for m in [re.search(r"^attnlist\.(\d+)\.", k)] if m}
    NBLOCK = (max(idxs) + 1) if idxs else 0

    print(f"\nParámetros del framework -> BACKBONE={BACKBONE}, INPLACE={INPLACE}, NBLOCK={NBLOCK}", flush=True)
    
    print("Evaluación de los modelos en progreso...\n", flush=True) 

    dataset = BrainAgeDataset(ID_PATH, list(DATA_PATHS.values()))
    dataloader = DataLoader(dataset, batch_size=1)

    models = {}
    used_model_files = {}  # <--- NUEVO

    for plane in ["axial", "coronal", "sagittal"]:
        model_path = os.path.join(BASE_DIR, f"../models/{MODEL_FILES[plane]}")
        model = GlobalLocalBrainAge(inplace=INPLACE, patch_size=PATCH_SIZE,
                                    step=STEP, nblock=NBLOCK, backbone=BACKBONE).to(DEVICE)
        _sd = torch.load(model_path, map_location=DEVICE)
        # Aceptar distintos formatos de checkpoint
        if hasattr(_sd, "state_dict"):              # por si guardaron el modelo completo
            _sd = _sd.state_dict()
        elif isinstance(_sd, dict) and "state_dict" in _sd and isinstance(_sd["state_dict"], dict):
            _sd = _sd["state_dict"]

        missing_unexpected = model.load_state_dict(_sd, strict=False)  # <-- clave: strict=False
        if hasattr(missing_unexpected, "missing_keys") or hasattr(missing_unexpected, "unexpected_keys"):
            mk = getattr(missing_unexpected, "missing_keys", [])
            uk = getattr(missing_unexpected, "unexpected_keys", [])
            
        # --- calibración rápida de BatchNorm (mejora mucho si el ckpt no trae running stats) ---
        CALIB_N = 32   # podés bajar a 10 si querés hacerlo más corto
        model.train()
        with torch.no_grad():
            idx_plane = ["axial", "coronal", "sagittal"].index(plane)
            c = 0
            for imgs, _, _ in dataloader:
                x_cal = imgs[idx_plane].to(DEVICE)
                _ = model(x_cal)  # forward para actualizar BN
                c += 1
                if c >= CALIB_N:
                    break
        # --- fin calibración ---
        
        model.eval()
        models[plane] = model
        used_model_files[plane] = MODEL_FILES[plane]  

    results = {p: [] for p in models}
    ensemble_preds = []
    targets = []
    ids = []

    with torch.no_grad():
        for images_list, age, id_ in dataloader:
            targets.append(age.item())
            ids.append(id_[0])
            preds = []
            for i, plane in enumerate(models):
                x = images_list[i].to(DEVICE)
                out = models[plane](x)
                if isinstance(out, list):
                    # out = [B×1, B×1, ...]; concatenamos y promediamos en la dim de "cabezas/patches"
                    out = torch.cat(out, dim=1).mean(dim=1, keepdim=True)
                pred = out.squeeze().cpu().item()

                results[plane].append(pred)
                preds.append(pred)
            ensemble_preds.append(np.mean(preds))

    targets = np.array(targets)
    ensemble_preds = np.array(ensemble_preds)
    metrics = {}

    for plane in models:
        preds = np.array(results[plane])
        mae = mean_absolute_error(targets, preds)
        mse = mean_squared_error(targets, preds)
        corr, _ = pearsonr(targets, preds)
        acc_5 = np.mean(np.abs(preds - targets) < 5) * 100
        metrics[plane] = (mae, mse, corr, acc_5)

    mae_e = mean_absolute_error(targets, ensemble_preds)
    mse_e = mean_squared_error(targets, ensemble_preds)
    corr_e, _ = pearsonr(targets, ensemble_preds)
    acc_e = np.mean(np.abs(ensemble_preds - targets) < 5) * 100

    # -----------------------------
    # GUARDAR PREDICCIONES PARA GRAFICAR
    # -----------------------------
    pred_dir = os.path.join(BASE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Guardar true una sola vez (es igual para todos)
    np.savetxt(os.path.join(pred_dir, "val_true_axial.txt"), targets)
    np.savetxt(os.path.join(pred_dir, "val_true_coronal.txt"), targets)
    np.savetxt(os.path.join(pred_dir, "val_true_sagittal.txt"), targets)
    np.savetxt(os.path.join(pred_dir, "val_true_ensemble.txt"), targets)

    # Guardar predicciones por modelo
    for plane in models:
        np.savetxt(os.path.join(pred_dir, f"val_pred_{plane}.txt"), results[plane])

    # Guardar ensemble
    np.savetxt(os.path.join(pred_dir, "val_pred_ensemble.txt"), ensemble_preds)

    # Guardar en scripts/eval
    OUTPUT_PATH = os.path.join(BASE_DIR, "val_summary.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(targets)

    with open(OUTPUT_PATH, "a") as f:
        f.write(f"\n==================== {timestamp} ====================\n")
        # Guardar parámetros del framework
        f.write(f"⚙️ Parámetros del framework -> BACKBONE={BACKBONE}, INPLACE={INPLACE}, NBLOCK={NBLOCK}\n")  
        # Guardar también los nombres de los modelos usados
        f.write("Modelos utilizados:\n")                        
        for plane, fname in used_model_files.items():              
            f.write(f"- {plane:<8}: {fname}\n")                    
        for plane in models:
            mae, mse, corr, _ = metrics[plane]
            preds = np.array(results[plane])
            count_within_5 = np.sum(np.abs(preds - targets) < 5)
            f.write(
                f"Modelo {plane:<8}: MAE={mae:.2f} | MSE={mse:.2f} | Pearson={corr:.3f} | "
                f"Sujetos con error < 5 años: {count_within_5}/{total} ({100*count_within_5/total:.1f}%)\n"
            )
        count_within_5_e = np.sum(np.abs(ensemble_preds - targets) < 5)
        f.write(
            f"\nEnsemble  : MAE={mae_e:.2f} | MSE={mse_e:.2f} | Pearson={corr_e:.3f} | "
            f"Sujetos con error < 5 años: {count_within_5_e}/{total} ({100*count_within_5_e/total:.1f}%)\n"
        )

    print(f"\nResultados guardados en: {OUTPUT_PATH}", flush=True)

    # Mostrar solo el bloque recién agregado
    with open(OUTPUT_PATH, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i in reversed(range(len(lines))):
        if lines[i].strip() == f"==================== {timestamp} ====================":
            start_idx = i
            break

    if start_idx is not None:
        print("\n" + "".join(lines[start_idx:]).strip())

if __name__ == "__main__":
    main()
