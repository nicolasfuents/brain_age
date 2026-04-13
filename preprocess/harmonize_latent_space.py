#!/usr/bin/env python3

# Modo de uso:
# "$CONDA_PREFIX/bin/python" -u harmonize_latent_space.py

import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
PROJECT_ROOT = Path("/home/nfuentes/scratch/brain_age_project/openBHB_dataset")

# Origen de los datos crudos/desbalanceados
DATA_SOURCES = [
    PROJECT_ROOT / "data" / "DB_Lautaro_quasiraw" / "processed_p99",
    PROJECT_ROOT / "data" / "DB_Lautaro_quasiraw" / "processed_fomo_p99",
    PROJECT_ROOT / "data" / "OASIS3" / "MR_867_HC" / "processed",
    PROJECT_ROOT / "data" / "DB_INTECNUS_BRAVO" / "processed"
]

# Destino de los tensores armonizados
OUT_DIR = PROJECT_ROOT / "data" / "HARMONIZED_OOD_P99"
PLANES = ["axial", "coronal", "sagittal"]
N_JOBS = 96 

MNI_MASK_PATH = PROJECT_ROOT / "data" / "atlases" / "MNI152_T1_1mm_brain_mask_SOLID_v2.nii.gz"

# ==============================================================================
# PREPARACIÓN DE LAS MÁSCARAS 2.5D Y DIMENSIONES TARGET
# ==============================================================================
def get_2p5d_masks(mask_path, num_slices=5):
    """Extrae los slices centrales de la máscara 3D y define los Targets Topológicos"""
    img = nib.load(mask_path)
    vol = img.get_fdata() > 0 
    
    masks_2p5d = {}
    for axis, plane in zip([2, 1, 0], ["axial", "coronal", "sagittal"]):
        c = vol.shape[axis] // 2
        half = num_slices // 2
        idxs = [c + o for o in range(-half, half + 1)]
        
        if axis == 0: slices = [vol[i, :, :] for i in idxs]
        elif axis == 1: slices = [vol[:, i, :] for i in idxs]
        else: slices = [vol[:, :, i] for i in idxs]
            
        mask_tensor = torch.from_numpy(np.stack(slices, 0).astype(bool))
        masks_2p5d[plane] = mask_tensor
        
    return masks_2p5d

# ==============================================================================
# WORKER DE ARMONIZACIÓN
# ==============================================================================
def harmonize_tensor(file_path, plane, global_masks):
    try:
        db_name = "UNKNOWN"
        for ds in DATA_SOURCES:
            if str(ds) in str(file_path):
                db_name = ds.parts[-2] if ds.name == "processed" else ds.name
                break
                
        out_path = OUT_DIR / db_name / plane / file_path.name
        if out_path.exists():
            return "[SKIP] Ya armonizado"
            
        out_path.parent.mkdir(parents=True, exist_ok=True)

        obj = torch.load(file_path, map_location='cpu', weights_only=False)
        tensor = obj["image"].float() if isinstance(obj, dict) and "image" in obj else obj.float()

        mask_2p5d = global_masks[plane]
        target_h, target_w = mask_2p5d.shape[1], mask_2p5d.shape[2]
        _, curr_h, curr_w = tensor.shape

        # 1. Estandarización Topológica Dinámica (Cero-Padding y Center Crop)
        # Se ajusta el tensor a las dimensiones exactas de la máscara MNI de su plano
        pad_h = max(0, target_h - curr_h)
        pad_w = max(0, target_w - curr_w)

        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - (pad_w // 2), pad_h // 2, pad_h - (pad_h // 2))
            tensor = F.pad(tensor, padding, mode='constant', value=0.0)

        tensor = TF.center_crop(tensor, [target_h, target_w])
        tensor = tensor.clone() # Aislar memoria

        # 2. Enmascaramiento Espacial Invariante (Domain Alignment)
        # Ahora las dimensiones coinciden perfectamente
        tensor[~mask_2p5d] = 0.0

        # 3. Estandarización Robusta Intracraneana (Contraste Invariante)
        mask_pos = tensor > 1e-4
        if mask_pos.any():
            valid_voxels = tensor[mask_pos]
            p01 = torch.quantile(valid_voxels, 0.01).item()
            p99 = torch.quantile(valid_voxels, 0.99).item()
            rango = p99 - p01 if (p99 - p01) > 0 else 1.0
            
            tensor[mask_pos] = torch.clamp((tensor[mask_pos] - p01) / rango, 0.0, 1.0)
            
        tensor[~mask_pos] = 0.0

        # 4. Guardado
        if isinstance(obj, dict):
            obj["image"] = tensor
            obj["meta"]["harmonization"] = "MNI152_masked_robust_P01_P99_TopologicalStd"
            torch.save(obj, out_path)
        else:
            torch.save(tensor, out_path)

        return "[OK] Harmonizado"
        
    except Exception as e:
        return f"[ERROR] {file_path.name}: {e}"

# ==============================================================================
# SANITY CHECK VISUAL (MONTAGE)
# ==============================================================================
def generate_sanity_montage(data_sources, global_masks):
    print("\n[*] Generando Montage de Estandarización Topológica...")
    mask_ax = global_masks["axial"]
    target_h, target_w = mask_ax.shape[1], mask_ax.shape[2]
    
    valid_sources = [ds for ds in data_sources if ds.exists()]
    if not valid_sources:
        print("[!] No hay bases de datos disponibles para el montage.")
        return

    fig, axes = plt.subplots(1, len(valid_sources), figsize=(5 * len(valid_sources), 5))
    if len(valid_sources) == 1: axes = [axes]
    
    for ax, ds in zip(axes, valid_sources):
        # Buscamos el primer archivo axial disponible
        pt_files = list(ds.rglob("axial/*.pt"))
        if not pt_files:
            # Fallback si no hay subcarpeta axial explícita pero los archivos lo son
            pt_files = list(ds.rglob("*.pt"))
            
        if not pt_files:
            ax.set_title(f"{ds.name}\nSin datos", color="red")
            ax.axis("off")
            continue
            
        file_path = pt_files[0]
        obj = torch.load(file_path, map_location='cpu', weights_only=False)
        tensor = obj["image"].float() if isinstance(obj, dict) and "image" in obj else obj.float()
        
        _, curr_h, curr_w = tensor.shape
        pad_h = max(0, target_h - curr_h)
        pad_w = max(0, target_w - curr_w)
        
        # TRUCO VISUAL: Paddeamos con gris (0.3) para que el borde inyectado sea visible
        tensor_vis = tensor.clone()
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - (pad_w // 2), pad_h // 2, pad_h - (pad_h // 2))
            tensor_vis = F.pad(tensor_vis, padding, mode='constant', value=0.3)
            
        tensor_vis = TF.center_crop(tensor_vis, [target_h, target_w])
        
        # Extraemos el slice central para visualizar
        slice_img = tensor_vis[2].numpy()
        
        ax.imshow(slice_img, cmap="bone")
        ax.set_title(f"{ds.name}\nOrig: {curr_h}x{curr_w} -> Target: {target_h}x{target_w}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_img = PROJECT_ROOT / "padding_sanity_check.png"
    plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor='black')
    print(f"[*] Montage guardado exitosamente en: {out_img}")

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    print("--- INICIANDO CIRUGÍA MATRICIAL DE ARMONIZACIÓN (DOMAIN ALIGNMENT) ---")
    if not MNI_MASK_PATH.exists():
        sys.exit(f"Falta máscara MNI152 en: {MNI_MASK_PATH}")
        
    global_masks = get_2p5d_masks(MNI_MASK_PATH)
    print("[*] Máscaras MNI152 2.5D y dimensiones Target cargadas en memoria.")
    
    tasks = []
    for source in DATA_SOURCES:
        if not source.exists(): 
            print(f"[*] Omitiendo origen no disponible: {source.name}")
            continue
        for f in source.rglob("*.pt"):
            plane = f.parent.name.lower() if f.parent.name.lower() in PLANES else f.stem.lower()
            if plane in PLANES:
                tasks.append((f, plane))

    print(f"[*] Detectados {len(tasks)} tensores para armonizar.")
    print(f"[*] Procesando en {N_JOBS} workers paralelos...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(harmonize_tensor)(f, p, global_masks) for f, p in tqdm(tasks)
    )
    
    ok = sum(1 for r in results if "[OK]" in r)
    err = sum(1 for r in results if "[ERROR]" in r)
    skip = sum(1 for r in results if "[SKIP]" in r)
    
    print("\n--- REPORTE FINAL ---")
    print(f"Tensores armonizados: {ok}")
    print(f"Omitidos (ya existían): {skip}")
    print(f"Errores I/O: {err}")
    print(f"Ruta unificada de destino: {OUT_DIR}")
    
    # Disparamos el montage de sanity check
    generate_sanity_montage(DATA_SOURCES, global_masks)
    print("¡Base de datos lista para entrenar sin Domain Shift!")

if __name__ == "__main__":
    main()