#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Auditoría de tensores P99")
parser.add_argument("--dir", type=str, required=True, help="Ruta al directorio que contiene los planos (axial, coronal, sagittal)")
parser.add_argument("--csv", type=str, default="tensor_audit.csv", help="Nombre del archivo CSV de salida")
args = parser.parse_args()

BASE_DIR = Path(args.dir)
PLANES = ["axial", "coronal", "sagittal"]
OUTPUT_CSV = Path(args.csv)
N_JOBS = 48

def audit_tensor(file_path, plane):
    # Esquema estricto para evitar colapso del DataFrame
    result_template = {
        "Filename": file_path.name, "Plane": plane, "Shape": None, "Dtype": None,
        "Min": None, "Max": None, "Mean": None, "Std": None,
        "Sparsity": None, "NaNs": None, "Infs": None, "Status": "UNPROCESSED"
    }
    
    try:
        # Relajamos weights_only temporalmente por si el .pt es un diccionario anidado
        obj = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Heurística de desempaquetado si el objeto es un diccionario (ej. dict de MONAI)
        if isinstance(obj, dict):
            # Buscar la clave heurística que contenga el tensor volumétrico
            tensor = None
            for key in ["image", "img", "data", "tensor"]:
                if key in obj and isinstance(obj[key], torch.Tensor):
                    tensor = obj[key]
                    break
            if tensor is None:
                result_template["Status"] = "DICT_NO_TENSOR_FOUND"
                return result_template
        elif isinstance(obj, torch.Tensor):
            tensor = obj
        else:
            result_template["Status"] = f"INVALID_TYPE_{type(obj).__name__}"
            return result_template

        t_float = tensor.float()
        
        # Extracción de métricas
        result_template["Shape"] = str(tuple(tensor.shape))
        result_template["Dtype"] = str(tensor.dtype)
        result_template["Min"] = t_float.min().item()
        result_template["Max"] = t_float.max().item()
        result_template["Mean"] = t_float.mean().item()
        result_template["Std"] = t_float.std().item()
        result_template["NaNs"] = torch.isnan(t_float).sum().item()
        result_template["Infs"] = torch.isinf(t_float).sum().item()
        
        total_voxels = t_float.numel()
        zero_voxels = (t_float == 0.0).sum().item()
        result_template["Sparsity"] = zero_voxels / total_voxels if total_voxels > 0 else 0.0
        result_template["Status"] = "OK"
        
        return result_template
        
    except Exception as e:
        result_template["Status"] = f"ERROR: {str(e)}"
        return result_template

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {BASE_DIR}")

    # Relajamos la heurística para buscar en la jerarquía generada: Sujeto/t1_tensors/plano/*.pt
    tasks = []
    for plane in PLANES:
        # Busca recursivamente cualquier .pt que esté dentro de una carpeta con el nombre del plano
        found_files = list(BASE_DIR.rglob(f"*/{plane}/*.pt"))
        for f in found_files:
            tasks.append((f, plane))
                
    if not tasks:
        print("No se detectaron tensores en el árbol de directorios.")
        return

    print(f"Auditando {len(tasks)} tensores con esquema estricto...")
    
    results = Parallel(n_jobs=N_JOBS)(delayed(audit_tensor)(f, p) for f, p in tqdm(tasks))
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- REPORTE DE INTEGRIDAD TENSORIAL ---")
    
    fallos = df[df["Status"] != "OK"]
    if not fallos.empty:
        print(f"Fallos detectados: {len(fallos)} tensores.")
        # Muestra la distribución de errores para identificar el patrón de falla de la lectura
        print(fallos["Status"].value_counts())
        
    df_ok = df[df["Status"] == "OK"]
    if df_ok.empty:
        print("\nCRÍTICO: Ningún tensor pudo ser validado numéricamente. Revisar la traza de Status superior.")
        return

    # Análisis de inestabilidad
    corruptos = df_ok[(df_ok["NaNs"] > 0) | (df_ok["Infs"] > 0)]
    if not corruptos.empty:
        print(f"¡ADVERTENCIA! {len(corruptos)} tensores presentan inestabilidad numérica (NaNs/Infs).")
    else:
        print("Estabilidad numérica: OK (0 NaNs, 0 Infs en los tensores validados).")

    print("\n--- ISOMORFISMO DIMENSIONAL POR PLANO ---")
    for plane in PLANES:
        df_plane = df_ok[df_ok["Plane"] == plane]
        if not df_plane.empty:
            print(f"[{plane.upper()}] Shapes:")
            print(df_plane["Shape"].value_counts().to_string())

    print("\n--- MOMENTOS ESTADÍSTICOS GLOBALES (P99 NORMALIZATION) ---")
    print(df_ok[["Min", "Max", "Mean", "Sparsity"]].agg(['mean', 'std', 'min', 'max']))

if __name__ == "__main__":
    main()
