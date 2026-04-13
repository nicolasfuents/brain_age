#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_and_harmonize.py

Pipeline unificado de preprocesamiento y armonización de dominio para T1 estructural.
Este script implementa, en un único flujo end-to-end, los siguientes pasos:

1. brainprep:
   - reorientación estándar
   - registro afín/no-lineal a MNI152 1 mm isotrópico
   - corrección de bias field (N4)

2. Armonización espacial:
   - aplicación de la máscara intracraneana fija SOLID_v2 en espacio MNI
   - eliminación del ruido extra-axial
   - homogenización topológica entre dominios

3. Armonización de contraste:
   - cálculo robusto de percentiles P1–P99 dentro de la máscara
   - clip y reescalado de intensidades a [0, 1]

4. Extracción 2.5D y serialización:
   - generación de tensores centrales de 5 slices por plano
   - exportación a .pt
   - exportación opcional de NIfTI para inspección visual

==============================================================================
USO
==============================================================================

Ejemplo básico:
    "$CONDA_PREFIX/bin/python" -u preprocess_and_harmonize.py

Ejemplo indicando subdirectorio BIDS específico:
    "$CONDA_PREFIX/bin/python" -u preprocess_and_harmonize.py --filter anat

Ejemplo cambiando carpeta de salida:
    "$CONDA_PREFIX/bin/python" -u preprocess_and_harmonize.py --out /ruta/a/processed

Ejemplo cambiando paralelismo:
    "$CONDA_PREFIX/bin/python" -u preprocess_and_harmonize.py --n_jobs 32

Ejemplo para bases de datos con nombres de archivo no estandarizados:
    "$CONDA_PREFIX/bin/python" -u preprocess_and_harmonize.py --wildcard
    
==============================================================================
REQUISITOS DE ENTORNO
==============================================================================

Este script asume que el entorno ya tiene disponibles:

- Python con:
    nibabel
    numpy
    pandas
    torch
    tqdm
    joblib

- FSL correctamente cargado en el entorno
- FreeSurfer correctamente configurado
- brainprep.sh disponible en:
    /home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/preprocess/brainprep.sh

Variables de entorno requeridas:
- FSLDIR
- FREESURFER_HOME
- FS_LICENSE
- CONDA_PREFIX (recomendado)

Además, deben existir en el sistema:
- fslreorient2std
- fslorient
- fslcc

==============================================================================
ESTRUCTURA ESPERADA DE UNA NUEVA BASE DE DATOS
==============================================================================

Para preprocesar otra base de datos, la organización mínima esperada es:

BASE_DATASET/
├── participants.csv
├── SUBJ_001/
│   └── ... archivos NIfTI T1w ...
├── SUBJ_002/
│   └── ... archivos NIfTI T1w ...
└── SUBJ_003/
    └── ... archivos NIfTI T1w ...

Es decir:

1. Debe existir una carpeta raíz de entrada (IN_DIR).
2. Dentro de esa carpeta, cada sujeto debe tener su propia subcarpeta.
3. El nombre de cada subcarpeta debe coincidir con la columna Subject_ID del CSV.
4. Dentro de cada carpeta de sujeto debe existir al menos un archivo T1w:
       *T1w*.nii
    o
       *T1w*.nii.gz

El script busca recursivamente esos archivos dentro de cada sujeto.

==============================================================================
ESTRUCTURA DEL CSV DE METADATOS
==============================================================================

El archivo participants.csv (o equivalente) debe contener, como mínimo, estas columnas:

- Subject_ID
- Age

Opcionalmente puede incluir:
- Sex

Importante:
- `Subject_ID` y `Age` son obligatorias.
- `Sex` no es obligatoria, pero si se incluye debe llamarse exactamente `Sex`.
- La posición de las columnas en el CSV no importa.
- Se recomienda codificar `Sex` de forma consistente como `F` o `M`.
- Si `Sex` no existe, está vacía o es NaN, el script asigna automáticamente `Unknown`.

Ejemplo:

Subject_ID,Age,Sex,Group
SUBJ_001,72,F,CN
SUBJ_002,68,M,MCI
SUBJ_003,80,F,AD

Importante:
- Subject_ID debe coincidir exactamente con el nombre de la carpeta del sujeto.
- Age debe ser numérico y convertible a float.

Si tu base usa otros nombres de columna (por ejemplo ID, subject, participant_id),
deberás adaptar manualmente estas líneas del script:

    subject_id = str(row["Subject_ID"]).strip()
    
Con la misma idea, hacer las modificaciones pertinentes para el caso de que Age y Sex estén nombrados de distinta manera (incluyendo minus/mayus).    

==============================================================================
BÚSQUEDA DE T1w
==============================================================================

El script busca archivos T1w así:

1. Intento Estándar:
   Busca recursivamente patrones de nombre estándar (*T1w*.nii, *T1w*.nii.gz).
    - si NO se usa --filter:
        busca recursivamente dentro de toda la carpeta del sujeto:
            *T1w*.nii
            *T1w*.nii.gz

    - si se usa --filter <subdir>:
        restringe la búsqueda a:
            IN_DIR / Subject_ID / <subdir> / ...

        Ejemplo:
            --filter anat
        obliga a buscar T1w dentro de:
            SUBJ_001/anat/
            SUBJ_002/anat/
            etc.

        Esto es útil para datasets en formato BIDS o semiestructurados.

2. Modo Robusto (Flag --wildcard):
   Si no encuentra ningún archivo con el patrón "T1w", y el flag está activo,
   el script tomará el primer archivo .nii o .nii.gz que encuentre.
   Esto es vital para bases de datos no-BIDS con nomenclaturas ad-hoc 
   (ej: "SagIR-FSPGR450.nii.gz").

Importante: El script siempre prioriza el patrón "T1w" para evitar procesar 
volúmenes T2 o FLAIR que pudieran coexistir en la misma carpeta.

==============================================================================
CASO LONGITUDINAL
==============================================================================

Si un sujeto tiene múltiples T1w válidas, el script aplica una heurística simple:

- busca patrones tipo:
      sess-dXXXX
- selecciona la sesión con menor número de días
- la interpreta como baseline

Ejemplo:
    sess-d0001  -> preferida frente a sess-d0365

Si tus datasets longitudinales usan otra convención, deberás adaptar la función
de selección del archivo.

==============================================================================
SALIDA GENERADA
==============================================================================

Para cada sujeto procesado exitosamente, la salida queda organizada así:

OUT_DIR/
└── SUBJ_001/
    ├── t1_tensors/
    │   ├── SUBJ_001_full_preprocessed.nii.gz
    │   ├── axial/
    │   │   ├── SUBJ_001.pt
    │   │   └── SUBJ_001_tensor.nii.gz
    │   ├── coronal/
    │   │   ├── SUBJ_001.pt
    │   │   └── SUBJ_001_tensor.nii.gz
    │   └── sagittal/
    │       ├── SUBJ_001.pt
    │       └── SUBJ_001_tensor.nii.gz
    └── ...

Contenido de cada archivo .pt:
    {
        "image": Tensor[5, H, W],
        "age": torch.float32,
        "meta": {
            "source": nombre_archivo_original,
            "csv_id": Subject_ID,
            "p01": valor_percentil_1,
            "p99": valor_percentil_99,
            "fslcc": correlación_con_MNI,
            "harmonization": descripcion_del_pipeline
        }
    }

==============================================================================
MÁSCARA REQUERIDA
==============================================================================

La máscara intracraneana fija debe existir en:

    PROJECT_ROOT / "data" / "atlases" / "MNI152_T1_1mm_brain_mask_SOLID_v2.nii.gz"

Esta máscara debe:

- estar en espacio MNI152 1 mm isotrópico
- tener la misma forma que la salida de brainprep
- representar la región intracraneana sólida usada para armonización espacial

Si cambiás de template o resolución, deberás usar una máscara consistente con ese
nuevo espacio.

==============================================================================
SUPUESTOS IMPORTANTES PARA REUTILIZAR EL SCRIPT EN OTRA BASE
==============================================================================

Para reutilizar este pipeline en otra cohorte, deben cumplirse estas condiciones:

1. La imagen T1 puede ser llevada correctamente a MNI152 por brainprep.
2. La salida final de brainprep tiene la misma geometría que la máscara SOLID_v2.
3. Cada sujeto tiene una T1w identificable por nombre.
4. El CSV de metadatos permite obtener:
   - identificador del sujeto
   - edad cronológica
5. La resolución objetivo sigue siendo MNI152 1 mm isotrópico.
6. La estrategia 2.5D sigue siendo central:
   - 5 slices axiales
   - 5 slices coronales
   - 5 slices sagitales

Si cualquiera de estos supuestos cambia, el script debe ajustarse.

==============================================================================
QUÉ HAY QUE MODIFICAR SI CAMBIÁS DE BASE DE DATOS
==============================================================================

Al adaptar el script a otra base, revisá al menos estas variables:

- IN_DIR
- OUT_DIR
- CSV_METADATA
- cómo se leen Subject_ID y Age desde el CSV
- patrón de búsqueda de T1w
- posible uso de --filter
- heurística longitudinal si la nomenclatura de sesiones cambia

En la mayoría de los casos, no hace falta tocar el pipeline de armonización en sí;
solo la parte de entrada/metadatos.

==============================================================================
CONTROL DE CALIDAD
==============================================================================

El script calcula un QC automático usando fslcc contra el template MNI:

- si fslcc <= 0.75
    el sujeto se descarta con estado [SKIP]

Esto ayuda a eliminar registros fallidos o alineaciones patológicas.
El umbral puede cambiarse si la nueva cohorte lo requiere.

==============================================================================
RESUMEN CONCEPTUAL
==============================================================================

Este script está pensado para producir una representación de entrada invariante
entre dominios, asegurando:

- misma geometría global (MNI152)
- misma topología intracraneana (SOLID_v2)
- misma normalización robusta de contraste (P1–P99)
- misma extracción 2.5D
- mismo formato de serialización final (.pt)

La intención es minimizar domain shift entre cohortes antes del entrenamiento
o la inferencia del modelo de brain age.
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import shutil
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import re

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACION
# ==============================================================================
PROJECT_ROOT = Path("/home/nfuentes/scratch/brain_age_project/openBHB_dataset")
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "preprocess"
BASH_SCRIPT_T1 = SCRIPTS_DIR / "brainprep.sh"

IN_DIR = PROJECT_ROOT / "data" / "DB_INTECNUS_BRAVO"
OUT_DIR = IN_DIR / "processed"
CSV_METADATA = IN_DIR / "participants.csv"

CMD_FSLREORIENT = shutil.which("fslreorient2std")
CMD_FSLORIENT   = shutil.which("fslorient")

FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:
    sys.exit("[ERROR] FSLDIR no definido. Cargá el módulo FSL.")

MNI_TEMPLATE = Path(FSLDIR) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"

# Máscara sólida para armonización espacial invariante
MNI_MASK_PATH = PROJECT_ROOT / "data" / "atlases" / "MNI152_T1_1mm_brain_mask_SOLID_v2.nii.gz"

# Paralelismo
N_JOBS = 64
N_GPUS = 4

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def repair_header_and_reorient(raw_path, output_std_path, work_dir):
    try:
        subprocess.run(
            [CMD_FSLREORIENT, str(raw_path), str(output_std_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8")
        if "Orientation information not stored" in err_msg or "non-zero code" in err_msg:
            temp_nii = work_dir / f"temp_fix_{os.getpid()}.nii.gz"
            shutil.copy(raw_path, temp_nii)
            try:
                subprocess.run([CMD_FSLORIENT, "-setqformcode", "1", str(temp_nii)], check=True)
                subprocess.run([CMD_FSLORIENT, "-setsformcode", "1", str(temp_nii)], check=True)
                subprocess.run(
                    [CMD_FSLREORIENT, str(temp_nii), str(output_std_path)],
                    check=True,
                    stdout=subprocess.DEVNULL
                )
                if temp_nii.exists():
                    os.remove(temp_nii)
                return True
            except Exception:
                if temp_nii.exists():
                    os.remove(temp_nii)
                return False
        return False


def stack_central_slices(vol, axis, num=5):
    assert vol.ndim == 3
    c = vol.shape[axis] // 2
    half = num // 2
    idxs = [c + o for o in range(-half, half + 1)]
    raw_slices = []

    for i in idxs:
        if axis == 0:
            sl = vol[i, :, :]
        elif axis == 1:
            sl = vol[:, i, :]
        else:
            sl = vol[:, :, i]
        raw_slices.append(sl.astype(np.float32, copy=False))

    return np.stack(raw_slices, 0)


def load_solid_mask(mask_path):
    if not mask_path.exists():
        sys.exit(f"[ERROR] No existe la máscara SOLID_v2: {mask_path}")
    img = nib.load(str(mask_path))
    mask = img.get_fdata() > 0
    return mask.astype(bool), img.affine


def robust_normalize_p01_p99(data, mask):
    """
    Normalización robusta dentro de la máscara:
    - calcula P1 y P99 sobre voxels intracraneanos
    - clipea a [P1, P99]
    - escala a [0, 1]
    - fuerza fondo a cero
    """
    valid = mask & np.isfinite(data)

    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32), 0.0, 1.0

    vox = data[valid].astype(np.float32)

    p01 = np.percentile(vox, 1)
    p99 = np.percentile(vox, 99)

    if p99 <= p01:
        p99 = p01 + 1.0

    out = np.zeros_like(data, dtype=np.float32)
    clipped = np.clip(data[valid], p01, p99)
    out[valid] = (clipped - p01) / (p99 - p01)

    out[~mask] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return out, float(p01), float(p99)


def select_t1_file(raw_path_dir, dir_filter=None, wildcard=False):
    target_dir = raw_path_dir / dir_filter if dir_filter else raw_path_dir
    
    if dir_filter and not target_dir.exists():
        return None, f"Directorio filtro '{dir_filter}' no existe"

    # 1. Intento estándar: buscar explícitamente T1w
    nifti_files = list(target_dir.rglob("*T1w*.nii")) + list(target_dir.rglob("*T1w*.nii.gz"))

    # 2. Intento robusto: si no hay T1w y wildcard está activo, buscamos cualquier NIfTI
    if len(nifti_files) == 0 and wildcard:
        nifti_files = list(target_dir.rglob("*.nii")) + list(target_dir.rglob("*.nii.gz"))
        if len(nifti_files) > 0:
            print(f"[*] Wildcard activado para {raw_path_dir.name}: Usando {nifti_files[0].name}")

    if len(nifti_files) == 0:
        return None, "Imagen T1w (o NIfTI genérico) no encontrada"

    if len(nifti_files) > 1:
        def extract_days(filename):
            match = re.search(r"sess-d(\d+)", str(filename))
            return int(match.group(1)) if match else float("inf")

        nifti_files.sort(key=lambda x: extract_days(x.name))
        return nifti_files[0], f"Longitudinal: n={len(nifti_files)}. Seleccionada baseline {nifti_files[0].name}"

    return nifti_files[0], ""


def prepare_env_for_worker(gpu_id, prep_dir):
    env_gpu = os.environ.copy()
    env_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_gpu["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    env_gpu["OMP_NUM_THREADS"] = "1"

    conda_bin = os.path.join(env_gpu.get("CONDA_PREFIX", "/home/nfuentes/miniforge3/envs/brain_age_env"), "bin")
    fs_bin = os.path.join(env_gpu.get("FREESURFER_HOME", ""), "bin")
    fsl_bin = os.path.join(env_gpu.get("FSLDIR", ""), "bin")

    env_gpu["PATH"] = f"{conda_bin}:{fs_bin}:{fsl_bin}:" + env_gpu.get("PATH", "")

    if "FS_LICENSE" not in env_gpu:
        env_gpu["FS_LICENSE"] = os.path.join(os.environ.get("HOME", ""), ".licenses", "freesurfer.lic")

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env_gpu["PATH"] = f"{conda_prefix}/bin:" + env_gpu.get("PATH", "")

    # Dummy dpkg para brainprep
    dummy_dpkg = prep_dir / "dpkg"
    with open(dummy_dpkg, "w") as f:
        f.write("#!/bin/bash\necho ''\n")
    dummy_dpkg.chmod(0o755)
    env_gpu["PATH"] = f"{prep_dir}:" + env_gpu.get("PATH", "")

    return env_gpu


def run_brainprep(t1_std, prep_dir, env_gpu):
    # Parche bug ANTs N4BiasFieldCorrection en brainprep
    ghost_dir = os.path.join(os.getcwd(), " " + str(prep_dir / "quasiraw"))
    os.makedirs(ghost_dir, exist_ok=True)

    bash_process = subprocess.run(
        ["bash", str(BASH_SCRIPT_T1), str(t1_std), str(prep_dir)],
        env=env_gpu,
        capture_output=True,
        text=True
    )

    if bash_process.returncode != 0:
        error_msg = bash_process.stderr.strip().replace("\n", " | ")
        return False, f"Bash falló (Código {bash_process.returncode}). Traza: {error_msg}"

    t1_mni = prep_dir / "quasiraw" / "t1_std_orient_desc-6apply_T1w.nii.gz"
    if not t1_mni.exists():
        return False, "Fallo el pipeline de brainprep. Archivo MNI no generado."

    return True, t1_mni


def compute_fslcc(t1_mni):
    qc_res = subprocess.run(
        ["fslcc", str(t1_mni), str(MNI_TEMPLATE)],
        capture_output=True,
        text=True
    )

    cc_val = 0.0
    if qc_res.stdout.strip():
        cc_val = float(qc_res.stdout.strip().split("\n")[0].split()[-1])

    return cc_val


# ==============================================================================
# WORKER PRINCIPAL UNIFICADO
# ==============================================================================
def process_subject(row, idx, output_root, dir_filter, solid_mask, wildcard):
    gpu_id = idx % N_GPUS

    subject_id = str(row["Subject_ID"]).strip()
    age = float(row["Age"])

    if "Sex" in row and pd.notna(row["Sex"]):
        sex_raw = str(row["Sex"]).strip().lower()
        if sex_raw in ["f", "female", "femenino", "mujer"]:
            sex = "F"
        elif sex_raw in ["m", "male", "masculino", "varon", "varón", "hombre"]:
            sex = "M"
        else:
            sex = "Unknown"
    else:
        sex = "Unknown"

    subj_dir = output_root / subject_id
    prep_dir = subj_dir / "preprocess"
    t1_tensor_dir = subj_dir / "t1_tensors"

    if prep_dir.exists():
        shutil.rmtree(prep_dir, ignore_errors=True)
    if t1_tensor_dir.exists():
        shutil.rmtree(t1_tensor_dir, ignore_errors=True)

    prep_dir.mkdir(parents=True, exist_ok=True)
    t1_tensor_dir.mkdir(parents=True, exist_ok=True)
    for p in ["axial", "coronal", "sagittal"]:
        (t1_tensor_dir / p).mkdir(exist_ok=True)

    # Definición de ruta con fallback recursivo para estructuras tipo SC/CC
    raw_path_dir = IN_DIR / subject_id

    if not raw_path_dir.exists():
        # Búsqueda recursiva agnóstica a la profundidad
        matches = list(IN_DIR.glob(f"**/{subject_id}"))
        # Extraemos el primer match que sea efectivamente un directorio
        raw_path_dir = next((m for m in matches if m.is_dir()), None)
        
        if raw_path_dir is None:
            return f"[ERROR] {subject_id}: No se encontró la carpeta en {IN_DIR}"

    # Selección de archivo respetando tu firma con dir_filter y wildcard
    raw_path, warning_msg = select_t1_file(raw_path_dir, dir_filter, wildcard)

    if raw_path is None:
        return f"[ERROR] {subject_id}: {warning_msg}"

    try:
        env_gpu = prepare_env_for_worker(gpu_id, prep_dir)

        # 1. Reorientación estándar
        t1_std = prep_dir / "t1_std_orient.nii.gz"
        if not repair_header_and_reorient(raw_path, t1_std, prep_dir):
            return f"[ERROR] {subject_id}: Fallo reorientación"

        # 2. brainprep: registro MNI + N4
        ok, result = run_brainprep(t1_std, prep_dir, env_gpu)
        if not ok:
            return f"[ERROR] {subject_id}: {result}"

        t1_mni = result

        # 3. QC automático
        cc_val = compute_fslcc(t1_mni)
        if cc_val <= 0.75:
            return f"[SKIP] {subject_id} - QC falló (fslcc={cc_val})"

        # 4. Carga volumen registrado a MNI
        img = nib.load(str(t1_mni))
        data = np.squeeze(img.get_fdata(dtype=np.float32))

        if data.shape != solid_mask.shape:
            return f"[ERROR] {subject_id}: Shape incompatible con máscara SOLID_v2. Vol={data.shape}, Mask={solid_mask.shape}"

        # 5. Enmascaramiento espacial invariante
        data_masked = np.zeros_like(data, dtype=np.float32)
        data_masked[solid_mask] = data[solid_mask]

        # 6. Normalización robusta P1-P99 dentro de la máscara
        data_norm, p01, p99 = robust_normalize_p01_p99(data_masked, solid_mask)

        # 7. Extracción 2.5D
        raw_sag = stack_central_slices(data_norm, 0)
        raw_cor = stack_central_slices(data_norm, 1)
        raw_ax  = stack_central_slices(data_norm, 2)

        # 7.b Estandarización Topológica Dinámica (Padding/Crop)
        # Extraemos los Targets esperados directamente de las dimensiones de la máscara SOLID_v2 (MNI152)
        # solid_mask shape es típicamente (X, Y, Z) ej: (182, 218, 182)
        dim_x, dim_y, dim_z = solid_mask.shape
        targets = {
            "sagittal": (dim_y, dim_z), # Plano YZ
            "coronal":  (dim_x, dim_z), # Plano XZ
            "axial":    (dim_x, dim_y)  # Plano XY
        }

        def apply_topo_std(tensor_2p5d, target_h, target_w):
            d, h, w = tensor_2p5d.shape
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            # 1. Zero-Padding simétrico (usando numpy.pad)
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                # Pad solo en H y W, no en la dimensión Depth (los 5 cortes)
                tensor_2p5d = np.pad(tensor_2p5d, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # 2. Center-Crop
            d, h, w = tensor_2p5d.shape
            crop_top = (h - target_h) // 2
            crop_left = (w - target_w) // 2
            return tensor_2p5d[:, crop_top:crop_top+target_h, crop_left:crop_left+target_w]

        ten_sag = apply_topo_std(raw_sag, *targets["sagittal"])
        ten_cor = apply_topo_std(raw_cor, *targets["coronal"])
        ten_ax  = apply_topo_std(raw_ax, *targets["axial"])

        # 8. Export NIfTI para inspección
        nib.save(
            nib.Nifti1Image(data_norm, img.affine),
            t1_tensor_dir / f"{subject_id}_full_preprocessed.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(ten_ax.transpose(1, 2, 0), np.eye(4)),
            t1_tensor_dir / "axial" / f"{subject_id}_tensor.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(ten_cor.transpose(1, 2, 0), np.eye(4)),
            t1_tensor_dir / "coronal" / f"{subject_id}_tensor.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(ten_sag.transpose(1, 2, 0), np.eye(4)),
            t1_tensor_dir / "sagittal" / f"{subject_id}_tensor.nii.gz"
        )

        # 9. Guardado .pt
        meta = {
            "source": raw_path.name,
            "csv_id": subject_id,
            "p01": float(p01),
            "p99": float(p99),
            "fslcc": float(cc_val),
            "harmonization": "brainprep_MNI152_N4 + SOLID_v2_mask + robust_P01_P99 + TopoStd",
        }
        age_t = torch.tensor(age, dtype=torch.float32)

        torch.save(
            {"image": torch.from_numpy(ten_ax), "age": age_t, "sex": sex, "meta": meta},
            t1_tensor_dir / "axial" / f"{subject_id}.pt"
        )
        torch.save(
            {"image": torch.from_numpy(ten_cor), "age": age_t, "sex": sex, "meta": meta},
            t1_tensor_dir / "coronal" / f"{subject_id}.pt"
        )
        torch.save(
            {"image": torch.from_numpy(ten_sag), "age": age_t, "sex": sex, "meta": meta},
            t1_tensor_dir / "sagittal" / f"{subject_id}.pt"
        )

        # Limpieza para ahorrar inodos
        shutil.rmtree(prep_dir, ignore_errors=True)

        suffix = f" ({warning_msg})" if warning_msg else ""
        return f"[OK] {subject_id} - GPU: {gpu_id} (fslcc={cc_val:.4f}){suffix}"

    except Exception as e:
        return f"[ERROR] {subject_id}: {e}"


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--filter", type=str, default=None,
                        help="Subdirectorio BIDS específico a buscar (ej. anat2)")
    parser.add_argument("--wildcard", action="store_true",
                        help="Si no encuentra *T1w*, toma el primer .nii/.nii.gz que encuentre.")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    print("--- PROCESAMIENTO END-TO-END UNIFICADO CON ARMONIZACIÓN DE DOMINIO ---")
    print("1) brainprep: MNI152 1mm + N4")
    print("2) máscara SOLID_v2")
    print("3) normalización robusta P1-P99 intra-máscara")
    print("4) extracción 2.5D + serialización .pt")

    df = pd.read_csv(CSV_METADATA)
    solid_mask, _ = load_solid_mask(MNI_MASK_PATH)

    print(f"Procesando {len(df)} sujetos. Lanzando {args.n_jobs} workers en paralelo sobre {N_GPUS} GPUs...")

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_subject)(row, idx, output_root, args.filter, solid_mask, args.wildcard)
        for idx, row in tqdm(df.iterrows(), total=len(df))
    )

    ok_count = sum(1 for r in results if "[OK]" in r)
    skip_count = sum(1 for r in results if "[SKIP]" in r)
    err_count = sum(1 for r in results if "[ERROR]" in r)

    print("\n--- RESUMEN FINAL ---")
    print(f"[OK] Exitosos: {ok_count}")
    print(f"[SKIP] Descartados por QC: {skip_count}")
    print(f"[ERROR] Errores: {err_count}")

    if err_count > 0 or skip_count > 0:
        log_path = output_root / "logs.txt"
        print(f"\nDetalle de descartados/errores guardado en {log_path}")
        with open(log_path, "w") as f:
            for r in results:
                if "[ERROR]" in r or "[SKIP]" in r:
                    f.write(r + "\n")


if __name__ == "__main__":
    main()