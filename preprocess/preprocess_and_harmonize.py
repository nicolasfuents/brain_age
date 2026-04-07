#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

IN_DIR = PROJECT_ROOT / "data" / "OASIS3" / "MR_867_HC"
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


def select_t1_file(raw_path_dir, dir_filter=None):
    if dir_filter:
        target_dir = raw_path_dir / dir_filter
        if not target_dir.exists():
            return None, f"Directorio filtro '{dir_filter}' no existe"
        nifti_files = list(target_dir.rglob("*T1w*.nii")) + list(target_dir.rglob("*T1w*.nii.gz"))
    else:
        nifti_files = list(raw_path_dir.rglob("*T1w*.nii")) + list(raw_path_dir.rglob("*T1w*.nii.gz"))

    if len(nifti_files) == 0:
        return None, "Imagen T1w no encontrada"

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
def process_subject(row, idx, output_root, dir_filter, solid_mask):
    gpu_id = idx % N_GPUS

    subject_id = str(row["Subject_ID"]).strip()
    age = float(row["Age"])

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

    raw_path_dir = IN_DIR / subject_id
    raw_path, warning_msg = select_t1_file(raw_path_dir, dir_filter)

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
        ten_sag = stack_central_slices(data_norm, 0)
        ten_cor = stack_central_slices(data_norm, 1)
        ten_ax  = stack_central_slices(data_norm, 2)

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
            "harmonization": "brainprep_MNI152_N4 + SOLID_v2_mask + robust_P01_P99",
        }
        age_t = torch.tensor(age, dtype=torch.float32)

        torch.save(
            {"image": torch.from_numpy(ten_ax), "age": age_t, "meta": meta},
            t1_tensor_dir / "axial" / f"{subject_id}.pt"
        )
        torch.save(
            {"image": torch.from_numpy(ten_cor), "age": age_t, "meta": meta},
            t1_tensor_dir / "coronal" / f"{subject_id}.pt"
        )
        torch.save(
            {"image": torch.from_numpy(ten_sag), "age": age_t, "meta": meta},
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
        delayed(process_subject)(row, idx, output_root, args.filter, solid_mask)
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