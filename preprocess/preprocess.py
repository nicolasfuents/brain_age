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

# ==============================================================================
# CONFIGURACION
# ==============================================================================
# Agregamos openBHB_dataset a la raíz para que coincida con tu directorio real
PROJECT_ROOT = Path("/home/nfuentes/scratch/brain_age_project/openBHB_dataset")
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "preprocess"
BASH_SCRIPT_T1 = SCRIPTS_DIR / "brainprep.sh" 

IN_DIR = PROJECT_ROOT / "data" / "OASIS3" / "MR_867_HC"
OUT_DIR = IN_DIR / "processed"
CSV_METADATA = IN_DIR / "participants.csv"


CMD_FSLREORIENT = shutil.which('fslreorient2std')
CMD_FSLORIENT   = shutil.which('fslorient')

FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:
    sys.exit("[ERROR] FSLDIR no definido. Cargá el módulo FSL.")
MNI_TEMPLATE = Path(FSLDIR) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def repair_header_and_reorient(raw_path, output_std_path, work_dir):
    try:
        subprocess.run([CMD_FSLREORIENT, str(raw_path), str(output_std_path)], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8')
        if "Orientation information not stored" in err_msg or "non-zero code" in err_msg:
            temp_nii = work_dir / f"temp_fix_{os.getpid()}.nii.gz"
            shutil.copy(raw_path, temp_nii)
            try:
                subprocess.run([CMD_FSLORIENT, "-setqformcode", "1", str(temp_nii)], check=True)
                subprocess.run([CMD_FSLORIENT, "-setsformcode", "1", str(temp_nii)], check=True)
                subprocess.run([CMD_FSLREORIENT, str(temp_nii), str(output_std_path)], 
                               check=True, stdout=subprocess.DEVNULL)
                if temp_nii.exists(): os.remove(temp_nii)
                return True
            except Exception:
                if temp_nii.exists(): os.remove(temp_nii)
                return False
        return False

def stack_central_slices(vol, axis, num=5):
    assert vol.ndim == 3
    c = vol.shape[axis] // 2
    half = num // 2
    idxs = [c + o for o in range(-half, half + 1)]
    raw_slices = []
    for i in idxs:
        if axis == 0: sl = vol[i, :, :]
        elif axis == 1: sl = vol[:, i, :]
        else: sl = vol[:, :, i]
        raw_slices.append(sl.astype(np.float32, copy=False))
    return np.stack(raw_slices, 0)

# ==============================================================================
# WORKER PRINCIPAL
# ==============================================================================
def process_subject(row, idx, output_root, dir_filter):
    gpu_id = idx % 4
    
    env_gpu = os.environ.copy()
    env_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_gpu["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    env_gpu["OMP_NUM_THREADS"] = "1"

    # Inyeccion estricta de variables de entorno para sub-shell
    conda_bin = os.path.join(env_gpu.get("CONDA_PREFIX", "/home/nfuentes/miniforge3/envs/brain_age_env"), "bin")
    fs_bin = os.path.join(env_gpu.get("FREESURFER_HOME", ""), "bin")
    fsl_bin = os.path.join(env_gpu.get("FSLDIR", ""), "bin")
    
    env_gpu["PATH"] = f"{conda_bin}:{fs_bin}:{fsl_bin}:" + env_gpu.get("PATH", "")
    
    if "FS_LICENSE" not in env_gpu:
        env_gpu["FS_LICENSE"] = os.path.join(os.environ.get("HOME", ""), ".licenses", "freesurfer.lic")

    # Inyección explícita del path de Conda para resolver brainprep en el sub-shell
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env_gpu["PATH"] = f"{conda_prefix}/bin:" + env_gpu.get("PATH", "")
        
    # Extracción genérica de metadatos (Ajustar según columnas de OASIS)
    subject_id = str(row['Subject_ID']).strip()
    age = float(row['Age'])
    
    subj_dir = output_root / subject_id
    prep_dir = subj_dir / "preprocess"
    t1_tensor_dir = subj_dir / "t1_tensors"
    
    if prep_dir.exists(): shutil.rmtree(prep_dir, ignore_errors=True)
    if t1_tensor_dir.exists(): shutil.rmtree(t1_tensor_dir, ignore_errors=True)
        
    prep_dir.mkdir(parents=True, exist_ok=True)
    t1_tensor_dir.mkdir(parents=True, exist_ok=True)
    for p in ["axial", "coronal", "sagittal"]:
        (t1_tensor_dir / p).mkdir(exist_ok=True)
    
    raw_path_dir = IN_DIR / subject_id
    
    # Lógica de filtrado BIDS (Aplica el flag si existe)
    if dir_filter:
        target_dir = raw_path_dir / dir_filter
        if not target_dir.exists():
            return f"[ERROR] {subject_id}: Directorio filtro '{dir_filter}' no existe"
        nifti_files = list(target_dir.rglob("*T1w*.nii")) + list(target_dir.rglob("*T1w*.nii.gz"))
    else:
        nifti_files = list(raw_path_dir.rglob("*T1w*.nii")) + list(raw_path_dir.rglob("*T1w*.nii.gz"))
    
    if len(nifti_files) == 0: 
        return f"[ERROR] {subject_id}: Imagen T1w no encontrada"
    elif len(nifti_files) > 1:
        # Heurística longitudinal: Seleccionar la sesión baseline (menor dXXXX)
        import re
        def extract_days(filename):
            match = re.search(r'sess-d(\d+)', str(filename))
            return int(match.group(1)) if match else float('inf')
        
        nifti_files.sort(key=lambda x: extract_days(x.name))
        raw_path = nifti_files[0]
        warning_msg = f" (Longitudinal: n={len(nifti_files)}. Seleccionada baseline {raw_path.name})"
    else:
        raw_path = nifti_files[0]
        warning_msg = ""

    try:
        # 1. FSL Reorient
        t1_std = prep_dir / "t1_std_orient.nii.gz"
        if not repair_header_and_reorient(raw_path, t1_std, prep_dir):
            return f"[ERROR] {subject_id}: Fallo reorientación"

        # --- PARCHE DPKG PARA BRAINPREP EN ROCKY LINUX HPC ---
        # Creamos un ejecutable 'dpkg' falso en el directorio temporal
        dummy_dpkg = prep_dir / "dpkg"
        with open(dummy_dpkg, "w") as f:
            f.write("#!/bin/bash\necho ''\n")
        dummy_dpkg.chmod(0o755)
        
        # Inyectamos el directorio del dummy al inicio del PATH
        env_gpu["PATH"] = f"{prep_dir}:" + env_gpu.get("PATH", "")

        # --- PARCHE BUG ANTs N4BiasFieldCorrection EN BRAINPREP ---
        # brainprep formatea mal la lista de salidas agregando un espacio: "[out1, out2]"
        # ANTs lee el segundo archivo como " /home/..." y lo interpreta como ruta relativa.
        # Creamos el subdirectorio fantasma (" ") en el working directory para que escriba sin crashear.
        ghost_dir = os.path.join(os.getcwd(), " " + str(prep_dir / "quasiraw"))
        os.makedirs(ghost_dir, exist_ok=True)

        # 2. Topología (Bash)
        bash_process = subprocess.run(
            ["bash", str(BASH_SCRIPT_T1), str(t1_std), str(prep_dir)], 
            env=env_gpu, capture_output=True, text=True
        )
        
        if bash_process.returncode != 0:
            error_msg = bash_process.stderr.strip().replace('\n', ' | ')
            return f"[ERROR] {subject_id}: Bash falló (Código {bash_process.returncode}). Traza: {error_msg}"
        
        # El nombre de salida está determinísticamente fijado por brainprep
        t1_mni = prep_dir / "quasiraw" / "t1_std_orient_desc-6apply_T1w.nii.gz"
        if not t1_mni.exists():
            return f"[ERROR] {subject_id}: Fallo el pipeline de brainprep. Archivo MNI no generado."

        # 3. QC Automático (fslcc)
        qc_res = subprocess.run(["fslcc", str(t1_mni), str(MNI_TEMPLATE)], 
                                capture_output=True, text=True)
        cc_val = 0.0
        if qc_res.stdout.strip():
            cc_val = float(qc_res.stdout.strip().split('\n')[0].split()[-1])

        if cc_val <= 0.75:
            return f"[SKIP] {subject_id} - QC falló (fslcc={cc_val})"

        # 4. Harmonización P99 y Serialización
        img = nib.load(t1_mni)
        data = np.squeeze(img.get_fdata(dtype=np.float32))

        mask_pos = data > 0
        p99 = np.percentile(data[mask_pos], 99) if mask_pos.any() else data.max()
        if p99 <= 0: p99 = 1.0
        
        data = np.clip(data, 0, p99) / p99

        ten_sag = stack_central_slices(data, 0)
        ten_cor = stack_central_slices(data, 1)
        ten_ax  = stack_central_slices(data, 2)

        # --- NUEVO: Exportar NIfTIs para inspección visual ---
        # 1. El volumen 3D completo preprocesado y normalizado (usamos el affine original)
        nib.save(nib.Nifti1Image(data, img.affine), t1_tensor_dir / f"{subject_id}_full_preprocessed.nii.gz")
        
        # 2. Los tensores 2.5D (transponemos de (5, H, W) a (H, W, 5) para que el visualizador permita hacer scroll)
        nib.save(nib.Nifti1Image(ten_ax.transpose(1, 2, 0), np.eye(4)), t1_tensor_dir / "axial" / f"{subject_id}_tensor.nii.gz")
        nib.save(nib.Nifti1Image(ten_cor.transpose(1, 2, 0), np.eye(4)), t1_tensor_dir / "coronal" / f"{subject_id}_tensor.nii.gz")
        nib.save(nib.Nifti1Image(ten_sag.transpose(1, 2, 0), np.eye(4)), t1_tensor_dir / "sagittal" / f"{subject_id}_tensor.nii.gz")
        # -----------------------------------------------------

        meta = {"source": raw_path.name, "p99": float(p99), "fslcc": cc_val, "csv_id": subject_id}
        age_t = torch.tensor(age, dtype=torch.float32)

        torch.save({"image": torch.from_numpy(ten_ax), "age": age_t, "meta": meta}, 
                   t1_tensor_dir / "axial" / f"{subject_id}.pt")
        torch.save({"image": torch.from_numpy(ten_cor), "age": age_t, "meta": meta}, 
                   t1_tensor_dir / "coronal" / f"{subject_id}.pt")
        torch.save({"image": torch.from_numpy(ten_sag), "age": age_t, "meta": meta}, 
                   t1_tensor_dir / "sagittal" / f"{subject_id}.pt")

        # Limpieza de temporales para ahorrar inodos en el clúster
        shutil.rmtree(prep_dir)

        return f"[OK] {subject_id} - GPU: {gpu_id} (fslcc={cc_val}){warning_msg}"
        
    except Exception as e:
        return f"[ERROR] {subject_id}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--filter", type=str, default=None, help="Subdirectorio BIDS específico a buscar (ej. anat2)")
    args = parser.parse_args()
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"--- PROCESAMIENTO END-TO-END DE LA BASE DE DATOS ---")
    df = pd.read_csv(CSV_METADATA)
    
    n_jobs = 64 
    print(f"Procesando {len(df)} sujetos. Lanzando {n_jobs} workers en paralelo sobre 4 GPUs...")
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_subject)(row, idx, output_root, args.filter) for idx, row in tqdm(df.iterrows(), total=len(df))
    )
    
    ok_count = sum(1 for r in results if "[OK]" in r)
    skip_count = sum(1 for r in results if "[SKIP]" in r)
    err_count = sum(1 for r in results if "[ERROR]" in r)
    
    print("\n--- RESUMEN FINAL ---")
    print(f"[OK] Exitosos: {ok_count}")
    print(f"[SKIP] Descartados por QC: {skip_count}")
    print(f"[ERROR] Errores: {err_count}")
    
    if err_count > 0 or skip_count > 0:
        print("\nDetalle de descartados/errores guardado en logs.txt")
        with open(output_root / "logs.txt", "w") as f:
            for r in results:
                if "[ERROR]" in r or "[SKIP]" in r: 
                    f.write(r + "\n")

if __name__ == "__main__":
    main()
