#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=96     
#SBATCH --time=2:58:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nicolasfuentes11@gmail.com
#SBATCH -p compute
#SBATCH --output=../logs/prep_intecnus-%j.out
#SBATCH --error=../logs/prep_intecnus-%j.err

# 1. Limpiar módulos del cluster
module --force purge
module load StdEnv/2023
module load dcm2niix/1.0.20230411
module load fsl/6.0.7.18
module load ants/2.6.2
module load freesurfer/7.4.1
module load cuda/12.6

# License de Freesurfer (CRÍTICO)
export FS_LICENSE=$HOME/.licenses/freesurfer.lic
if [ ! -f "$FS_LICENSE" ]; then
    echo "¡CUIDADO! No encuentro la licencia de FreeSurfer en $FS_LICENSE"
fi

source $FSLDIR/etc/fslconf/fsl.sh

# --- 2. ACTIVAR CONDA ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate brain_age_env

# A. Buscamos la librería C++ de Conda automáticamente
LIB_STD=$(find $CONDA_PREFIX/lib -name "libstdc++.so.6" | head -n 1)

if [ ! -z "$LIB_STD" ]; then
    echo "Inyectando librería C++: $LIB_STD"
    export LD_PRELOAD=$LIB_STD
else
    echo "ERROR CRÍTICO: No encontré libstdc++.so.6"
    exit 1
fi

# B. ESCUDO CONTRA CRASHES
export GLIBC_TUNABLES=glibc.malloc.tcache_count=0

# C. Configuración de Hilos
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

# --- 5. EJECUCIÓN ---
echo "Iniciando procesamiento unificado en $(hostname)"
cd ~/scratch/brain_age_project/openBHB_dataset/scripts/preprocess

# Ejecutamos el script unificado
#"$CONDA_PREFIX/bin/python" -u preprocess_vMarzo2026.py
#"$CONDA_PREFIX/bin/python" -u preprocess.py
"$CONDA_PREFIX/bin/python" -u preprocess_INTECNUS.py