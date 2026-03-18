#!/bin/bash
#SBATCH --job-name=train_triplane
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=nicolasfuentes11@gmail.com
#SBATCH --output=../logs/train_triplane_%A_%a.out
#SBATCH --error=../logs/train_triplane_%A_%a.err

# ------------------------------------------------------------------------------
# DEFINICIÓN DE LA BATERÍA DE MODELOS
# Formato: "PLANO BACKBONE LOSS_TYPE SIGMA NBLOCK AUG_ROT AUG_NOISE LR SEED TAG"
# ------------------------------------------------------------------------------
MODELS=(
    # --- CONFIGURACIÓN EXACTA SEGÚN TABLA DE REFERENCIA ---
    "coronal vgg16 mse 1.0 8 15 0.02 0.0001 42 coronal_final"
    "sagittal vgg16 mse 1.0 6 10 0.01 0.0001 42 sagittal_final"
    "axial resnet18 soft 2.0 6 10 0.01 0.0001 42 axial_final"
)

# Parseo de la configuración
IFS=" " read -r PLANO BACKBONE LOSS SIGMA BLOCKS AUG_ROT AUG_NOISE LR SEED TAG <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

# IDENTIFICADOR COMÚN
COMMON_ID="TRIPLANE_FINAL_NO_OASIS_${SLURM_ARRAY_JOB_ID}"

echo "================================================================================"
echo ">>> JOB ID: ${SLURM_ARRAY_TASK_ID} | EXPERIMENTO: ${COMMON_ID}"
echo ">>> CONFIG: Plano=$PLANO | Backbone=$BACKBONE | Loss=$LOSS | Bloques=$BLOCKS"
echo ">>> AUG: Rot=$AUG_ROT | Noise=$AUG_NOISE | LR=$LR | Seed=$SEED | Tag='$TAG'"
echo "================================================================================"

# 1. Entorno
export TORCH_HOME=/scratch/nfuentes/brain_age_project/torch_cache
module --force purge
module load StdEnv/2023

source ~/miniforge3/etc/profile.d/conda.sh
conda activate brain_age_env

# 2. Preload
LIB_STD=$(find $CONDA_PREFIX/lib -name "libstdc++.so.6" | head -n 1)
if [ ! -z "$LIB_STD" ]; then export LD_PRELOAD=$LIB_STD; fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/brain_age_project/openBHB_dataset/scripts/train

# 3. EJECUCIÓN
"$CONDA_PREFIX/bin/python" train_improved.py \
    --planes $PLANO \
    --backbone $BACKBONE \
    --loss_type $LOSS \
    --sigma $SIGMA \
    --nblock $BLOCKS \
    --aug_rot $AUG_ROT \
    --aug_noise $AUG_NOISE \
    --lr $LR \
    --timestamp $COMMON_ID \
    --suffix "$TAG" \
    --seed $SEED \
    --drop_rate 0.2 \
    --enable-aug

echo ">>> FIN DEL JOB."