#!/bin/bash
#SBATCH --job-name=train_triplane_mse
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=nicolasfuentes11@gmail.com
#SBATCH --output=../logs/train_mse_%A_%a.out
#SBATCH --error=../logs/train_mse_%A_%a.err

# ------------------------------------------------------------------------------
# DEFINICIÓN DE LA BATERÍA DE MODELOS (TODO VGG16 + MSE)
# Formato: "PLANO BACKBONE LOSS_TYPE SIGMA NBLOCK AUG_ROT AUG_NOISE LR PATIENCE SEED TAG"
# ------------------------------------------------------------------------------
MODELS=(
    "coronal vgg16 mse 1.0 8 15 0.02 0.0001 30 42 coronal_all_mse"
    "sagittal vgg16 mse 1.0 6 10 0.01 0.0001 40 42 sagittal_all_mse"
    "axial vgg16 mse 1.0 6 10 0.01 0.0001 40 42 axial_all_mse"
)

IFS=" " read -r PLANO BACKBONE LOSS SIGMA BLOCKS AUG_ROT AUG_NOISE LR PATIENCE SEED TAG <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

COMMON_ID="TRIPLANE_ALL_MSE_NO_OASIS_${SLURM_ARRAY_JOB_ID}"

echo "================================================================================"
echo ">>> JOB ID: ${SLURM_ARRAY_TASK_ID} | EXPERIMENTO: ${COMMON_ID}"
echo ">>> CONFIG: Plano=$PLANO | Backbone=$BACKBONE | Loss=$LOSS | Bloques=$BLOCKS"
echo ">>> AUG: Rot=$AUG_ROT | Noise=$AUG_NOISE | LR=$LR | Patience=$PATIENCE | Tag='$TAG'"
echo "================================================================================"

export TORCH_HOME=/scratch/nfuentes/brain_age_project/torch_cache
module --force purge
module load StdEnv/2023

source ~/miniforge3/etc/profile.d/conda.sh
conda activate brain_age_env

LIB_STD=$(find $CONDA_PREFIX/lib -name "libstdc++.so.6" | head -n 1)
if [ ! -z "$LIB_STD" ]; then export LD_PRELOAD=$LIB_STD; fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/brain_age_project/openBHB_dataset/scripts/train

"$CONDA_PREFIX/bin/python" train_improved.py \
    --planes $PLANO \
    --backbone $BACKBONE \
    --loss_type $LOSS \
    --sigma $SIGMA \
    --nblock $BLOCKS \
    --aug_rot $AUG_ROT \
    --aug_noise $AUG_NOISE \
    --lr $LR \
    --patience $PATIENCE \
    --timestamp $COMMON_ID \
    --suffix "$TAG" \
    --seed $SEED \
    --drop_rate 0.2 \
    --enable-aug

echo ">>> FIN DEL JOB."