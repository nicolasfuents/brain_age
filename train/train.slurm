#!/bin/bash
#SBATCH --job-name=train_all_planes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4
#SBATCH --time=23:59:00
#SBATCH -p compute_full_node
#SBATCH --output=../logs/slurm-%j.out
#SBATCH --error=../logs/slurm-%j.err


module load anaconda3
source activate gpu_env

cd ~/scratch/brain_age_project/openBHB_dataset/scripts/train

echo "Ejecutando entrenamiento..."
python train.py --ensemble --gaussian-noise
