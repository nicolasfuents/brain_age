#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4
#SBATCH --time=23:58:00
#SBATCH -p compute_full_node
#SBATCH --output=../logs/preprocess-%j.out
#SBATCH --error=../logs/preprocess-%j.err


module load anaconda3
source activate gpu_env

cd ~/scratch/brain_age_project/openBHB_dataset/scripts/eval

python preprocess_openbhb_multiplano.py