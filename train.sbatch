#!/bin/bash
#SBATCH -J cyclone-train
#SBATCH -A ICCS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
python3 -u train.py
