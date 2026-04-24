#!/bin/bash
#SBATCH -p cornell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -J test_gpu
#SBATCH -o logs/%x-%j.out

hostname
nvidia-smi
sleep 30