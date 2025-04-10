#!/bin/bash
#SBATCH --job-name=new_acsr
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:A40:1          # Request 2 GPUs of any type
#SBATCH --mem=3G
#SBATCH --export=ALL                 # Export your environment to the compute node
#SBATCH --output=/pasteur/appa/homes/bsow/ACSR_github/logs/%x.log

echo "Running decoding"

# Run Python script with alpha, beta, and beam_width as arguments
python /pasteur/appa/homes/bsow/ACSR_github/src/train.py

# Log job completion
echo "Job finished at: $(date)"
