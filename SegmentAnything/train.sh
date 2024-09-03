#!/bin/bash
#SBATCH --job-name=u
#SBATCH --nodes=1
#SBATCH --partition=partition_gpu_1
#SBATCH --nodelist=node4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=./slurm_output/job_output%A.out

python train.py -bs 1  -p pretrained_weights/sam_vit_b_01ec64.pth