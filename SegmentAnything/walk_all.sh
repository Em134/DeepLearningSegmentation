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


image_encoder_freeze_strategies=()
prompt_encoder_freeze_strategies=( 'prompt_encoder' 'prompt_encoder' '' '')
mask_decoder_freeze_strategies=( 'mask_decoder' '' 'mask_decoder' '')

for ((i=11; i>=-1; i--)); do
  strategy=""
  for ((j=i; j>=0; j--)); do
    strategy+="image_encoder.block.$j. "
  done
  image_encoder_freeze_strategies+=("$strategy")
done

for image_encoder in "${image_encoder_freeze_strategies[@]}"; do
    for prompt_encoder in "${prompt_encoder_freeze_strategies[@]}"; do
        for mask_decoder in "${mask_decoder_freeze_strategies[@]}"; do
            python all_steps.py -f $image_encoder $prompt_encoder $mask_decoder -p pretrained_weights/sam_vit_b_01ec64.pth -bs 1
        done
    done
done