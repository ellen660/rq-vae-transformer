#!/bin/bash

# Path to the YAML file
yaml_file="mlm"
path="/data/scratch/ellen660/rq-vae-transformer/rqvae/params/$yaml_file.yaml"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run_name="30_seconds_with_layernorm"
resume_from=""

# Set PATH_TO_USE based on whether RESUME_PATH is empty or not
if [ -n "$resume_from" ]; then
    path="$resume_from/config.yaml"
else
    path="$path"
fi

echo "Using path: $path"

declare -A hyperparameters
hyperparameters=(
  [".optimizer.init_lr"]="0.0005"
  [".dataset.batch_size"]="8"
  [".common.max_epoch"]="1000"
  [".dataset.masking_ratio"]="0.5"
)

#Iterate over a bunch 
lr_list=("0.0005")  
batch_size_list=("12")  
masking_ratio=("0.5")

# Iterate over all combinations of hyperparameters
for lr in "${lr_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    for mask in "${masking_ratio[@]}"; do
      hyperparameters[".optimizer.init_lr"]=$lr
      hyperparameters[".dataset.batch_size"]=$batch_size
      hyperparameters[".dataset.masking_ratio"]=$mask

      yq -yi '.exp_details.description = "30_seconds"' "$path"

      # Replace parameters in the YAML file using yq
      # Shouldn't need to edit this part
      comment=""
      for param in "${!hyperparameters[@]}"; do
          new_value="${hyperparameters[$param]}"
          yq -yi "$param = $new_value" "$path"
          #split the param by '.' and get the last element
          param_name=$(echo $param | rev | cut -d'.' -f1 | rev)
          comment="$comment $param_name=$new_value"
          echo $comment
      done

      # Log directory
      curr_time=$(date +%Y%m%d)
      curr_minute=$(date +%H%M)
      curr_time="$curr_time-$curr_minute"
      log_dir="/data/scratch/ellen660/rq-vae-transformer/tensorboard/$yaml_file/$run_name/$curr_time/$comment"

      # Run the Python training script with the updated parameters
      python train_clean.py --config "$yaml_file" --log_dir "$log_dir" --resume_from "$resume_from"
    done
  done
done