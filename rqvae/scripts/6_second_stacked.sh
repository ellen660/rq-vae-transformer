#!/bin/bash

# Path to the YAML file
yaml_file="autoregressive_stacked"
path="/data/scratch/ellen660/rq-vae-transformer/rqvae/params/$yaml_file.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_name="6_second_6_codebooks"
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
  [".dataset.stack_every"]="5"
  [".loss.num_steps"]="4"
)

#Iterate over a bunch 
lr_list=("0.0005")  
batch_size_list=("4")  
num_steps_list=("4")
stack_every_list=("5")
embed_dim=("256") #MAKE SURE OF THIS

# Iterate over all combinations of hyperparameters
for lr in "${lr_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    for num_steps in "${num_steps_list[@]}"; do
      for embed in "${embed_dim[@]}"; do
        for stack_every in "${stack_every_list[@]}"; do
          hyperparameters[".optimizer.init_lr"]=$lr
          hyperparameters[".dataset.batch_size"]=$batch_size
          hyperparameters[".loss.num_steps"]=$num_steps
          hyperparameters[".dataset.stack_every"]=$stack_every

          yq -yi '.exp_details.description = "6_second_6_codebooks"' "$path"

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
    done
done