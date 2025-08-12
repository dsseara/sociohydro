#!/bin/bash
#
#SBATCH --job-name=census_forecasting
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --ntasks=2
#SBATCH --mem=8G

cd /home/jcolen/sociohydro/census_density_gridded
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

model_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "STARTING TO TRAIN" ${model_id}

configs=($(ls configs/relative_density_with_housing_sigma* | sort))
config="${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Using configuration ${config}"

crun -p /home/jcolen/envs/sociohydro python train_census_nn.py --model_id $model_id --config "$config"

echo "Starting to compute saliency" $model_id

crun -p /home/jcolen/envs/sociohydro python compute_census_saliency.py --model_id ${model_id}
