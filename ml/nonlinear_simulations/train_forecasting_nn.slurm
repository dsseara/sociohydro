#!/bin/bash
#
#SBATCH --job-name=nonlinear_NN
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --ntasks=2
#SBATCH --mem=4G

cd /home/jcolen/sociohydro/nonlinear_simulations
module load container_env pytorch-gpu/2.1.0

# Check python
crun -p /home/jcolen/envs/sociohydro python -c "print('Hello world')"

phases=("segregated" "integrated" "migration1" "migration2")

phase=${phases[${SLURM_ARRAY_TASK_ID}]}

echo "STARTING" $phase ${SLURM_ARRAY_TASK_ID}

crun -p /home/jcolen/envs/sociohydro python train_forecasting_nn.py \
    --n_epochs 200 \
    --savename models/SimulationForecasting_${phase}_${SLURM_ARRAY_TASK_ID} \
    --dynamic_type ${phase}