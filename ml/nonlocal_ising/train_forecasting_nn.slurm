#!/bin/bash
#
#SBATCH --job-name=nonlocal_ising
#SBATCH --output=/home/jcolen/scratch/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --account=pi-vitelli
#SBATCH --mem=4G

source /home/jcolen/.bashrc
module load python

cd /project/vitelli/jonathan/sociohydro/nonlocal_ising

# ACTIVATE CONDA env 
conda activate /project/vitelli/jonathan/sociohydro_env
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda activate /project/vitelli/jonathan/sociohydro_env
# Something is wrong in my bashrc and above seems to workaround

echo "STARTING" ${SLURM_ARRAY_TASK_ID}

python train_forecasting_nn.py \
    --n_epochs 100 \
	--sigma 10 \
    --savename models/SimulationForecasting_sigma10_${SLURM_ARRAY_TASK_ID}
