#!/bin/bash
#SBATCH --job-name=gpu    # job name
#SBATCH --output=gpu_tf.out  # output log file
#SBATCH --error=gpu_tf.err   # error file
#SBATCH --time=25:00:00   # 1 hour of wall time
#SBATCH --nodes=1         # 1 GPU node
#SBATCH --ntasks=1        # Number of MPI tasks (i.e. processes)
#SBATCH --mem=120000       # Job memory request
#SBATCH --partition=gpu2  # GPU2 partition
#SBATCH --gres=gpu:1      # Request 4 GPU

# Load all required modules below. 
module purge
module load cuda/10.1
module load Anaconda3/2019.03
source activate /home/jbertini/.conda/envs/tf-gpu-2.2-conda-10.1 
# Add lines here to run your GPU-based computations.
~/.conda/envs/tf-gpu-2.2-conda-10.1/bin/python train.py
