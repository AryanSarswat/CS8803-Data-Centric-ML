#!/bin/bash

#SBATCH -JExpt_Baseline                           # Job name
#SBATCH -N1 --gres=gpu:1 -C A100-40GB --ntasks-per-node=20   # Number of nodes, GPUs, and cores required
#SBATCH -t960                            # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=che321@gatech.edu           # e-mail address for notifications

cd $HOME/scratch/DML/CS8803-Data-Centric-ML/                       # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
conda activate dml
python3 main.py
