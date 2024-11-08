#!/bin/bash

#SBATCH -JExpt_Baseline                           # Job name
#SBATCH -N1 --gres=gpu:1 --ntasks-per-node=20   # Number of nodes, GPUs, and cores required
#SBATCH -t960                            # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=che321@gatech.edu           # e-mail address for notifications

cd /home/hice1/che321/scratch/DML/CS8803-Data-Centric-ML                       # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
conda init
conda activate dml
python3 main.py --image_data_folder "../datasets/coco/images" --pickle_folder "pickle_folder" --epochs 100 --batch_size 256 --learning_rate 1e-4 --weight_decay 1e-6 --num_workers 20 --log_wandb True --project_name svp --experiment_name bce --save_dir saved_models

