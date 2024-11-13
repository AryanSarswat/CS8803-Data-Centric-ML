#!/bin/bash

#SBATCH -JData_Selection_Test                           # Job name
#SBATCH -N1 --gres=gpu:RTX_6000:1 --ntasks-per-node=20   # Number of nodes, GPUs, and cores required
#SBATCH -t960                            # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=knazarali3@gatech.edu           # e-mail address for notifications

cd $HOME/scratch/CS8803-Data-Centric-ML/                       # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
conda init
conda activate dml_project
python3 main_data_selection.py --image_data_folder ../coco/images --pickle_folder ./ --epochs 100 --batch_size 128 --learning_rate 1e-5 --weight_decay 1e-5 --num_workers 20 --log_wandb True --project_name svp --experiment_name vit_data_selection_rtx_0.00001 --save_dir saved_models

