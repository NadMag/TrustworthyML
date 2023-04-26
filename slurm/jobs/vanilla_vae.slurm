#!/bin/sh
#SBATCH --job-name=tml_hw1
#SBATCH --output=/home/ML_courses/03683533_2021/nadav_ohad_dvir/nadav/TrustworthyML/slurm/logs/tml_hw1.out
#SBATCH --error=/home/ML_courses/03683533_2021/nadav_ohad_dvir/nadav/TrustworthyML/slurm/logs/tml_hw1.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --gpus=1
python ./main_a.py
