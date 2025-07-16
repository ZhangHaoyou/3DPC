#!/bin/bash
#SBATCH --job-name=fem_main
#SBATCH --partition=ai
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=log/main_out.log
#SBATCH --error=log/main_err.log

# Set working directory
cd ~/research/pc_fem

# Load Python module (if needed, depending on how venv was created)
module load python/3.10.13

# Activate virtual environment
source ~/env/interface/bin/activate

# Run your Python program
python3 main.py