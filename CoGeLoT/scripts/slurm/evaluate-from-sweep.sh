#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=22
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --output=jobs/eval/%x.%j.out

module load apps/ffmpeg

SLURM_JOB_NAME='bash' pdm run python -m wandb agent daiki-takahashi-/default_200M-evaluation/"$1"
