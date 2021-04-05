#!/bin/bash

#SBATCH --job-name=qa
#SBATCH --output=qa_so.%j.out
#SBATCH --error=qa_so.%j.err

#SBATCH  --partition=xye
##SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
#SBATCH  --ntasks-per-node=2
#SBATCH  --mem=64G

module load cuda
module load singularity

singularity exec --nv docker://woffee/keras:v2 sh run.sh
