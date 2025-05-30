#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/slurm-output-combined/out_%A.out
#SBATCH --job-name=cd_ddp1
#SBATCH --account=pr_113_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/combine.ext3:rw \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate rscd; torchrun --nproc_per_node=4 src/scripts/train_para.py src/scripts/configs/train_combine.yml"
