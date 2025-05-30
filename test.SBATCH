#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=res
#SBATCH --account=pr_113_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs
export HF_ENDPOINT=https://hf-mirror.com

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/scd.ext3:rw \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate rscd; export CUDA_HOME=/usr/local/cuda-12.1; cd /scratch/ds5725/alvpr/Robust-Scene-Change-Detection; python test_grounding_dino.py"
