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

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/scd.ext3:rw \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate rscd; python src/scripts/evaluate.py /scratch/ds5725/alvpr/Robust-Scene-Change-Detection/src/output/cd_rs_lang/best.val.pth"
