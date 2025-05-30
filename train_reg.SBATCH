#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=cyws3d
#SBATCH --account=pr_113_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/habitat.ext3:rw \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate cyws3d; python src/scripts/train.py src/scripts/configs/train.yml"
