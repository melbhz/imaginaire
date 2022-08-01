#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition interactive
#SBATCH --time 1-23:00:00 
#SBATCH --cpus-per-task=8 

cd /data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments/zoom20_test
module load numpy/1.18.0-python-3.7.4
module load opencv
python3 NHMRC_v8_cvpool_zm20.py
##sinteractive -p interactive --time=2:59:00 --cpus-per-task=8
##sinteractive -p shortgpgpu --time=1:00:00 --qos=gpgpuresplat --gres=gpu:1 --cpus-per-task=24
