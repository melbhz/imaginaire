#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpgpuresplat
#SBATCH --job-name="AlcDrink_zm20"
#SBATCH --time=6-20:00:00
#SBATCH --mem=20G

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Note: --nproc_per_node=2 should be the same as --gres=gpu:p100:2, otherwise gpu resource wasted
## CHANGE HERE
cd /data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments_PAPER/base10_zoom20/AlcDrink

module load gcccore/10.2.0
module load python/3.8.6
module load cudnn/8.0.4.30-cuda-11.1.1

## CHANGE HERE
python -m torch.distributed.launch --nproc_per_node=1 --master_port=9902 /data/scratch/projects/punim1358/HZ_GANs/imaginaire/train.py \
--config ./AlcDrink.yaml #\--resume 1 --checkpoint /data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments/AlcDrink_zL2H_style10/logs/2022_0509_2228_16_AlcDrink_style10/epoch_00002_iteration_000165000_checkpoint.pt
