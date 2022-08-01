#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpgpuresplat
#SBATCH --job-name="BMI_zm20"
#SBATCH --time=6-20:00:00
#SBATCH --mem=20G

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Note: --nproc_per_node=2 should be the same as --gres=gpu:p100:2, otherwise gpu resource wasted
## CHANGE HERE
cd /data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments_PAPER/base10_zoom20/BMI

module load gcccore/10.2.0
module load python/3.8.6
module load cudnn/8.0.4.30-cuda-11.1.1

## CHANGE HERE
python -m torch.distributed.launch --nproc_per_node=1 --master_port=9904 /data/scratch/projects/punim1358/HZ_GANs/imaginaire/train.py \
--config ./BMI.yaml #\--resume 1 --checkpoint /data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments/BMI_zlow2high_style10/logs/2022_0516_1037_08_BMI_style10/epoch_00000_iteration_000100000_checkpoint.pt
