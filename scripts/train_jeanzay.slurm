#!/bin/bash
#SBATCH --job-name=adperc_centerpoint
#SBATCH --account=cks@h100
#SBATCH --constraint=h100
#SBATCH --partition=gpu_p6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/fswork/projects/rech/cks/uso64da/AdPerception/logs/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/cks/uso64da/AdPerception/logs/%j.err

module purge
module load arch/h100
module load anaconda-py3/2024.06
module load cuda/12.1.0

conda activate /lustre/fsn1/projects/rech/cks/uso64da/conda/envs/AdPerception

export WORK=/lustre/fswork/projects/rech/cks/uso64da
export SCRATCH=/lustre/fsn1/projects/rech/cks/uso64da
export PYTHONPATH=$WORK/AdPerception/third_party/Pointcept:$WORK/AdPerception/third_party/SimpleTrack:$WORK/AdPerception
export TMPDIR=$SCRATCH/tmp
export CUDA_LAUNCH_BLOCKING=1

mkdir -p $WORK/AdPerception/logs $SCRATCH/tmp

cd $WORK/AdPerception

RESUME=""
if [ -f "checkpoints/resume.pth" ]; then
    RESUME="--resume checkpoints/resume.pth"
    echo "Resuming from checkpoint"
fi

python -u scripts/train.py \
    --config    jeanzay \
    --version   v1.0-trainval \
    --dataroot  /lustre/fsn1/projects/rech/cks/uso64da/data/nuscenes \
    --epochs    5 \
    --lr        1e-2 \
    --workers   8 \
    --no-amp \
    $RESUME

echo "Job finish : $(date)"
