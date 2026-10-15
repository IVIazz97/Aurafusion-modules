#!/bin/bash
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# Leonardo (CINECA) wrapper around scripts/run.sh: fills in the envs and checkpoints there. Submit from the repo root:
#   sbatch --job-name=AF3DGS_carton --export=ALL,PLY=...,COLMAP=...,MASK_DIR=...,OUT=output/carton scripts/leonardo/run.sbatch
# (sbatch --export splits on commas: pass STEPS as a+b+c.)
set -euo pipefail
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp/job_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor" TRITON_CACHE_DIR="$TMPDIR/triton"   # not the submitting shell's
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export GS_PY=${GS_PY:-/leonardo_work/IscrC_EditGS/envs_inpaint/aurafusion_3dgs/bin/python}
export AF_PY=${AF_PY:-/leonardo_scratch/fast/IscrC_EditGS/envs/aurafusion/bin/python}
export AF360_REPO=${AF360_REPO:-/leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official}
export CKPT_DIR=${CKPT_DIR:-/leonardo_work/IscrC_EditGS/repos/CVPR2027/AuraFusion-3DGS/checkpoints}
echo "host=$(hostname)"
bash "${SLURM_SUBMIT_DIR:-.}/scripts/run.sh"
