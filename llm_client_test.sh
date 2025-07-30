#!/bin/bash

#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --job-name=test_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=./out/test-%A.out
#SBATCH --error=./out/test-%A.out
#SBATCH --overcommit



# from https://github.com/sgl-project/sglang/issues/3206


export TMPDIR=$JOBSCRATCH
module purge
module load arch/h100
module load python/3.11.5
ulimit -c 0
module load cuda/12.8.0
conda activate aces_sglang49p5
cd /lustre/fswork/projects/rech/imi/uqv82bm/multi_node/
python llm_client.py --model "/lustre/fsn1/projects/rech/imi/uqv82bm/hf/Qwen3-Coder-480B-A35B-Instruct-FP8" --ep_moe