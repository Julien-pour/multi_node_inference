#!/bin/bash

#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --job-name=test_multi_node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=./out/test-%A.out
#SBATCH --error=./out/test-%A.out


# /!\ problem with vllm multi-node: /!\

# cannot use $JOBSCRATCH here, see https://github.com/ray-project/ray/issues/7724
# but due to the 107 character socket length limit in unix
# so you need to clean your tmpdir manually 
# export TMPDIR=$JOBSCRATCH

# cannot use Expert parallel ): in multi-node vllm as you need sudo to install this:
# https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels
# see this for more details: 
# https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html



# TL;DR: use SGLang except if problem with SGLang, then use vllm

module purge
module load arch/h100
module load python/3.11.5
ulimit -c 0
module load cuda/12.8.0
conda activate vllm10
cd /lustre/fswork/projects/rech/imi/uqv82bm/multi_node/



model="/lustre/fsn1/projects/rech/imi/uqv82bm/hf/DeepSeek-R1-0528"


# Infer tensor parallelism (tp) automatically: total GPUs across all nodes
tp=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." 
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle potential space-separated IP addresses (IPv4/IPv6) - take the first one (try this one)
head_node_ip=$(echo $head_node_ip | cut -d' ' -f1)

export VLLM_HOST_IP=$head_node_ip

# --- Launch Head Node ---
echo "Head node: $head_node, Head node IP: $head_node_ip"


srun --nodes=1 --ntasks=1 -w "$head_node" bash -c \
"export OUTLINES_CACHE_DIR=/tmp/${SLURM_JOB_ID}_0 && \
 export VLLM_HOST_IP=$head_node_ip && \
 echo \"BEGIN_IP on Head Node:\" && hostname -I && echo \"END_IP on Head Node\" && \
 ray start --head --port=5000 && sleep 10 && ray status && \
 vllm serve $model --port 9876 --trust-remote-code \
  --tensor-parallel-size $tp \
  --gpu-memory-utilization 0.85" &
  

HEAD_PID=$! # Capture the background process ID


# --- Give Head Node Time to Initialize ---
echo "Waiting for head node to initialize..."
# sleep 30 # Adjust this time if necessary, maybe 15-60 seconds


# --- Launch Worker Nodes ---

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node

# Loop starts from 1 because 0 is the head node
for ((i = 1; i < worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i (Rank $i) at $node_i"

  srun --nodes=1 --ntasks=1 -w "$node_i" bash -c \
  "export OUTLINES_CACHE_DIR=/tmp/${SLURM_JOB_ID}_${i} && \
   worker_node_ip=\$(hostname --ip-address | cut -d' ' -f1) && \
  export VLLM_HOST_IP=\$worker_node_ip &&  \
  echo 'VLLM_HOST_IP for worker $i: \$VLLM_HOST_IP' &&  \
  ray start --block --address=${head_node_ip}:5000" &

  WORKER_PID=$! # Capture the last worker's PID if needed for specific waiting

done

# --- Wait for all background processes (head and workers) to start ---
# This waits for the head node and all workers launched in the background.
# It doesn't guarantee they are fully initialized, but ensures they were started.
# wait $HEAD_PID
# Optionally wait for the last worker specifically if you need to ensure it started
# wait $WORKER_PID 


# --- Run Test Script ---
# This should run AFTER all server processes are initiated.
# Use the port of the head node as that's typically where the service endpoint is.
echo "All server processes initiated. Running test script..."
python test.py --model_path "$model" --port 9876 # <-- Use port args of vllm serve launched on head node

# --- Keep Job Alive ---
echo "Test script finished. Keeping job alive for 4 hours..."
sleep 4h

echo "Job script finished."