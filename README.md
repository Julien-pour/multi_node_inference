# multi_node_inference

Working code for multi nodes inference on slurm cluster (tested on [Jean-Zay](http://www.idris.fr/eng/jean-zay/index.html))

# SGLang (recommended)

## Installation steps

1. (Recommended) Create a new conda environment.
```
module load arch/h100
module load python/3.11.5
conda create -n sglang python=3.11 -y
module load cuda/12.8.0
conda activate sglang
```

2. Install SGLang (check latest version here: [https://pypi.org/project/sglang/](https://pypi.org/project/sglang/))

```
pip install sglang[all]==0.4.9.post5
```


## serve 

1. Basic slurm script (that also launch a python script asynchronously for your experience for example)

Slurm script on how to launch SGLang engine on multiple nodes and test it at [multi_node.sh](multi_node.sh) (you can change the number of nodes, gpu, model and other params )
```
sbatch multi_node.sh 
```

2. Python script to launch automatically SGLang

You just need to call the LLMClient class in [llm_client.py](llm_client.py) directly from python.
Then you just need to launch normally your script with slurm, see example here: [llm_client_test.py](llm_client_test.py)



# vLLM 

Prefere SGLang see this issues:
- cannot use $JOBSCRATCH here, see https://github.com/ray-project/ray/issues/7724
but due to the 107 character socket length limit in unix
so you need to clean your tmpdir manually 
export TMPDIR=$JOBSCRATCH

- cannot use Expert Parallel ): in multi-node vllm as you need sudo to install this:
https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels
see this for more details: 
https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html


## Installation steps 

1. (Recommended) Create a new conda environment.
```
module load arch/h100
module load python/3.11.5
conda create -n vllm10 python=3.11 -y
module load cuda/12.8.0
conda activate vllm10
```

2. Install vLLM (check latest version here: [https://pypi.org/project/vllm/](https://pypi.org/project/vllm/))

```
pip install vllm
```

## serve 

1. Basic slurm script (that also launch a python script asynchronously for your experience for example)
Slurm script on how to launch vLLM engine on multiple nodes and test it at [multi_node_vllm.sh](multi_node_vllm.sh) (you can change the number of nodes, gpu, model and other params )



## TOOD:
- add init method to launch vLLM in llm_client_test.py (multi nodes)