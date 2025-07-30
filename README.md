# multi_node_inference

Working code for multi nodes inference on slurm cluster (tested on [Jean-Zay](http://www.idris.fr/eng/jean-zay/index.html))

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


## TOOD:
- test with vllm