# To Do:
- image filtering
- image deduplication
- kb builder

## Searxng

1. make sure docker run on correct port
2. make sure settings.yml uses json as output
3. settings.yml must be correct else docker container wont run (common failure is language settings)
4. can configure the search engines and safe searches in settings.yml
5. every change in settings.yml must restart container


## overall:
1. query rewrite with LLM
    1.1 transform content into knowledge base --> `process.py`
        - exploit / explore logic for query rewrite
        - 

2. features in `scraper.py`
    - checkpointing and dynamic write
    - after links are processed push to a done location OR set logic to skip those since already scraped


3. features in `scraper.py`


## KB and LLM knowledge extraction:
- singapore context domain
- named entities
- entities, objects, scenes/events/concepts


## Current Limitations:
1. Too many request on searxng

2. need to parameter tune for the threshold for gliner, etc

3. 



### tweaks:
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8123 --gpu-memory-utilization 0.85 --max-model-len 16k
CUDA_VISIBLE_DEVICES=4 vllm serve google/gemma-3-12b-it --port 8124 --gpu-memory-utilization 0.85 --max-model-len 24k
CUDA_VISIBLE_DEVICES=5 vllm serve unsloth/Llama-3.2-3B-Instruct --port 8125 --gpu-memory-utilization 0.8 --max-model-len 80k

cloudflared tunnel --url http://localhost:8123
cloudflared tunnel --url http://localhost:8124
cloudflared tunnel --url http://localhost:8125
```