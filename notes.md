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


```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-12b-it --gpu-memory-utilization 0.85 --port 8124 --max-model-len 16k

CUDA_VISIBLE_DEVICES=2 vllm serve unsloth/Llama-3.2-3B-Instruct --gpu-memory-utilization 0.5 --port 8125 --max-model-len 16k

CUDA_VISIBLE_DEVICES=1 vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --gpu-memory-utilization 0.85 --port 8124 --max-model-len 16k
```

## Query Expansion


## Current Limitations:
1. Too many request on searxng

2. need to parameter tune for the threshold for gliner, etc

3. 

