import yaml
import logging
from dataclasses import dataclass
from typing import Optional
from ..utils.logger import logger

@dataclass
class CrawlerConfig:
    language: str = "en"
    pages: int = 1
    time_range: str = "year"
    timeout: int = 25
    links_file_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/raw_links/global_links.jsonl"
    searxng_url: str = "http://localhost:3628/search"
    concurrency: int = 8


@dataclass
class ScraperConfig:
    # -- configs -- #
    concurrency: int = 4
    # -- file path and data --- #
    links_file_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/raw_links/links.jsonl"
    images_dir: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/images_metadata"
    markdown_dir: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/text_data"
    pdf_dir: str =  "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/pdf_data"
    # -- models -- # 
    validator_model_name: str = "google/gemma-3-12b-it"
    validator_base_url: str = "http://localhost:8124/v1"


@dataclass
class TopicExtractorConfig:
    # -- configs -- #
    concurrency: int = 32
    fuzzy_threshold: int = 85
    semantic_threshold: float = 0.85
    # -- file path and data --- #
    data_file: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/text_data"
    labels_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/seed_data/_entity_labels.jsonl"
    abbrev_map_path : str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/seed_data/_lexical_labels.json"
    seed_entities_file : str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/seed_data/classified_entities_filtered_sampled_500.jsonl"
    output_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/entities/extracted_entities.json"
    db_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/entities"
    # -- models -- # 
    embedding_model: str = "google/embeddinggemma-300m"
    ner_base_url: str = "https://jefferson-authentic-technique-researchers.trycloudflare.com/v1"
    ner_model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    cleaner_model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    cleaner_base_url: str = "https://emperor-concerning-kurt-hawk.trycloudflare.com/v1"
    validator_model_name: str = "google/gemma-3-12b-it"
    validator_base_url: str = "https://pensions-void-floor-knew.trycloudflare.com/v1"


@dataclass
class QueryExpansionConfig:
    # -- configs -- #
    num_queries_per_entity: int = 6
    num_queries_per_labels: int = 6
    n_labels: int = 5                       # number of labels to extract (should be between 1 to max number of labels specified to the gliner model)
    n_entities: float = 0.05                # 1% of graph at most
    entities_cap: int = 128                 # capped at 128 entities extracted    
    # -- file path and data --- #
    db_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/entities"
    queries_file_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/queries.text"
    # -- models -- # 
    base_url: str = "http://localhost:8124/v1"
    model_name: str = "google/gemma-3-12b-it"
    

def load_config(file_path):
    config_classes = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config_class_params = yaml.safe_load(f)
        
        if config_class_params is None:
            raise ValueError("Empty YAML file")
            
        config_classes["crawler"] = CrawlerConfig(**config_class_params["crawler"])
        config_classes["scraper"] = ScraperConfig(**config_class_params["scraper"])
        config_classes["topic_extractor"] = TopicExtractorConfig(**config_class_params["topic_extractor"])
        config_classes["query_expansion"] = QueryExpansionConfig(**config_class_params["query_expansion"])
        
        return config_classes

    except FileNotFoundError as e:
        logger.error(f"{e}: Filepath for `config.yaml` has errors: {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"{e}: Invalid YAML in `config.yaml` file: {file_path}")
        return None
    except TypeError as e:
        logger.error(f"{e}: Invalid configuration parameters in config.yaml: {file_path}")
        return None
    except KeyError as e:
        logger.error(f"{e}: Missing key in config.yaml file: {file_path}")
        return None