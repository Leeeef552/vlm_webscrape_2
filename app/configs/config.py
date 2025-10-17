import yaml
import logging
from dataclasses import dataclass
from typing import Optional
from ..utils.logger import logger

@dataclass
class MainConfig:
    queries_file: str = "/home/leeeefun681/volume/eefun/webscraping2/vlm_webscrape/app/seed_data/test.txt"
    batch_size: int = 1
    tracker_file: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/processed_queries.txt"

@dataclass
class CrawlerConfig:
    language: str = "en"
    pages: int = 1
    safesearch: int =2
    time_range: str = "year"
    timeout: int = 25
    links_file_path: str = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/raw_links/global_links.jsonl"
    searxng_url: str = "http://localhost:3628/search"
    concurrency: int = 8
    validator_model_name: str = "google/gemma-3-12b-it"
    validator_base_url: str = "http://localhost:8124/v1"
    validator_workers: int = 16

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
    

def load_config(file_path):
    config_classes = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config_class_params = yaml.safe_load(f)
        
        if config_class_params is None:
            raise ValueError("Empty YAML file")
            
        config_classes["crawler"] = CrawlerConfig(**config_class_params["crawler"])
        config_classes["scraper"] = ScraperConfig(**config_class_params["scraper"])
        config_classes["main"] = MainConfig(**config_class_params["main"])

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