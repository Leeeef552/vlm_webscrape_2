import json
import logging
from dataclasses import dataclass
from typing import Optional
from ..utils.logger import logger


@dataclass
class CrawlerConfig:
    query: str = ""
    language: str = "en"
    pages: int = 1
    time_range: str = "year"
    timeout: int = 4
    links_file_path: str = "app/storage/raw_links"
    shelf_path: str = "app/storage/raw_links/db_link_hashing"


@dataclass
class ScraperConfig:
    concurrency: int = 4
    links_file_path: str = "app/storage/raw_links/links.jsonl"
    images_outfile: str = "app/storage/images_metadata/images_metadata.json"
    markdown_outfile: str = "app/storage/images_metadata/text_markdown.json"


def load_config(file_path):
    config_classes = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config_class_params = json.load(f)
        config_classes["crawler"] = CrawlerConfig(**config_class_params["crawler"])
        config_classes["scraper"]  = ScraperConfig(**config_class_params["scraper"])
        return config_classes

    except FileNotFoundError as e:
        logger.error(f"{e}: Filepath for `config.json` has errors: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"{e}: Invalid JSON in `config.json` file: {file_path}")
        return None
    except TypeError as e:
        logger.error(f"{e}: Invalid configuration parameters in config.json: {file_path}")
        return None
