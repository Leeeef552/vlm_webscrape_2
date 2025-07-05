import json
import logging
from dataclasses import dataclass
from typing import Optional

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Configuration for the Link Scraper."""
    # defaults
    query: str = ""
    region: str = "wt-wt"
    safesearch: str = "Moderate"
    timelimit: Optional[str] = "1y"
    max_results: int = 50
    timeout: int = 3
    base_dir: str = "/home/leeeefun681/volume/eefun/webscraping/webscraping/storage/links"
    db_dir: str = "/home/leeeefun681/volume/eefun/webscraping/webscraping/storage/links/db"

def load_config_from_json(json_path: str = "Scraping/configs/config.json") -> Optional[ScraperConfig]:
    """Load configuration from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return ScraperConfig(**config_data)
    except FileNotFoundError:
        logger.error(f"Config file not found: {json_path}")
        logger.info("Creating a sample config.json file...")
        create_sample_config(json_path)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return None
    except TypeError as e:
        logger.error(f"Invalid configuration parameters: {e}")
        return None
    
def get_config(json_path) -> Optional[ScraperConfig]:
    """Return the scraper configuration from JSON file."""
    return load_config_from_json(json_path)

def create_sample_config(json_path: str = "config.json"):
    """Create a sample configuration JSON file."""
    sample_config = {
        "query": "python web scraping",
        "region": "wt-wt",
        "safesearch": "Off",
        "timelimit": None,
        "max_results": 50,
        "timeout": 3,
        "base_dir": "/home/leeeefun681/volume/eefun/webscraping/webscraping/storage/links",
        "db_dir": "/home/leeeefun681/volume/eefun/webscraping/webscraping/storage/links/db"
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample config file created: {json_path}")
    logger.info("Please edit the config.json file and run again.")
