import logging
from .configs.config import get_config
from .core.search_engine import Search_Engine
from .core.scrape_content import ImageMetadataExtractor

CONFIG_FILEPATH = "app/configs/config.json"


# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def transform_query(base_query: str) -> list[str]:
    """
    Transform the base query into multiple search variations.
    This is where your query transformer logic would go.
    """
    # Example query transformations - replace with your actual logic
    transformations = [
        base_query,
        f"{base_query} foods",
        f"{base_query} locations",
        f"{base_query} landmarks",
        f"best things to do {base_query} ",
        f"{base_query} travel tips"
    ]
    return transformations

def main():
    """Main orchestration function."""
    logger.info("Starting Webcrawl...")
    
    # Load configuration
    logger.info("Getting configurations from config file...")
    config = get_config(CONFIG_FILEPATH)
    if not config:
        logger.error("Failed to load configuration.")
        return
    
    logger.info(f"Base query: {config.query}")
    
    # ------------------------------------------------------------------
    # Query Transform
    # ------------------------------------------------------------------

    logger.info("Transforming query...")
    transformed_queries = transform_query(config.query)
    logger.info(f"Generated {len(transformed_queries)} query variations")


    # ------------------------------------------------------------------
    # DuckDuckGo Search
    # ------------------------------------------------------------------
    
    # Initialize scraper
    logger.info("Initialising Search...")
    scraper = Search_Engine(config)
    
    # Process each transformed query
    logger.info(f"Processing query: '{config.query}'")
    results_count = scraper.search_and_store_batch(transformed_queries)
    
    logger.info(f"Scraping completed! Total new links found: {results_count}")

if __name__ == "__main__":
    main()
