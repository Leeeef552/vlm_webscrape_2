import logging
from .configs.config import get_config
from .core.links_scraper import LinkScraper
from .core.content_process import ImageMetadataExtractor

CONFIG_FILEPATH = "Scraping/configs/config.json"


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
        f"{base_query} tutorial",
        f"{base_query} guide",
        f"{base_query} examples",
        f"best {base_query}",
        f"{base_query} tips"
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
    scraper = LinkScraper(config)
    
    # Process each transformed query
    total_results = 0
    for i, query in enumerate(transformed_queries, 1):
        logger.info(f"Processing query {i}/{len(transformed_queries)}: '{query}'")
        results_count = scraper.search_and_store(query)
        total_results += results_count
    
    logger.info(f"Scraping completed! Total new links found: {total_results}")


    # ------------------------------------------------------------------
    # Content Scraping
    # ------------------------------------------------------------------
    logger.info("Initialising Webcrawl...")
    with open





if __name__ == "__main__":
    main()
