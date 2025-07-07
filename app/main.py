import argparse
import asyncio
import logging
from .configs.config import load_config
from .core.crawler import Crawler
from .core.scraper import Scraper
from .utils.logger import logger
from .utils.utils import load_links

def parse_args():
    parser = argparse.ArgumentParser(description="Web Crawler and Scraper Pipeline")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="app/configs/config.json",
        help="Path to configuration JSON file"
    )
    return parser.parse_args()

def transform_query(base_query: str) -> list[str]:
    return [
        base_query,
        f"{base_query} foods",
        f"{base_query} locations",
        f"{base_query} landmarks",
        f"best things to do {base_query}",
        f"{base_query} travel tips"
    ]

async def main(config_path):
    logger.info(f"Getting configurations from {config_path}...")
    config = load_config(config_path)
    if config is None:
        logger.error("Could not load config. Exiting...")
        return

    crawler_config = config["crawler"]
    scraper_config = config["scraper"]

    logger.info("Transforming Query...")
    transformed_queries = transform_query(crawler_config.query)
    logger.info(f"Generated {len(transformed_queries)} query variations")

    logger.info("Initialising Crawler...")
    crawler = Crawler(crawler_config)
    logger.info(f"Processing query: '{crawler_config.query}'")
    results_count = crawler.search_and_store_batch(transformed_queries)
    logger.info(f"Crawl completed! Total new links found: {results_count}")

    logger.info("Scraping Links...")
    links = load_links(scraper_config.links_file_path)
    async with Scraper(scraper_config) as scraper:
        results = await scraper.extract_all_content(links)
        images = results["images"]
        markdown = results["markdowns"]
        # Optionally save results here

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.config))
