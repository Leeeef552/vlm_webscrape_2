import asyncio
from pathlib import Path
import json
from datetime import datetime

from .configs.config import load_config
from .core.crawler import Crawler
from .core.scraper import Scraper
from .core.topic_extractor import TopicExtractor
from .core.query_expansion import QueryExpansion
from .utils.logger import logger
from .utils.utils import open_queries_txt

import time

# === Configuration Variables ===
# Path to the JSON config file
CONFIG_PATH = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/configs/config.yaml"
# Number of full pipeline iterations (search → scrape → topics → expand)
NUM_ITERATIONS = 5
# Initial root query for the first crawl
queries_path = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/seed_data/seed_queries.txt"
INITIAL_QUERY = open_queries_txt(queries_path)


async def main():
    start = time.time()
    # Load pipeline configuration
    logger.info(f"Loading configurations from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.error("Could not load config. Exiting...")
        return

    crawler_cfg = config["crawler"]
    scraper_cfg = config["scraper"]
    topic_cfg = config["topic_extractor"]
    expand_cfg = config["query_expansion"]

    # Initialize crawler
    crawler = Crawler(crawler_cfg)

    current_queries = INITIAL_QUERY

    for i in range(NUM_ITERATIONS):
        print("=" * 29)
        print(f"====    Iteration {i+1}/{NUM_ITERATIONS}    ====")
        print("=" * 29)

        # 1) WEB SEARCH -----------------------------------------------------
        if len(current_queries) == 1:
            q = current_queries[0]
            logger.info("Crawling web (general) for single query: %s", q)
            run_links_file = crawler.search_and_store(q)          # web only
        else:
            logger.info("Crawling web (general) for %d queries", len(current_queries))
            run_links_file = crawler.search_and_store_batch(current_queries)  # web only

        # 2) IMAGE SEARCH ---------------------------------------------------
        if len(current_queries) == 1:
            q = current_queries[0]
            logger.info("Crawling IMAGES for single query: %s", q)
            crawler.search_and_store_images(q)                    # images only
        else:
            logger.info("Crawling IMAGES for %d queries", len(current_queries))
            run_links_file = crawler.search_and_store_images_batch(current_queries, run_links_file)  # images only
        
        # 3) Scraping Content From Links -----------------------------------
        logger.info("Loading links for scraping...")
        scraper_cfg.links_file_path = run_links_file
        # links = load_links(scraper_cfg.links_file_path) # 

        logger.info(f"Scraping links with concurrency={scraper_cfg.concurrency}...")
        async with Scraper(scraper_cfg) as scraper:
            out = await scraper.extract_all_content(scraper_cfg.links_file_path)
            images = out["images"]
            markdowns = out["markdowns"]
            
        # 4) Save Scraped Outputs ------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = Path(scraper_cfg.images_dir) / f"images_metadata_{ts}.json"
        md_file = Path(scraper_cfg.markdown_dir) / f"text_markdown_{ts}.json"
        img_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(img_file, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)
        with open(md_file, "w", encoding="utf-8") as f:
            json.dump(markdowns, f, ensure_ascii=False, indent=2)

        # 5) Topic extraction ----------------------------------------------
        logger.info("Extracting topics from markdown output...")
        topic_cfg.data_file = str(md_file)
        extractor = TopicExtractor(topic_cfg)
        stats = extractor.extract_from_file()
        logger.info(f"Total unique entities: {stats['total_entities']}")
        for label, count in sorted(stats['counts_by_label'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label}: {count}")

        # 6) Query expansion -----------------------------------------------
        logger.info("Generating new queries via expansion...")
        expander = QueryExpansion(expand_cfg)
        current_queries = expander.get_queries(4)
        logger.info(f"Generated {len(current_queries)} queries")

    logger.info(f"Pipeline run complete. Total time taken: {time.time() - start}")

if __name__ == "__main__":
    asyncio.run(main())
