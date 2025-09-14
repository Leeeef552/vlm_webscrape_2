import asyncio
import random
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
CONFIG_PATH = "/home/intern_volume/intern_volume/eefun/webscraping/scraping/vlm_webscrape/app/configs/config.yaml"
MAX_ITERATIONS = 20
BATCH_SIZE = 250
QUERIES_PATH = "/home/intern_volume/intern_volume/eefun/webscraping/scraping/vlm_webscrape/app/seed_data/seed_queries.txt"

async def main():
    start = time.time()
    logger.info(f"Loading configurations from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.error("Could not load config. Exiting...")
        return

    crawler_cfg = config["crawler"]
    scraper_cfg = config["scraper"]
    topic_cfg = config["topic_extractor"]
    expand_cfg = config["query_expansion"]

    crawler = Crawler(crawler_cfg)

    logger.info("Initializing Knowledge Base with seed entities...")
    db_dir = Path(topic_cfg.db_path)
    db_dir.mkdir(parents=True, exist_ok=True)
    
    seed_extractor = TopicExtractor(topic_cfg)
    seed_extractor._init_db()
    logger.info("Knowledge Base initialized successfully.")

    expander = QueryExpansion(expand_cfg)
    logger.info("QueryExpander initialized successfully.")

    # Load initial queries
    current_queries = open_queries_txt(QUERIES_PATH)
    logger.info(f"Loaded {len(current_queries)} initial queries.")

    iteration = 0

    while iteration < MAX_ITERATIONS and current_queries:
        iteration += 1
        print("=" * 29)
        print(f"====    Iteration {iteration}/{MAX_ITERATIONS}    ====")
        print("=" * 29)

        # Shuffle the current queries
        random.shuffle(current_queries)

        # Take a batch of up to BATCH_SIZE queries
        batch_queries = current_queries[:BATCH_SIZE]
        current_queries = current_queries[BATCH_SIZE:]  # Remove the batch from the list

        logger.info(f"Processing batch of {len(batch_queries)} queries...")

        # 1) WEB SEARCH ----------------------------------------------------
        logger.info("Crawling web (general) for %d queries", len(batch_queries))
        run_links_file = crawler.search_and_store_batch(batch_queries)

        # 2) IMAGE SEARCH --------------------------------------------------
        logger.info("Crawling IMAGES for %d queries", len(batch_queries))
        run_links_file = crawler.search_and_store_images_batch(batch_queries, run_links_file)

        # 3) Scraping Content From Links -----------------------------------
        logger.info("Loading links for scraping...")
        scraper_cfg.links_file_path = run_links_file
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
        expander.refresh_cache()  # Refresh cache with new data
        new_queries = expander.get_queries(4)
        current_queries.extend(new_queries)
        logger.info(f"Generated {len(new_queries)} new queries. Total queries now: {len(current_queries)}")


    total_seconds = time.time() - start
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    logger.info(f"Pipeline run complete. Total time taken: {hours}h {minutes}m {seconds:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())