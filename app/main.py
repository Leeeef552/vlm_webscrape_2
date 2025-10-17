import asyncio
import random
from pathlib import Path
import json
from datetime import datetime
from .configs.config import load_config
from .core.crawler import Crawler
from .core.scraper import Scraper
from .utils.logger import logger
from .utils.utils import open_queries_txt
import time
import os

# === Configuration Variables ===
CONFIG_PATH = "/home/leeeefun681/volume/eefun/webscraping2/vlm_webscrape/app/configs/config.yaml"

async def main():
    start = time.time()
    logger.info(f"Loading configurations from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.error("Could not load config. Exiting...")
        return

    # main configurations
    main_cfg = config["main"]
    BATCH_SIZE = main_cfg.batch_size
    QUERIES_PATH = main_cfg.queries_file
    TRACKER_FILE = main_cfg.tracker_file

    # crawler and scraper configs
    crawler_cfg = config["crawler"]
    scraper_cfg = config["scraper"]

    crawler = Crawler(crawler_cfg)
    logger.info("Crawler initialized successfully.")

    # === Load seed queries ===
    all_queries = open_queries_txt(QUERIES_PATH)
    logger.info(f"Loaded {len(all_queries)} total seed queries.")

    # === Load tracker and filter processed queries ===
    processed_queries = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            processed_queries = set(line.strip() for line in f if line.strip())

    current_queries = [q for q in all_queries if q not in processed_queries]
    logger.info(f"{len(processed_queries)} queries already processed, {len(current_queries)} remaining.")

    iteration = 0
    total_iterations = (len(current_queries) + BATCH_SIZE - 1) // BATCH_SIZE

    while current_queries:
        iteration += 1
        print("=" * 29)
        print(f"====    Iteration {iteration}/{total_iterations}    ====")
        print("=" * 29)

        # Shuffle remaining queries
        random.shuffle(current_queries)

        # Take one batch
        batch_queries = current_queries[:BATCH_SIZE]
        current_queries = current_queries[BATCH_SIZE:]

        logger.info(f"Processing batch of {len(batch_queries)} queries...")

        # ---- 1) Web search ----
        logger.info("Crawling general web for %d queries", len(batch_queries))
        run_links_file = crawler.search_and_store_batch(batch_queries)

        # ---- 2) Image search ----
        logger.info("Crawling images for %d queries", len(batch_queries))
        run_links_file = crawler.search_and_store_images_batch(batch_queries, run_links_file)

        # ---- 3) Scraping phase ----
        logger.info("Scraping extracted links...")
        scraper_cfg.links_file_path = run_links_file
        async with Scraper(scraper_cfg) as scraper:
            out = await scraper.extract_all_content(scraper_cfg.links_file_path)
            images = out["images"]
            markdowns = out["markdowns"]

        # ---- 4) Save scraped outputs ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = Path(scraper_cfg.images_dir) / f"images_metadata_{ts}.json"
        md_file = Path(scraper_cfg.markdown_dir) / f"text_markdown_{ts}.json"
        img_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(img_file, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)
        with open(md_file, "w", encoding="utf-8") as f:
            json.dump(markdowns, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(markdowns)} markdowns and {len(images)} image metadata records.")

        # ---- 5) Update tracker file ----
        with open(TRACKER_FILE, "a", encoding="utf-8") as f:
            for q in batch_queries:
                f.write(q + "\n")

        logger.info(f"Appended {len(batch_queries)} processed queries to tracker file.")
        logger.info(f"{len(current_queries)} queries remaining.")

    # === Run complete ===
    total_seconds = time.time() - start
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    logger.info(f"Run complete. Total time: {hours}h {minutes}m {seconds:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
