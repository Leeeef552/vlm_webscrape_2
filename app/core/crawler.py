import os
import shelve
import json
import time
import requests
from datetime import datetime
import tldextract

# ScraperConfig class
from ..configs.config import CrawlerConfig
from ..utils.logger import logger


class Crawler:
    """Searches SearXNG and stores unique links in a shelf + JSONL file."""

    def __init__(self, scraper_config: CrawlerConfig):
        self.scraper_config = scraper_config
        self.searx_url = self.scraper_config.searxng_url

        # Fixed delay between requests to avoid rate limiting
        self.delay = 1.0  # seconds

        # Paths for JSONL & Shelf
        self.jsonl_path = self.scraper_config.links_file_path
        self.shelf_path = self.scraper_config.shelf_path

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _make_searx_request(self, query: str, page: int = 1):
        """Make a single request to the SearXNG API and return the raw results list."""
        params = {
            "q": query,
            "format": "json",
            "language": self.scraper_config.language,
            "categories": "general",  # Fixed parameter
            "pageno": page,
            "time_range": self.scraper_config.time_range,
        }

        try:
            resp = requests.get(
                self.searx_url, params=params, timeout=self.scraper_config.timeout
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search_and_store(self, query: str):
        """Search for *query* and persist unique results.

        Returns
        -------
        int
            Number of new links added to the shelf/JSONL.
        """
        logger.info("Searching for '%s'...", query)
        count = 0
        skipped = 0

        with shelve.open(self.shelf_path) as shelf, open(
            self.jsonl_path, "a", encoding="utf-8"
        ) as jsonl_file:
            # Iterate through pages
            for page in range(1, self.scraper_config.pages + 1):
                logger.info(
                    "Processing page %s/%s", page, self.scraper_config.pages
                )

                results = self._make_searx_request(query, page)
                if not results:
                    logger.warning("No results found for page %s", page)
                    continue

                for hit in results:
                    url = hit.get("url")
                    if not url:
                        logger.warning("Skipping result with no URL")
                        continue

                    # Check for exact URL match (dedup)
                    if url in shelf:
                        skipped += 1
                        continue

                    timestamp = datetime.utcnow().isoformat()

                    # Result object similar to DuckDuckGo format
                    result = {
                        "title": hit.get("title", ""),
                        "href": url,
                        "content": hit.get("content", ""),
                        "stored_at": timestamp,
                        "original_query": query,
                        "page": page,
                        "engine": hit.get("engine", "unknown"),
                    }

                    # Persist full result to JSONL (append‑only)
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                    os.fsync(jsonl_file.fileno())

                    # Mark URL in shelf to prevent future duplicates (minimal metadata)
                    shelf[url] = {
                        "stored_at": timestamp,
                        "title": hit.get("title", ""),
                        "original_query": query,
                    }
                    shelf.sync()

                    count += 1

                # Be polite to the API
                time.sleep(self.delay)

        logger.info("Done. Added %s new links. Skipped %s duplicates.", count, skipped)
        return count

    def search_and_store_batch(self, queries: list[str]):
        """Search and store links for multiple queries.

        Returns
        -------
        dict
            Aggregate stats for the batch run.
        """
        logger.info("Starting batch search for %s queries...", len(queries))

        total_added = 0
        total_skipped = 0

        with shelve.open(self.shelf_path) as shelf, open(
            self.jsonl_path, "a", encoding="utf-8"
        ) as jsonl_file:
            for i, query in enumerate(queries, 1):
                logger.info("Processing query %s/%s: '%s'", i, len(queries), query)

                query_added = 0
                query_skipped = 0

                try:
                    # Iterate through pages for this query
                    for page in range(1, self.scraper_config.pages + 1):
                        logger.info(
                            "Processing page %s/%s for query '%s'",
                            page,
                            self.scraper_config.pages,
                            query,
                        )

                        results = self._make_searx_request(query, page)
                        if not results:
                            logger.warning("No results found for page %s", page)
                            continue

                        for hit in results:
                            url = hit.get("url")
                            if not url:
                                logger.warning("Skipping result with no URL")
                                continue

                            # Dedup by exact URL
                            if url in shelf:
                                query_skipped += 1
                                continue

                            timestamp = datetime.utcnow().isoformat()

                            result = {
                                "title": hit.get("title", ""),
                                "href": url,
                                "content": hit.get("content", ""),
                                "stored_at": timestamp,
                                "original_query": query,
                                "page": page,
                                "engine": hit.get("engine", "unknown"),
                            }

                            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            os.fsync(jsonl_file.fileno())

                            shelf[url] = {
                                "stored_at": timestamp,
                                "title": hit.get("title", ""),
                                "original_query": query,
                            }
                            shelf.sync()

                            query_added += 1

                        time.sleep(self.delay)

                    logger.info(
                        "Query '%s' completed. Added %s new links. Skipped %s duplicates.",
                        query,
                        query_added,
                        query_skipped,
                    )

                except Exception as e:
                    logger.error("Failed to process query '%s': %s", query, e)
                    continue

                total_added += query_added
                total_skipped += query_skipped

                # Pause between queries to be respectful
                if i < len(queries):
                    time.sleep(self.delay)

        logger.info(
            "Batch search completed. Total added: %s new links. Total skipped: %s duplicates.",
            total_added,
            total_skipped,
        )
        return {
            "total_added": total_added,
            "total_skipped": total_skipped,
            "queries_processed": len(queries),
        }
