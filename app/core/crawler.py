import os
import json
import time
import requests
from datetime import datetime
from ..configs.config import CrawlerConfig
from ..utils.logger import logger

class Crawler:
    """Searches SearXNG and stores unique links in a JSONL file (no DB)."""

    def __init__(self, crawler_config: CrawlerConfig):
        self.crawler_config = crawler_config
        self.searx_url = crawler_config.searxng_url
        self.delay = 1.0  # polite delay between requests

        # Path for JSONL
        self.jsonl_path = crawler_config.links_file_path

        # Ensure output directory exists
        jsonl_dir = os.path.dirname(self.jsonl_path)
        if jsonl_dir and not os.path.isdir(jsonl_dir):
            os.makedirs(jsonl_dir, exist_ok=True)

        # Load all already‐stored URLs into a set for deduplication
        self.seen_urls: set[str] = set()
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        url = entry.get("href")
                        if url:
                            self.seen_urls.add(url)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line.")

    def _make_searx_request(self, query: str, page: int = 1) -> list[dict]:
        """Make a single request to the SearXNG API and return the raw results list."""
        params = {
            "q": query,
            "format": "json",
            "language": self.crawler_config.language,
            "categories": "general",
            "pageno": page,
            "time_range": self.crawler_config.time_range,
        }
        try:
            resp = requests.get(self.searx_url, params=params, timeout=self.crawler_config.timeout)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []

    def search_and_store(self, query: str) -> int:
        logger.info("Searching for '%s'...", query)
        added = 0
        skipped = 0

        with open(self.jsonl_path, "a", encoding="utf-8") as jsonl_file:
            for page in range(1, self.crawler_config.pages + 1):
                logger.info("Processing page %s/%s", page, self.crawler_config.pages)
                results = self._make_searx_request(query, page)
                if not results:
                    logger.warning("No results found on page %s", page)
                    continue

                for hit in results:
                    url = hit.get("url")
                    if not url:
                        logger.warning("Skipping result with no URL")
                        continue

                    if url in self.seen_urls:
                        skipped += 1
                        continue

                    timestamp = datetime.utcnow().isoformat()
                    record = {
                        "title": hit.get("title", ""),
                        "href": url,
                        "content": hit.get("content", ""),
                        "stored_at": timestamp,
                        "original_query": query,
                        "page": page,
                        "engine": hit.get("engine", "unknown"),
                    }

                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                    os.fsync(jsonl_file.fileno())

                    self.seen_urls.add(url)
                    added += 1

                time.sleep(self.delay)

        logger.info("Done. Added %s new links. Skipped %s duplicates.", added, skipped)
        return added

    def search_and_store_batch(self, queries: list[str]) -> dict:
        logger.info("Starting batch search for %s queries...", len(queries))
        total_added = 0
        total_skipped = 0

        # We can append to the same JSONL file across all queries
        with open(self.jsonl_path, "a", encoding="utf-8") as jsonl_file:
            for i, query in enumerate(queries, start=1):
                logger.info("Processing query %s/%s: '%s'", i, len(queries), query)
                query_added = 0
                query_skipped = 0

                for page in range(1, self.crawler_config.pages + 1):
                    logger.info("Processing page %s/%s for query '%s'", page, self.crawler_config.pages, query)
                    results = self._make_searx_request(query, page)
                    if not results:
                        logger.warning("No results found on page %s", page)
                        continue

                    for hit in results:
                        url = hit.get("url")
                        if not url:
                            logger.warning("Skipping result with no URL")
                            continue

                        if url in self.seen_urls:
                            query_skipped += 1
                            continue

                        timestamp = datetime.utcnow().isoformat()
                        record = {
                            "title": hit.get("title", ""),
                            "href": url,
                            "content": hit.get("content", ""),
                            "stored_at": timestamp,
                            "original_query": query,
                            "page": page,
                            "engine": hit.get("engine", "unknown"),
                        }

                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        jsonl_file.flush()
                        os.fsync(jsonl_file.fileno())

                        self.seen_urls.add(url)
                        query_added += 1

                    time.sleep(self.delay)

                logger.info(
                    "Query '%s' completed. Added %s new links. Skipped %s duplicates.",
                    query,
                    query_added,
                    query_skipped,
                )
                total_added += query_added
                total_skipped += query_skipped

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
