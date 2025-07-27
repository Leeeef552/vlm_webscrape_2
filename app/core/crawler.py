import os
import json
import time
import requests
from datetime import datetime
from ..configs.config import CrawlerConfig
from ..utils.logger import logger

class Crawler:
    """Searches SearXNG and stores unique links in a master JSONL and in a per‐run JSONL."""

    def __init__(self, crawler_config: CrawlerConfig):
        self.crawler_config = crawler_config
        self.searx_url     = crawler_config.searxng_url
        # master file for persistence
        self.master_path   = crawler_config.links_file_path

        # ensure master directory exists
        master_dir = os.path.dirname(self.master_path)
        if master_dir and not os.path.isdir(master_dir):
            os.makedirs(master_dir, exist_ok=True)

        # load seen URLs for dedupe
        self.seen_urls: set[str] = set()
        if os.path.exists(self.master_path):
            with open(self.master_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        url = entry.get("href")
                        if url:
                            self.seen_urls.add(url)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line in master file.")

    def _make_searx_request(self, query: str, page: int = 1) -> list[dict]:
        params = {
            "q": query,
            "format": "json",
            "language": self.crawler_config.language,
            "categories": "general",
            "pageno": page,
            "time_range": self.crawler_config.time_range,
        }
        try:
            resp = requests.get(self.searx_url, params=params,
                                timeout=self.crawler_config.timeout)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as e:
            logger.error(f"SearX request failed: {e}")
            return []

    def _make_run_filename(self, prefix: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(self.master_path) or "."
        run_name = f"{prefix}_links_{ts}.jsonl"
        return os.path.join(base_dir, run_name)

    def search_and_store(self, query: str) -> str:
        logger.info("Searching for '%s'...", query)

        run_path = self._make_run_filename("run")
        added = skipped = 0

        # open both master (append) and run (write fresh)
        with open(self.master_path, "a", encoding="utf-8") as master_f, \
             open(run_path,        "w", encoding="utf-8") as run_f:

            for page in range(1, self.crawler_config.pages + 1):
                logger.info("Page %d/%d", page, self.crawler_config.pages)
                results = self._make_searx_request(query, page)
                if not results:
                    logger.warning("No results on page %d", page)
                    continue

                for hit in results:
                    url = hit.get("url")
                    if not url:
                        logger.warning("Skipping hit with no URL")
                        continue

                    if url in self.seen_urls:
                        skipped += 1
                        continue

                    record = {
                        "title":           hit.get("title", ""),
                        "href":            url,
                        "content":         hit.get("content", ""),
                        "stored_at":       datetime.utcnow().isoformat(),
                        "original_query":  query,
                        "page":            page,
                        "engine":          hit.get("engine", "unknown"),
                    }

                    line = json.dumps(record, ensure_ascii=False)
                    master_f.write(line + "\n")
                    master_f.flush()
                    os.fsync(master_f.fileno())

                    run_f.write(line + "\n")
                    run_f.flush()
                    os.fsync(run_f.fileno())

                    self.seen_urls.add(url)
                    added += 1

        logger.info("Done. %d new, %d skipped.", added, skipped)
        return run_path

    def search_and_store_batch(self, queries: list[str]) -> str:
        logger.info("Batch search for %d queries...", len(queries))

        run_path = self._make_run_filename("batch_run")
        total_added = total_skipped = 0

        with open(self.master_path, "a", encoding="utf-8") as master_f, \
             open(run_path,        "w", encoding="utf-8") as run_f:

            for i, query in enumerate(queries, start=1):
                logger.info("Query %d/%d: '%s'", i, len(queries), query)
                for page in range(1, self.crawler_config.pages + 1):
                    results = self._make_searx_request(query, page)
                    if not results:
                        logger.warning("No results on page %d for '%s'", page, query)
                        continue

                    for hit in results:
                        url = hit.get("url")
                        if not url:
                            logger.warning("Skipping hit with no URL")
                            continue

                        if url in self.seen_urls:
                            total_skipped += 1
                            continue

                        record = {
                            "title":           hit.get("title", ""),
                            "href":            url,
                            "content":         hit.get("content", ""),
                            "stored_at":       datetime.utcnow().isoformat(),
                            "original_query":  query,
                            "page":            page,
                            "engine":          hit.get("engine", "unknown"),
                        }

                        line = json.dumps(record, ensure_ascii=False)
                        master_f.write(line + "\n")
                        master_f.flush()
                        os.fsync(master_f.fileno())

                        run_f.write(line + "\n")
                        run_f.flush()
                        os.fsync(run_f.fileno())

                        self.seen_urls.add(url)
                        total_added += 1


        logger.info("Batch done. %d new, %d skipped.", total_added, total_skipped)
        return run_path
