import os
import json
import time
import requests
from datetime import datetime
from ..configs.config import CrawlerConfig
from ..utils.logger import logger
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List


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
        self.lock = threading.Lock()
        self.concurrency = self.crawler_config.concurrency

    def _make_run_filename(self, prefix: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(self.master_path) or "."
        run_name = f"{prefix}_links_{ts}.jsonl"
        return os.path.join(base_dir, run_name)

    def _get_run_path(self, prefix: str, run_path: Optional[str] = None) -> str:
        """Helper to determine run path - either create new or use provided."""
        if run_path is not None:
            # Ensure directory exists for provided path
            run_dir = os.path.dirname(run_path)
            if run_dir and not os.path.isdir(run_dir):
                os.makedirs(run_dir, exist_ok=True)
            return run_path
        return self._make_run_filename(prefix)

    ###################################
    ##        search helpers        ##
    ###################################
    def _make_searx_request(self, query: str, page: int = 1) -> list[dict]:
        if not query or not query.strip():
            logger.warning("Empty query supplied to SearXNG request.")
            return []

        # Ensure correct endpoint (some instances expect /search)
        url = self.searx_url
        if not url.rstrip("/").endswith("/search"):
            url = url.rstrip("/") + "/search"

        params = {
            "q": query,
            "format": "json",
            "language": self.crawler_config.language,
            "categories": "general",
            "pageno": page,
            "time_range": self.crawler_config.time_range,
        }

        logger.debug("SearXNG request: %s params: %s", url, params)
        try:
            resp = requests.get(url, params=params, timeout=self.crawler_config.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("SearX request failed (network/HTTP): %s", e)
            return []

        # Try to parse JSON
        try:
            data = resp.json()
        except ValueError:
            logger.error("Failed to decode JSON from SearXNG. Response text: %s", resp.text[:1000])
            return []

        # Sanity-check structure
        if "results" not in data:
            logger.warning("'results' key missing in SearXNG response. Full response: %s", data)
            return []

        results = data["results"]
        if not results:
            logger.info("Empty results for query=%r page=%d. Response snippet: %s", query, page, json.dumps(data)[:1000])
        return results

    def _make_searx_image_request(self, query: str, page: int = 1) -> list[dict]:
        """Single image-search request to SearXNG."""
        if not query or not query.strip():
            logger.warning("Empty query supplied to SearXNG image request.")
            return []

        url = self.searx_url.rstrip("/")
        if not url.endswith("/search"):
            url += "/search"

        params = {
            "q": query,
            "format": "json",
            "language": self.crawler_config.language,
            "categories": "images",          # <── images only
            "pageno": page,
            "time_range": self.crawler_config.time_range,
        }

        logger.debug("SearXNG image request: %s params: %s", url, params)
        try:
            resp = requests.get(url, params=params, timeout=self.crawler_config.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("SearX image request failed (network/HTTP): %s", e)
            return []

        try:
            data = resp.json()
        except ValueError:
            logger.error("Failed to decode JSON from SearXNG (images). Response text: %s",
                         resp.text[:1000])
            return []

        if "results" not in data:
            logger.warning("'results' key missing in SearXNG image response. Full response: %s", data)
            return []

        results = data["results"]
        if not results:
            logger.info("Empty image results for query=%r page=%d.", query, page)
        return results

    # def search_and_store(self, query: str) -> str:
    #     logger.info("Searching for '%s'...", query)

    #     run_path = self._make_run_filename("run")
    #     added = skipped = 0

    #     # open both master (append) and run (write fresh)
    #     with open(self.master_path, "a", encoding="utf-8") as master_f, \
    #          open(run_path,        "w", encoding="utf-8") as run_f:

    #         for page in range(1, self.crawler_config.pages + 1):
    #             logger.info("Page %d/%d", page, self.crawler_config.pages)
    #             results = self._make_searx_request(query, page)
    #             if not results:
    #                 logger.warning("No results on page %d", page)
    #                 continue

    #             for hit in results:
    #                 url = hit.get("url")
    #                 if not url:
    #                     logger.warning("Skipping hit with no URL")
    #                     continue

    #                 if url in self.seen_urls:
    #                     skipped += 1
    #                     continue

    #                 record = {
    #                     "title":           hit.get("title", ""),
    #                     "href":            url,
    #                     "content":         hit.get("content", ""),
    #                     "stored_at":       datetime.utcnow().isoformat(),
    #                     "original_query":  query,
    #                     "page":            page,
    #                     "engine":          hit.get("engine", "unknown"),
    #                 }

    #                 line = json.dumps(record, ensure_ascii=False)
    #                 master_f.write(line + "\n")
    #                 master_f.flush()
    #                 os.fsync(master_f.fileno())

    #                 run_f.write(line + "\n")
    #                 run_f.flush()
    #                 os.fsync(run_f.fileno())

    #                 self.seen_urls.add(url)
    #                 added += 1

    #     logger.info("Done. %d new, %d skipped.", added, skipped)
    #     return run_path

    # def search_and_store_batch(self, queries: list[str], run_path: Optional[str] = None) -> str:
    #     """Batch search with optional run path reuse."""
    #     logger.info("Batch search for %d queries...", len(queries))
    #     run_path = self._get_run_path("batch_run", run_path)  # Use provided path or create new
    #     total_added = total_skipped = 0
        
    #     with open(self.master_path, "a", encoding="utf-8") as master_f, \
    #          open(run_path, "w", encoding="utf-8") as run_f:  # Always overwrite if exists
            
    #         for i, query in enumerate(queries, start=1):
    #             logger.info("Query %d/%d: '%s'", i, len(queries), query)
    #             for page in range(1, self.crawler_config.pages + 1):
    #                 results = self._make_searx_request(query, page)
    #                 if not results:
    #                     logger.warning("No results on page %d for '%s'", page, query)
    #                     continue
    #                 for hit in results:
    #                     url = hit.get("url")
    #                     if not url:
    #                         logger.warning("Skipping hit with no URL")
    #                         continue
    #                     if url in self.seen_urls:
    #                         total_skipped += 1
    #                         continue
    #                     record = {
    #                         "title":           hit.get("title", ""),
    #                         "href":            url,
    #                         "content":         hit.get("content", ""),
    #                         "stored_at":       datetime.utcnow().isoformat(),
    #                         "original_query":  query,
    #                         "page":            page,
    #                         "engine":          hit.get("engine", "unknown"),
    #                     }
    #                     line = json.dumps(record, ensure_ascii=False)
    #                     master_f.write(line + "\n")
    #                     master_f.flush()
    #                     os.fsync(master_f.fileno())
    #                     run_f.write(line + "\n")
    #                     run_f.flush()
    #                     os.fsync(run_f.fileno())
    #                     self.seen_urls.add(url)
    #                     total_added += 1

    #     logger.info("Batch done. %d new, %d skipped.", total_added, total_skipped)
    #     return run_path

    # def search_and_store_images(self, query: str) -> str:
    #     """Search for images only (single query)."""
    #     logger.info("Searching IMAGES for '%s'...", query)

    #     run_path = self._make_run_filename("run_images")
    #     added = skipped = 0

    #     with open(self.master_path, "a", encoding="utf-8") as master_f, \
    #          open(run_path, "w", encoding="utf-8") as run_f:

    #         for page in range(1, self.crawler_config.pages + 1):
    #             logger.info("Image page %d/%d", page, self.crawler_config.pages)
    #             results = self._make_searx_image_request(query, page)
    #             if not results:
    #                 logger.warning("No image results on page %d", page)
    #                 continue

    #             for hit in results:
    #                 url = hit.get("url")
    #                 if not url:
    #                     logger.warning("Skipping image hit with no URL")
    #                     continue

    #                 if url in self.seen_urls:
    #                     skipped += 1
    #                     continue

    #                 record = {
    #                     "title":          hit.get("title", ""),
    #                     "href":           url,
    #                     "content":        hit.get("content", ""),
    #                     "stored_at":      datetime.utcnow().isoformat(),
    #                     "original_query": query,
    #                     "page":           page,
    #                     "engine":         hit.get("engine", "unknown"),
    #                     "type":           "image",          # explicit flag
    #                     "img_src":        hit.get("img_src"),
    #                     "thumbnail_src":  hit.get("thumbnail_src"),
    #                 }

    #                 line = json.dumps(record, ensure_ascii=False)
    #                 master_f.write(line + "\n")
    #                 master_f.flush()
    #                 os.fsync(master_f.fileno())

    #                 run_f.write(line + "\n")
    #                 run_f.flush()
    #                 os.fsync(run_f.fileno())

    #                 self.seen_urls.add(url)
    #                 added += 1

    #     logger.info("Image search done. %d new, %d skipped.", added, skipped)
    #     return run_path
    
    # def search_and_store_images_batch(self, queries: list[str], run_path: Optional[str] = None) -> str:

    #     """Batch image search with optional run path reuse."""
    #     logger.info("Batch image search for %d queries...", len(queries))
    #     run_path = self._get_run_path("batch_images", run_path)  # Use provided path or create new
    #     total_added = total_skipped = 0
        
    #     with open(self.master_path, "a", encoding="utf-8") as master_f, \
    #          open(run_path, "w", encoding="utf-8") as run_f:  # Always overwrite if exists
            
    #         for i, query in enumerate(queries, start=1):
    #             logger.info("Image query %d/%d: '%s'", i, len(queries), query)
    #             for page in range(1, self.crawler_config.pages + 1):
    #                 results = self._make_searx_image_request(query, page)
    #                 if not results:
    #                     logger.warning("No image results on page %d for '%s'", page, query)
    #                     continue
    #                 for hit in results:
    #                     url = hit.get("url")
    #                     if not url:
    #                         logger.warning("Skipping image hit with no URL")
    #                         continue
    #                     if url in self.seen_urls:
    #                         total_skipped += 1
    #                         continue
    #                     record = {
    #                         "title":          hit.get("title", ""),
    #                         "href":           url,
    #                         "content":        hit.get("content", ""),
    #                         "stored_at":      datetime.utcnow().isoformat(),
    #                         "original_query": query,
    #                         "page":           page,
    #                         "engine":         hit.get("engine", "unknown"),
    #                         "type":           "image",
    #                         "img_src":        hit.get("img_src"),
    #                         "thumbnail_src":  hit.get("thumbnail_src"),
    #                     }
    #                     line = json.dumps(record, ensure_ascii=False)
    #                     master_f.write(line + "\n")
    #                     master_f.flush()
    #                     os.fsync(master_f.fileno())
    #                     run_f.write(line + "\n")
    #                     run_f.flush()
    #                     os.fsync(run_f.fileno())
    #                     self.seen_urls.add(url)
    #                     total_added += 1

    #     logger.info("Batch image search done. %d new, %d skipped.", total_added, total_skipped)
    #     return run_path
    
    def _process_single_query(self, query: str, master_f, run_f, is_image=False) -> tuple[int, int]:
        added = skipped = 0
        make_request = self._make_searx_image_request if is_image else self._make_searx_request

        for page in range(1, self.crawler_config.pages + 1):
            results = make_request(query, page)
            if not results:
                continue
            for hit in results:
                url = hit.get("url")
                if not url:
                    continue
                with self.lock:
                    if url in self.seen_urls:
                        skipped += 1
                        continue
                    self.seen_urls.add(url)

                record = {
                    "title": hit.get("title", ""),
                    "href": url,
                    "content": hit.get("content", ""),
                    "stored_at": datetime.utcnow().isoformat(),
                    "original_query": query,
                    "page": page,
                    "engine": hit.get("engine", "unknown"),
                }
                if is_image:
                    record.update({
                        "type": "image",
                        "img_src": hit.get("img_src"),
                        "thumbnail_src": hit.get("thumbnail_src"),
                    })

                line = json.dumps(record, ensure_ascii=False)
                with self.lock:
                    master_f.write(line + "\n")
                    master_f.flush()
                    os.fsync(master_f.fileno())
                    run_f.write(line + "\n")
                    run_f.flush()
                    os.fsync(run_f.fileno())
                added += 1
        return added, skipped

    def search_and_store_batch(self, queries: List[str], run_path: Optional[str] = None) -> str:
        logger.info("Starting concurrent batch search for %d queries...", len(queries))
        run_path = self._get_run_path("batch_run", run_path)
        total_added = total_skipped = 0

        with open(self.master_path, "a", encoding="utf-8") as master_f, \
             open(run_path, "w", encoding="utf-8") as run_f:

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                future_to_query = {
                    executor.submit(self._process_single_query, q, master_f, run_f, False): q
                    for q in queries
                }

                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        added, skipped = future.result()
                        total_added += added
                        total_skipped += skipped
                        logger.debug("Query '%s' done: +%d new, %d skipped", query, added, skipped)
                    except Exception as exc:
                        logger.error("Query '%s' generated an exception: %s", query, exc)

        logger.info("Batch done. %d new, %d skipped.", total_added, total_skipped)
        return run_path

    def search_and_store_images_batch(self, queries: List[str], run_path: Optional[str] = None) -> str:
        logger.info("Starting concurrent batch image search for %d queries...", len(queries))
        run_path = self._get_run_path("batch_images", run_path)
        total_added = total_skipped = 0

        with open(self.master_path, "a", encoding="utf-8") as master_f, \
             open(run_path, "w", encoding="utf-8") as run_f:

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                future_to_query = {
                    executor.submit(self._process_single_query, q, master_f, run_f, True): q
                    for q in queries
                }

                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        added, skipped = future.result()
                        total_added += added
                        total_skipped += skipped
                        logger.debug("Image query '%s' done: +%d new, %d skipped", query, added, skipped)
                    except Exception as exc:
                        logger.error("Image query '%s' generated an exception: %s", query, exc)

        logger.info("Batch image search done. %d new, %d skipped.", total_added, total_skipped)
        return run_path