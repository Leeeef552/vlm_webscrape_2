import os
import shelve
import json
import time
import requests
from datetime import datetime
import tldextract
# ScraperConfig class
from ..configs.config import CrawlerConfig



### TO DO: SET UP LOGGER 

class Crawler:
    def __init__(self, scraper_config: CrawlerConfig):
        self.scraper_config = scraper_config
        self.searx_url = "http://localhost:8124/search"  # Fixed SearXNG URL
        
        # Ensure folders exist

        # Fixed delay between requests to avoid rate limiting
        self.delay = 1.0  # seconds

        # Paths for JSONL & Shelf
        self.jsonl_path = self.scraper_config.links_file_path
        self.shelf_path = self.scraper_config.shelf_path


    def _make_searx_request(self, query: str, page: int = 1):
        """Make a single request to SearXNG API."""
        params = {
            "q": query,
            "format": "json",
            "language": self.scraper_config.language,
            "categories": "general",  # Fixed parameter
            "pageno": page,
            "time_range": self.scraper_config.time_range
        }
        
        try:
            resp = requests.get(self.searx_url, params=params, timeout=self.scraper_config.timeout)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return []


    def search_and_store(self, query: str):
        """Search and store links for a specific query, with gentle rate-limiting and progress save."""
        print(f"[INFO] Searching for '{query}'...")
        count = 0
        skipped = 0

        with shelve.open(self.shelf_path) as shelf, open(self.jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            # Iterate through pages
            for page in range(1, self.scraper_config.pages + 1):
                print(f"[INFO] Processing page {page}/{self.scraper_config.pages}")
                
                results = self._make_searx_request(query, page)
                if not results:
                    print(f"[WARNING] No results found for page {page}")
                    continue

                for hit in results:
                    url = hit.get('url')
                    if not url:
                        print("[WARNING] Skipping result with no URL")
                        continue

                    # Check for exact URL match
                    if url in shelf:
                        skipped += 1
                        continue

                    # Add a timestamp and metadata
                    timestamp = datetime.utcnow().isoformat()
                    
                    # Create a result object similar to DuckDuckGo format
                    result = {
                        'title': hit.get('title', ''),
                        'href': url,
                        'content': hit.get('content', ''),
                        'stored_at': timestamp,
                        'original_query': query,
                        'page': page,
                        'engine': hit.get('engine', 'unknown')
                    }

                    # Write the full SearXNG result to JSONL
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
                    os.fsync(jsonl_file.fileno())

                    # Store minimal data in shelf to avoid duplicates
                    shelf[url] = {
                        'stored_at': timestamp,
                        'title': hit.get('title', ''),
                        'original_query': query
                    }
                    shelf.sync()

                    count += 1

                # Fixed delay between pages to avoid overwhelming the API
                time.sleep(self.delay)

        print(f"[INFO] Done. Added {count} new links. Skipped {skipped} duplicates.")
        return count


    def search_and_store_batch(self, queries: list[str]):
        """Search and store links for multiple queries."""
        print(f"[INFO] Starting batch search for {len(queries)} queries...")
        
        total_added = 0
        total_skipped = 0
        
        with shelve.open(self.shelf_path) as shelf, open(self.jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            for i, query in enumerate(queries, 1):
                print(f"[INFO] Processing query {i}/{len(queries)}: '{query}'")
                
                query_added = 0
                query_skipped = 0
                
                try:
                    # Iterate through pages for this query
                    for page in range(1, self.scraper_config.pages + 1):
                        print(f"[INFO] Processing page {page}/{self.scraper_config.pages} for query '{query}'")
                        
                        results = self._make_searx_request(query, page)
                        if not results:
                            print(f"[WARNING] No results found for page {page}")
                            continue

                        for hit in results:
                            url = hit.get('url')
                            if not url:
                                print("[WARNING] Skipping result with no URL")
                                continue

                            # Check for exact URL match
                            if url in shelf:
                                query_skipped += 1
                                continue

                            # Add a timestamp and metadata
                            timestamp = datetime.utcnow().isoformat()
                            
                            # Create a result object similar to DuckDuckGo format
                            result = {
                                'title': hit.get('title', ''),
                                'href': url,
                                'content': hit.get('content', ''),
                                'stored_at': timestamp,
                                'original_query': query,
                                'page': page,
                                'engine': hit.get('engine', 'unknown')
                            }

                            # Write the full SearXNG result to JSONL
                            jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                            jsonl_file.flush()
                            os.fsync(jsonl_file.fileno())

                            # Store minimal data in shelf to avoid duplicates
                            shelf[url] = {
                                'stored_at': timestamp,
                                'title': hit.get('title', ''),
                                'original_query': query
                            }
                            shelf.sync()

                            query_added += 1

                        # Fixed delay between pages to avoid overwhelming the API
                        time.sleep(self.delay)

                    print(f"[INFO] Query '{query}' completed. Added {query_added} new links. Skipped {query_skipped} duplicates.")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process query '{query}': {e}")
                    continue
                
                total_added += query_added
                total_skipped += query_skipped
                
                # Add a small delay between queries to be respectful
                if i < len(queries):  # Don't sleep after the last query
                    time.sleep(self.delay)

        print(f"[INFO] Batch search completed. Total added: {total_added} new links. Total skipped: {total_skipped} duplicates.")
        return {
            'total_added': total_added,
            'total_skipped': total_skipped,
            'queries_processed': len(queries)
        }