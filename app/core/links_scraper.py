import os
import shelve
import json
import time
from datetime import datetime
from duckduckgo_search import DDGS
import tldextract
from ..configs.config import ScraperConfig

class LinkScraper:
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.ddgs = DDGS(timeout=config.timeout)

        # Ensure folders exist
        os.makedirs(config.base_dir, exist_ok=True)
        os.makedirs(config.db_dir, exist_ok=True)

        # Fixed delay between requests to avoid rate limiting
        self.delay = 1.0  # seconds

        # Paths for JSONL & Shelf
        self.jsonl_path = os.path.join(config.base_dir, 'links.jsonl')
        self.shelf_path = config.db_dir


    def get_registered_domain(self, url):
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    

    def search_and_store(self, query: str):
        """Search and store links for a specific query, with gentle rate-limiting and progress save."""
        print(f"[INFO] Searching for '{query}'...")
        count = 0
        skipped = 0

        with shelve.open(self.shelf_path) as shelf, open(self.jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            # Ensure we have an iterator regardless of return type
            iterator = iter(self.ddgs.text(
                keywords=query,
                region=self.config.region,
                safesearch=self.config.safesearch,
                timelimit=self.config.timelimit,
                max_results=self.config.max_results
            ))

            while True:
                try:
                    r = next(iterator)
                except StopIteration:
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if 'rate limit' in msg or '429' in msg:
                        print(f"[INFO] Rate limited (error: {e}). Waiting for {self.delay} seconds before retrying...")
                        time.sleep(self.delay)
                        continue
                    else:
                        raise

                url = r.get('href') or r.get('link')
                if not url:
                    print("[WARNING] Skipping result with no URL")
                    continue

                reg_domain = self.get_registered_domain(url)
                if reg_domain in shelf:
                    skipped += 1
                    continue


                # Add a timestamp and metadata
                timestamp = datetime.utcnow().isoformat()
                r['stored_at'] = timestamp
                r['original_query'] = query

                # Write the full DuckDuckGo result to JSONL
                jsonl_file.write(json.dumps(r, ensure_ascii=False) + '\n')
                jsonl_file.flush()
                os.fsync(jsonl_file.fileno())

                # Store minimal data in shelf to avoid duplicates
                shelf[url] = {
                    'stored_at': timestamp,
                    'title': r.get('title', ''),
                    'original_query': query
                }
                shelf.sync()

                count += 1
                # Fixed delay to avoid overwhelming the API
                time.sleep(self.delay)

        print(f"[INFO] Done. Added {count} new links. Skipped {skipped} duplicates.")
        return count

    def iter_records(self):
        """Yield all stored DuckDuckGo results from the JSONL file."""
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line.strip())
        except FileNotFoundError:
            print(f"[WARNING] JSONL file not found: {self.jsonl_path}")
            return
