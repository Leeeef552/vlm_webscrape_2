from __future__ import annotations
import re
import asyncio
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit
from bs4 import BeautifulSoup, NavigableString
from playwright.async_api import (
    async_playwright,
    Playwright,
    Browser,
    BrowserContext,
)
import json
from datetime import datetime, timezone
import os
from ..configs.config import ScraperConfig
from ..utils.logger import logger
from openai import OpenAI, BadRequestError
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pathlib import Path
import aiofiles
from ..utils.helper_classes import ImageValidator, TextExtractor, SingaporeFilterSync


class Scraper:
    """Extract image and markdown content using async Playwright and asyncio."""
    def __init__(self, ScraperConfig):
        self.text_extractor = TextExtractor()
        self.image_validator = ImageValidator()
        self._pw: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.scraper_config = ScraperConfig
        self.semaphore = asyncio.Semaphore(self.scraper_config.concurrency)
        self.pdf_dir = self.scraper_config.pdf_dir

    async def __aenter__(self) -> Scraper:
        self._pw = await async_playwright().start()
        self.browser = await self._pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.browser:
            await self.browser.close()
        if self._pw:
            await self._pw.stop()

    # --------------------- Internal helpers ----------------------------

    @staticmethod
    def _canonicalise_url(url: str) -> str:
        parts = urlsplit(url)
        path = re.sub(r"/\d{2,4}/\d{2,4}/", "/", parts.path)
        path = re.sub(r"-\d{2,4}x\d{2,4}(?=\.\w+$)", "", path)
        query = re.sub(r"(\?|&)(w|width|h|height|size)=\d+", "", parts.query, flags=re.I)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path.rstrip("/"), query, ""))

    @staticmethod
    def _parse_resolution_from_url(url: str) -> tuple[int, int]:
        m = re.search(r"/(\d{2,4})/(\d{2,4})/", url)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"-([1-9]\d{2,4})x([1-9]\d{2,4})(?=\.\w+$)", url)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 0, 0

    @staticmethod
    def _is_likely_pdf_url(url: str) -> bool:
        """Improved heuristic to detect likely PDF URLs."""
        url_lower = url.lower()
        parsed = urlsplit(url_lower)

        # Case 1: Ends with .pdf
        if parsed.path.endswith(".pdf"):
            return True

        # Case 2: Contains 'pdf' in path or query (e.g., /viewcontent.cgi?article=...&context=...)
        if "pdf" in parsed.path or "pdf" in parsed.query:
            return True

        # Case 3: Known academic repository patterns
        if re.search(r"\b(viewcontent|article)\.cgi\?", parsed.path):
            # Common in ERIC, institutional repos
            return True

        # Case 4: Attachment or download indicators
        if re.search(r"\b(attachment|download|file|document)\b", parsed.path):
            return True

        return False

    @staticmethod
    def _confirm_pdf_url(url: str, timeout: int = 10) -> bool:
        """Confirm URL serves a PDF using HEAD, then GET with byte sniffing if needed."""
        # Try HEAD first
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout, stream=True)
            content_type = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return True
            # Some misconfigured servers
            if "octet-stream" in content_type and "pdf" in r.headers.get("Content-Disposition", "").lower():
                return True
        except Exception:
            pass  # Proceed to GET fallback

        # Fallback: GET first 1024 bytes and sniff magic number
        try:
            r = requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=timeout)
            r.raise_for_status()
            chunk = r.raw.read(1024)
            if chunk.startswith(b"%PDF"):
                return True
        except Exception:
            pass

        return False

    def _filter_pdf_urls(self, urls: list[str]) -> list[str]:
        """Filter and confirm PDF URLs concurrently."""
        likely_pdfs = [u for u in urls if self._is_likely_pdf_url(u)]
        confirmed: list[str] = []

        with ThreadPoolExecutor(max_workers=self.scraper_config.concurrency) as pool:
            futures = {pool.submit(self._confirm_pdf_url, u): u for u in likely_pdfs}
            
            # Use tqdm to show progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="PDF documents check"):
                url = futures[future]
                try:
                    if future.result():
                        confirmed.append(url)
                except Exception:
                    continue
        return confirmed

    async def _fetch_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Render a URL in a fresh context, then return its BeautifulSoup asynchronously."""
        if not self.browser:
            raise RuntimeError("Browser not started. Use 'async with'.")
        async with self.semaphore:
            context: BrowserContext = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()
            try:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except asyncio.TimeoutError:
                    logger.error("Timeout navigating to %s", url)
                    return None
                
                last_h = -1
                start_time = time.time()
                max_scroll_time = 20  # seconds

                for _ in range(10):
                    if time.time() - start_time > max_scroll_time:
                        break

                    # Add timeout for scroll height evaluation
                    try:
                        h = await asyncio.wait_for(
                            page.evaluate("document.body.scrollHeight"), 
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout evaluating scroll height on %s", url)
                        break
                    
                    if h == last_h:
                        break
                    last_h = h

                    # Add timeout for scroll operation
                    try:
                        await asyncio.wait_for(
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)"),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout scrolling on %s", url)
                        break

                    # Add timeout for getting page content
                    try:
                        html = await asyncio.wait_for(page.content(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.error("Timeout getting content from %s", url)
                        return None
                    
            except Exception as e:
                logger.error("Failed to fetch %s: %s", url, e)
                html = None
            finally:    
                await context.close()
        return BeautifulSoup(html, "html.parser") if html else None
        
    # .........................Extraction logic ...........................

    def _extract_image_data(self, img_tag, page_url: str, page_title: str, page_summary: str) -> Dict[str, Any]:
        """Extract metadata from a single <img>/<picture> element."""

        def parse_srcset(srcset: str) -> List[tuple[str, str]]:
            return [tuple(map(str.strip, entry.split(" "))) if " " in entry else (entry.strip(), "1x") for entry in srcset.split(",")]

        def pick_best(candidates: List[tuple[str, str]]) -> Optional[str]:
            if not candidates:
                return None

            def score(descriptor: str) -> int:
                m = re.match(r"(\d+)(w|x)", descriptor)
                return int(m.group(1)) if m else 1

            return max(candidates, key=lambda c: score(c[1]))[0]

        picture = img_tag.find_parent("picture")
        image_url: Optional[str] = None

        # Prefer highest‑res <source srcset="…"> inside <picture>
        if picture:
            for src in picture.find_all("source"):
                ss = src.get("srcset")
                if ss:
                    best = pick_best(parse_srcset(ss))
                    if best:
                        image_url = urljoin(page_url, best)
                        break

        # Fallback: srcset on <img>
        if not image_url and img_tag.get("srcset"):
            best = pick_best(parse_srcset(img_tag["srcset"]))
            if best:
                image_url = urljoin(page_url, best)

        # Fallback: plain src
        if not image_url:
            src = img_tag.get("src")
            if not src:
                return {}
            image_url = urljoin(page_url, src)

        if not self.image_validator.is_valid_image_url(image_url):
            return {}

        return {
            "image_url": image_url,
            "page_url": page_url,
            "page_title": page_title,
            "alt_text": img_tag.get("alt", "").strip(),
            "title_attribute": img_tag.get("title", "").strip(),
            "raw_caption": self.text_extractor.get_surrounding_text(img_tag),
            "page_summary": page_summary,
            "extracted_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

    def _extract_markdown_content(self, url: str, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract full text content from a page."""
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        page_summary = self.text_extractor.get_page_summary(soup)
        
        # Extract structured content
        content_data = self.text_extractor.extract_structured_content(soup)
        
        # Use markdown content as the main text_content (better for LLM ingestion)
        text_content = content_data["markdown_content"]
             
        return {
            "page_url": url,
            "page_title": page_title,
            "page_summary": page_summary,
            "text_content": text_content,
            "extracted_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

    # ----------------------------Public API------------------------------

    async def _extract_both_from_url(self, url: str) -> Dict[str, Any]:
        """Extract both images and markdown from a URL in one operation"""
        soup = await self._fetch_page_content(url)
        if not soup:
            return {"images": [], "markdown": {}}
        
        # Extract both in single pass
        markdown_data = self._extract_markdown_content(url, soup)
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        page_summary = self.text_extractor.get_page_summary(soup)
        
        # Image extraction (same as before)
        grouped = {}
        for img in soup.find_all("img"):
            data = self._extract_image_data(img, url, page_title, page_summary)
            if not data:
                continue
            key = self._canonicalise_url(data["image_url"])
            w, h = self._parse_resolution_from_url(data["image_url"])
            data["_w"], data["_h"] = w, h
            grouped.setdefault(key, []).append(data)

        images = []
        for group in grouped.values():
            best = max(group, key=lambda d: d["_w"] * d["_h"])
            best.pop("_w", None)
            best.pop("_h", None)
            images.append(best)

        return {"images": images, "markdown": markdown_data}

    async def extract_all_content(self, urls: List[str]) -> Dict[str, List[Any]]:
        # 1) Singapore filter
        filterer = SingaporeFilterSync(
            base_url=self.scraper_config.validator_base_url,
            model_name=self.scraper_config.validator_model_name,
            max_workers=self.scraper_config.concurrency,
        )
        logger.info("Running Singapore-relevance filter")
        sg_urls = await asyncio.to_thread(filterer.filter_entries, urls)

        # 2) Detect PDF URLs
        logger.info("Running Filter for PDF docs")
        pdf_urls = await asyncio.to_thread(self._filter_pdf_urls, sg_urls)
        html_urls = [u for u in sg_urls if u not in pdf_urls]
        logger.info("PDF documents filtered: %d / %d", len(pdf_urls), len(sg_urls))

        # 3) Persist PDF metadata
        if pdf_urls:
            pdf_path = Path(self.scraper_config.pdf_dir)
            pdf_path.mkdir(parents=True, exist_ok=True)
            metadata_file = pdf_path / "metadata.jsonl"

            async with aiofiles.open(metadata_file, "a") as f:
                for url in pdf_urls:
                    await f.write(
                        json.dumps(
                            {
                                "url": url,
                                "scraped_at": datetime.now(timezone.utc).isoformat() + "Z",
                            }
                        )
                        + "\n"
                    )

        # 4) Scrape non-PDF URLs
        semaphore = asyncio.Semaphore(self.scraper_config.concurrency)

        async def extract_safe(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._extract_both_from_url(url), timeout=90
                    )
                except Exception as e:
                    logger.error("Failed %s: %s", url, e)
                    return {"images": [], "markdown": {}}
                
        logger.info("Begin Scraping of %d HTML sites", len(html_urls))
        if html_urls:
            results = await tqdm_asyncio.gather(
                *(extract_safe(u) for u in html_urls),
                desc="Scraping URLs",
                unit="url",
            )
        else:
            results = []

        all_images, all_markdowns = [], []
        for res in results:
            all_images.extend(res["images"])
            all_markdowns.append(res["markdown"])

        return {"images": all_images, "markdowns": all_markdowns, "pdfs": [{"url": u} for u in pdf_urls]}