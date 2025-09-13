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
from tqdm.asyncio import tqdm_asyncio   # pip install tqdm>=4.62
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pathlib import Path
import aiofiles


class TextExtractor:
    """Handles text extraction from HTML elements."""
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lower‑casing and collapsing whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def get_page_summary(soup: BeautifulSoup) -> str:
        """Extract a short description from meta tags or the first paragraph."""
        meta_selectors = [
            ("description", {"name": "description"}),
            ("og:description", {"property": "og:description"}),
            ("twitter:description", {"name": "twitter:description"}),
        ]
        for _name, attrs in meta_selectors:
            tag = soup.find("meta", attrs)
            if tag and tag.get("content"):
                return tag["content"].strip()

        # Fallback: first non‑trivial <p>
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                return text
        return ""

    @staticmethod
    def get_surrounding_text(img_tag, max_chars: int = 1000) -> str:
        """Grab figcaptions and nearby text (prev/next sibling) for context."""
        def nearby(start, direction):
            collected: List[str] = []
            current = start
            while current and len(" ".join(collected)) < max_chars // 2:
                current = (
                    current.find_previous_sibling()
                    if direction == "prev"
                    else current.find_next_sibling()
                )
                if not current:
                    parent = start.parent
                    if parent and parent.name not in ["html", "body"]:
                        start = parent
                        current = start
                        continue
                    break

                text = current.strip() if isinstance(current, NavigableString) \
                    else current.get_text(strip=True) \
                    if current.name in ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"] else ""
                if text and len(text) > 10:
                    collected.append(text)
                    break
            return collected

        contexts: List[str] = []

        figure = img_tag.find_parent(["figure", "picture"])
        if figure:
            caption = figure.find("figcaption")
            if caption:
                contexts.append(caption.get_text(strip=True))

        for sib in img_tag.find_next_siblings(["p", "div", "span"], limit=3):
            txt = sib.get_text(strip=True)
            if txt and any(k in txt.lower() for k in ["caption", "image", "photo", "picture", "source"]):
                contexts.append(txt)
                break

        contexts = nearby(img_tag, "prev") + contexts + nearby(img_tag, "next")
        joined = re.sub(r"\s+", " ", " ".join(contexts))
        return joined[:max_chars] + ("…" if len(joined) > max_chars else "")

    @staticmethod
    def extract_structured_content(soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured content from the page in LLM-friendly format."""
        content_data = {
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "links": [],
            "structured_text": "",
            "markdown_content": ""
        }
        
        # Single pass extraction of links (since they're processed separately)
        for link in soup.find_all('a', href=True):
            text = link.get_text(strip=True)
            href = link.get('href')
            if text and href:
                content_data["links"].append({
                    "text": text,
                    "url": href
                })
        
        # Get main content area for focused processing
        main_content = (soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_=re.compile(r'content|main|article')) or 
                    soup)
        
        # Single traversal for content elements
        structured_parts = []
        markdown_parts = []
        
        # Add title and meta description to markdown
        title = soup.find('title')
        if title:
            title_text = title.get_text(strip=True)
            markdown_parts.append(f"# {title_text}\n")
        
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            markdown_parts.append(f"*{meta_desc['content']}*\n")
        
        # Process all content elements in a single pass
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'blockquote']):
            text = element.get_text(strip=True)
            if not text:
                continue
                
            if element.name.startswith('h'):
                # Handle headings
                level = int(element.name[1:])
                content_data["headings"].append({
                    "level": element.name,
                    "text": text
                })
                
                # Add to both structured and markdown formats
                structured_parts.append(f"{'#' * level} {text}")
                markdown_parts.append(f"{'#' * level} {text}")
                
            elif element.name == 'p':
                # Handle paragraphs
                if len(text) > 20:  # Filter out very short paragraphs
                    content_data["paragraphs"].append(text)
                    structured_parts.append(text)
                    markdown_parts.append(text)
                    
            elif element.name in ['ul', 'ol']:
                # Handle lists
                list_items = []
                list_markdown = []
                
                for li in element.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if item_text:
                        list_items.append(item_text)
                        prefix = "- " if element.name == 'ul' else "1. "
                        list_markdown.append(f"{prefix}{item_text}")
                
                if list_items:
                    content_data["lists"].append({
                        "type": element.name,
                        "items": list_items
                    })
                    markdown_parts.extend(list_markdown)
                    markdown_parts.append("")  # Empty line after list
                    
            elif element.name == 'blockquote':
                # Handle blockquotes (only in markdown)
                markdown_parts.append(f"> {text}")
        
        # Build final strings efficiently
        content_data["structured_text"] = "\n".join(structured_parts)
        content_data["markdown_content"] = "\n".join(markdown_parts)
        
        return content_data


class ImageValidator:
    """Validates image URLs and filters out ads/placeholders."""
    AD_DOMAIN_PATTERNS = [
        re.compile(
            r"\.(doubleclick\.net|googlesyndication\.com|adservice\.google\.com|"
            r"adnetwork\.com|adnxs\.com|yieldmanager\.com|pubmatic\.com|rubiconproject\.com|"
            r"applovin\.com|taboola\.com|outbrain\.com|smartadserver\.com|zedo\.com|"
            r"pulse3d\.com|casalemedia\.com|lijit\.com|analytics\.google\.com|"
            r"connect\.facebook\.net|ads\.pinterest\.com|analytics\.twitter\.com|"
            r"bat\.bing\.com|cdn\.adsafeprotected\.com|scorecardresearch\.com|"
            r"quantserve\.com|moatads\.com)$",
            re.IGNORECASE,
        )
    ]
    GOOD_PATH_PATTERNS = [
        re.compile(r"\b(image|img|photo|picture|media|upload|content|wp-content)\b", re.IGNORECASE)
    ]
    BAD_PATH_PATTERNS = [
        re.compile(
            r"\b(placeholder|spinner|tracking|pixel|blank|spacer|clear\.gif|"
            r"transparent\.png|loading|1x1\.|\.svg$|data:image/svg)\b",
            re.IGNORECASE,
        )
    ]
    GOOD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

    @staticmethod
    def is_valid_image_url(url: str) -> bool:
        if not url:
            return False
        parsed = urlsplit(url)
        domain, path = parsed.netloc.lower(), parsed.path.lower()
        if any(p.search(domain) for p in ImageValidator.AD_DOMAIN_PATTERNS):
            return False
        base_path = path.split("?", 1)[0]
        if any(base_path.endswith(ext) for ext in ImageValidator.GOOD_EXTENSIONS):
            return True
        if any(p.search(base_path) for p in ImageValidator.GOOD_PATH_PATTERNS):
            if not any(bp.search(base_path) for bp in ImageValidator.BAD_PATH_PATTERNS):
                return True
        return False


class SingaporeFilterSync:
    def __init__(self, base_url: str, model_name: str, max_workers: int = 16):
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy"
        )
        self.model_name = model_name
        self.max_workers = max_workers

    def _is_singapore_related(self, entry: Dict[str, Any]) -> bool:
            title = entry.get("title", "").strip()
            href = entry.get("href", "").strip()
            content = entry.get("content", entry.get("description", "")).strip()
            prompt = f"Title: {title}\nURL: {href}\nSnippet: {content}"

            system = (
                "You are given the title, URL, and a short snippet of a web page. "
                "Reply 'yes' or 'no' depending on whether it is related to Singapore. "
                )

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                n=3,
                temperature=0.3,
                max_completion_tokens=3,
            )

            yes_votes = sum(
                1 for c in resp.choices
                if c.message.content.strip().lower().startswith("yes")
            )
            return yes_votes > 1

    def filter_entries(self, file_path: str) -> List[str]:
        entries: List[Dict[str, Any]] = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Ensure the required keys exist
                        if "href" not in data or "title" not in data:
                            logger.error(
                                f"Missing 'href' or 'title' in line: {line}"
                            )
                            continue
                        entry = {
                            "title": data["title"],
                            "href": data["href"],
                            "content": data.get("content") or data.get("description", ""),
                        }
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(f"{e}: Invalid JSON line in: {file_path} at {line}")
        except FileNotFoundError as e:
            logger.error(f"{e}: JSONL file not found: {file_path}")
            return []

        if not entries:
            return []

        filtered: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_entry = {
                pool.submit(self._is_singapore_related, entry): entry
                for entry in entries
            }
            for future in tqdm(
                as_completed(future_to_entry),
                total=len(entries),
                desc="Singapore relevance check",
            ):
                entry = future_to_entry[future]
                try:
                    if future.result():
                        filtered.append(entry)
                except Exception as exc:
                    logger.error("LLM call failed for entry %s: %s", entry, exc)

        logger.info(
            "Kept %s/%s entries after Singapore filter", len(filtered), len(entries)
        )

        # Return only the href strings
        return [entry["href"] for entry in filtered]


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