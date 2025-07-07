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


class Scraper:
    """Extract image and markdown content using async Playwright and asyncio."""
    def __init__(self, ScraperConfig):
        self.text_extractor = TextExtractor()
        self.image_validator = ImageValidator()
        self._pw: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.scraper_config = ScraperConfig
        self.semaphore = asyncio.Semaphore(self.scraper_config.concurrency)

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
    
    def _load_links_from_jsonl(self, file_path: str) -> None:
        """Load all 'href' values from a JSONL file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "href" in data:
                            self.links.append(data["href"])
                        else:
                            print(f"[Line {line_number}] Missing 'href' key: {line}")
                    except json.JSONDecodeError as e:
                        print(f"[Line {line_number}] Invalid JSON: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Links file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read links file: {e}")

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

    async def _click_load_more(self, page) -> None:
        """Async click any “load more / show more” buttons until none remain."""
        selectors = [
            'button:has-text("load more")',
            'button:has-text("show more")',
            'a:has-text("load more")',
            'a:has-text("show more")',
        ]
        clicks = 0
        max_clicks = 5
        for _ in range(max_clicks):
            found = False
            for sel in selectors:
                locator = page.locator(sel)
                count = await locator.count()
                if count:
                    try:
                        await locator.first.wait_for(state="visible", timeout=3000)
                        await locator.first.scroll_into_view_if_needed()
                        await locator.first.click()
                        clicks += 1
                        found = True
                        await page.wait_for_timeout(1000)
                        logger.info("Clicked 'load more' button %s/%s", clicks, max_clicks)
                        break
                    except Exception:
                        continue
            if not found:
                break

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
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await self._click_load_more(page)
                last_h = -1
                for _ in range(20):
                    h = await page.evaluate("document.body.scrollHeight")
                    if h == last_h:
                        break
                    last_h = h
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1500)
                html = await page.content()
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
            "content_context": None,
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
        logger.info(f"Extracting content from: {url}")
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

    # New batch method
    async def extract_all_content(self, urls: List[str]) -> Dict[str, List[Any]]:
        """Extract both images and markdown for all URLs concurrently"""
        logger.info(f"Extracting content from {len(urls)} URLs")
        
        tasks = [self._extract_both_from_url(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate results
        all_images = []
        all_markdowns = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Extraction failed: {res}")
                continue
            all_images.extend(res["images"])
            all_markdowns.append(res["markdown"])
        
        # Handle file writing with error catching
        try:
            with open(self.scraper_config.images_outfile, "w", encoding="utf-8") as f:
                json.dump(all_images, f, ensure_ascii=False, indent=2)
            logger.info(f"Extracted {len(all_images)} images")
        except FileNotFoundError:
            logger.error(f"FileNotFoundError: Check images_outfile path '{self.scraper_config.images_outfile}' in your config.")
        except OSError as e:
            logger.error(f"OSError writing images_outfile '{self.scraper_config.images_outfile}': {e}")
        except TypeError as e:
            logger.error(f"JSON encoding error while writing images_outfile: {e}")

        try:
            with open(self.scraper_config.markdown_outfile, "w", encoding="utf-8") as f:
                json.dump(all_markdowns, f, ensure_ascii=False, indent=2)
            logger.info(f"Processed {len(all_markdowns)} webpages")
        except FileNotFoundError:
            logger.error(f"FileNotFoundError: Check markdown_outfile path '{self.scraper_config.markdown_outfile}' in your config.")
        except OSError as e:
            logger.error(f"OSError writing markdown_outfile '{self.scraper_config.markdown_outfile}': {e}")
        except TypeError as e:
            logger.error(f"JSON encoding error while writing markdown_outfile: {e}")
        
        return {"images": all_images, "markdowns": all_markdowns}
        

# async def main(): 
#     import time
#     start_time = time.time()
#     print("Starting the process...")
    
#     URLS = [
#         "https://strictlysingapore.com/tourist-tips/",
#         "https://www.visitsingapore.com/",
#         "https://migrationology.com/singapore-food/",
#         "https://builtinsingapore.com/articles/top-companies-in-singapore",
#         "https://bbcincorp.com/sg/articles/singapore-government-agencies",
#         "https://www.sutrahr.com/recruitment-agencies-in-singapore/"
#     ]

#     async with Scraper(concurrency=3) as extractor:
#         # Single extraction pass for all URLs
#         results = await extractor.extract_all_content(URLS)
#         images = results["images"]
#         markdown = results["markdowns"]

#     images_file_dir = "../storage/images_metadata"
#     images_outfile = f"{images_file_dir}/images_metadata_async.json"
#     os.makedirs(images_file_dir, exist_ok=True)
#     with open(images_outfile, "w", encoding="utf-8") as f:
#         json.dump(images, f, ensure_ascii=False, indent=2)
    
    
#     markdown_file_dir = "../storage/text_markdown"
#     markdown_outfile = f"{markdown_file_dir}/text_markdown_async.json"
#     os.makedirs(markdown_file_dir, exist_ok=True)
#     with open(markdown_outfile, "w", encoding="utf-8") as fp:
#         json.dump(markdown, fp, ensure_ascii=False, indent=2)

#     end_time = time.time()
#     print(f"extracted {len(images)} images")
#     print(f"extracted {len(markdown)} pages")
#     print(f"Process completed in {end_time - start_time:.2f} seconds.")

# if __name__ == "__main__":
#     asyncio.run(main())