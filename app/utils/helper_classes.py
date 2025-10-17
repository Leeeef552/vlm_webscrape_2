from bs4 import BeautifulSoup, NavigableString
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit
from ..utils.logger import logger
from openai import OpenAI, BadRequestError
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
        """Grab figcaptions and nearby text (prev/next sibling) for context to the image"""
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
        content = content[:5000]
        prompt = f"Title: {title}\nURL: {href}\nSnippet: {content}"

        system = (
            "You are given the title, URL, and a short snippet of a web page."
            "Reply 'yes' or 'no' depending on whether it is related to Singapore."
            "Only reply 'yes' if you are 100 percent certain it is about or related to Singapore"
            )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            n=3,
            temperature=0.3,
            max_completion_tokens=5,
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
                        logger.error(f"{e}: Invalid JSON line in: {file_path}")
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