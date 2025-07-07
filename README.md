# LLM-driven Web Crawler & Knowledge Miner

A modular, LLM-augmented pipeline to discover, scrape, extract, and expand knowledge on any topic using recursive search-engine queries, web scraping, and LLM-powered topic/entity extraction.

---

## 🚀 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture & Design](#architecture--design)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Pipeline Details](#pipeline-details)
* [Extending the Knowledge Base](#extending-the-knowledge-base)
* [Exploration vs. Exploitation](#exploration-vs-exploitation)
* [Roadmap & To Do](#roadmap--to-do)
* [Contributing](#contributing)
* [License](#license)

---

## 📖 Overview

This project provides a fully automated research engine that:

1. **Crawls** the web via a search‑engine API (e.g., SearXNG).
2. **Scrapes** HTML pages (and PDFs) using Playwright and BeautifulSoup.
3. **Extracts** structured content and media metadata in LLM‑friendly formats.
4. **Builds** a queryable knowledge base of topics, entities, contexts, and concepts.
5. **Expands** with new search queries via LLM‑driven query rewriting and exploration/exploitation strategies.

By combining search, scraping, and LLMs, you can bootstrap a focused knowledge graph or document store on any subject.

---

## ✨ Features

* **Crawler** (`core/crawler.py`)

  * Batch queries to SearXNG or other engines.
  * Deduplicates using a persistent `shelve` index.
  * Checkpointing to JSONL for safe writes.

* **Scraper** (`core/scraper.py`)

  * Async fetching via Playwright (headless Chromium).
  * Smart scrolling + “load more” clicks.
  * Text extraction with headings, paragraphs, lists, and markdown output.
  * Image metadata extraction & validation.

* **Knowledge Extraction** (`core/topic_extraction.py`)

  * (PLANNED) LLM-driven extraction of topics, entities, and concept graphs.

* **Query Transformation** (`main.py`)

  * Baseline query templates (e.g., adding “travel tips”, “landmarks”).
  * (PLANNED) LLM-based query rewriting for exploration/exploitation.

* **Storage** (`storage/`)

  * JSONL/text files for scraped links and content.
  * `shelve` for fast URL deduplication.

* **Utilities** (`utils/`)

  * Custom logger, link loader, and other helpers.

---

## 🏗️ Architecture & Design

```
[ User Query ]
      |
      v                +---------------------------+
[ Query Rewriter ]--> |  Core Crawler (SearXNG)   | --> JSONL links + shelve index
      |
      v                +---------------------------+
[ Expanded Queries ]
      |
      v                +---------------------------+
[ Core Scraper ]----> |  Playwright + BS4         | --> Markdown + images JSON
      |                +---------------------------+
      v
[ Topic Extraction ] (LLM)
      v
[ Knowledge Base ] (vector store / graph / JSON)
      |
      v
[ Next-Gen Query Expansion ]
```

1. **Initial Seed** —  start from a root topic.
2. **Query Expansion** — LLM rewrites to generate sub‑queries.
3. **Crawl & Scrape** — fetch links & content, store metadata.
4. **Topic Extraction** — invoke an LLM to pull out entities, themes.
5. **Knowledge Base** — index results (e.g., vector DB) for search.
6. **Recursive Loop** — pick new queries (explore/exploit) and repeat.

---

## 📂 Directory Structure

```
app/
├── configs/             # JSON schema & default config
├── core/
│   ├── crawler.py       # Search-engine crawler
│   ├── scraper.py       # Async web scraper & extractor
│   └── topic_extraction.py  # LLM-based topic/entity extractor (TBD)
├── storage/
│   ├── links.jsonl      # Collected link metadata
│   ├── markdown.json    # Scraped page content
│   └── images.json      # Extracted image metadata
├── utils/
│   ├── logger.py        # Unified logging
│   └── utils.py         # Helpers (link loading, etc.)
├── main.py              # CLI entrypoint & pipeline orchestration
└── README.md            # This file
```

---

## ⚙️ Installation

1. **Clone** the repo:

   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo
   ```

2. **Create & activate** a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Update your `.gitignore`** to exclude your virtual environment folder (e.g., `.venv/`) and other temporary files.

5. **Run** SearXNG locally (or point at another engine) on port `8124`.

## 🔧 Configuration

This pipeline relies on a **self‑hosted SearxNG** instance as its meta‑search engine. Follow the official installation guide:

* **SearxNG Admin Installation (Docker)**: [https://docs.searxng.org/admin/installation-docker.html](https://docs.searxng.org/admin/installation-docker.html)
* **Docker Image**: [https://hub.docker.com/r/searxng/searxng](https://hub.docker.com/r/searxng/searxng)

Once deployed (default port `8124`), update your `config.json` so that the `crawler.searxng_url` matches your SearxNG endpoint. Example config:

```json
{
  "crawler": {
    "searxng_url": "http://localhost:8124/search",
    "query": "Singapore",
    "language": "en",
    "pages": 1,
    "time_range": "year",
    "timeout": 3,
    "links_file_path": "app/storage/raw_links/links.jsonl",
    "shelf_path": "app/storage/raw_links/db_link_hashing/link_hash_db"
  },
  "scraper": {
    "concurrency": 4,
    "links_file_path": "app/storage/raw_links/links.jsonl",
    "images_outfile": "app/storage/images_metadata/images_metadata.json",
    "markdown_outfile": "app/storage/images_metadata/text_markdown.json"
  }
}
```

> **⚠️ Before running**: ensure your SearxNG container is up and the `searxng_url` is reachable (e.g., `curl http://localhost:8124/search?q=test`).

## 🚀 Usage

Run the full pipeline from your terminal:

```bash
python -m app.main --config app/configs/config.json
```

Or, import & call individual modules:

```python
from app.core.crawler import Crawler
from app.utils.utils import load_links

# 1. Crawl links
tool = Crawler(config.crawler)
new = tool.search_and_store_batch(["singapore attractions", "singapore food"])

# 2. Scrape content
links = load_links(config.scraper.links_file_path)
async with Scraper(config.scraper) as s:
    results = await s.extract_all_content(links)
```

---

## 🧠 Extending the Knowledge Base

* **Topic Extraction**: Use `core/topic_extraction.py` to feed scraped markdown into an LLM (OpenAI, Claude, etc.)

  * Extract **entities**, **topics**, **concepts**, **relations**.
  * Store in your vector DB or graph database.

* **Query Rewriting**: Replace the placeholder `transform_query()` in `main.py` with an LLM prompt that:

  1. **Exploits** high‑confidence areas (common topics).
  2. **Explores** sparse or niche subtopics.
  3. Ranks & returns top‑K new queries.

* **Exploitation vs. Exploration**:

  * Exploitation: repeat queries around frequent entities to deepen coverage.
  * Exploration: generate novel queries for long‑tail or emerging terms.
  * Use heuristic scoring (e.g., TF-IDF on your KB) or ask the LLM to rate.

---

## 📈 Exploration vs. Exploitation Strategies

1. **Frequency-based**: prioritize queries containing high‑frequency entities (exploit) vs. low‑frequency (explore).
2. **Recency-based**: use time\_range filters to surface fresh content.
3. **Diversity-aware**: ask the LLM to generate semantically diverse queries.

> Tip: prompt the LLM with your current KB summary, then ask:
>
> ```text
> "Given this list of discovered topics: [A, B, C, ...], suggest five new search queries that explore lesser-covered areas."
> ```

---

## 🛣️ Roadmap & To Do

* [x] Basic crawler implementation (`core/crawler.py`)
* [x] Basic scraper implementation (`core/scraper.py`)
* [x] Baseline query templates in `main.py`
* [ ] LLM-driven query rewriting (in `process.py`)
* [ ] Exploration/exploitation scoring and logic for query expansion
* [ ] Topic/entity extraction module (`core/topic_extraction.py`)
* [ ] PDF content support in `scraper.py`
* [ ] Checkpointing & resume support for long-running crawls
* [ ] Integration with a vector database or graph store for the knowledge base
* [ ] Recursive query-driven pipeline loop
* [ ] Detailed documentation and usage examples for each component

<!-- ## 🤝 Contributing

1. Fork & clone.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit your changes and push.
4. Open a Pull Request describing your enhancement.

Please follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE). -->
