import asyncio
from pathlib import Path
import json
from datetime import datetime

from .configs.config import load_config
from .core.crawler import Crawler
from .core.scraper import Scraper
from .core.topic_extractor import TopicExtractor
from .core.query_expansion import QueryExpansion
from .utils.logger import logger
from .utils.utils import load_links

# === Configuration Variables ===
# Path to the JSON config file
CONFIG_PATH = "/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/configs/config.json"
# Number of full pipeline iterations (search → scrape → topics → expand)
NUM_ITERATIONS = 3
# Initial root query for the first crawl
INITIAL_QUERY = [
    "Lee Kuan Yew biography and contributions",
    "Tharman Shanmugaratnam policy influence",
    "Goh Chok Tong economic reforms",
    "Halimah Yacob leadership and presidency",
    "S R Nathan legacy",
    "Ho Ching and Temasek Holdings",
    "Ong Teng Cheong public service contributions",
    "Changi Airport international reputation",
    "Sentosa Island attractions and development",
    "Jurong Island petrochemical hub",
    "Bukit Timah Nature Reserve biodiversity",
    "Pulau Ubin conservation and rustic life",
    "Kampong Glam cultural heritage",
    "Little India history and community",
    "Chinatown traditions and events",
    "Marina Bay Sands significance",
    "Gardens by the Bay architecture and gardens",
    "Singapore Zoo wildlife innovation",
    "National Gallery Singapore art collections",
    "Singapore Botanic Gardens UNESCO status",

    # Organizations & Structures
    "People's Action Party PAP history",
    "Workers' Party Singapore evolution",
    "Temasek Holdings investment strategies",
    "GIC sovereign wealth fund Singapore",
    "Monetary Authority of Singapore MAS role",
    "Economic Development Board Singapore EDB",
    "National Environment Agency Singapore NEA",
    "Urban Redevelopment Authority Singapore URA",
    "Housing and Development Board HDB history and functions",
    "Central Provident Fund Board CPF Board",
    "National Library Board Singapore NLB initiatives",

    # Social & Cultural Topics
    "Singlish features and origins",
    "Peranakan culture in Singapore",
    "Malay community in Singapore",
    "Indian community Singapore history",
    "Chinese clan associations Singapore",
    "Singapore hawker culture",
    "Singapore MRT system development",
    "Oral History Centre Singapore",
    "Operation Coldstore significance",
    "Singapore National Day Parade evolution",

    # Landmarks & Districts
    "Esplanade Theatres on the Bay events",
    "Bishan-Ang Mo Kio Park otters",
    "Kranji War Memorial history",
    "Singapore Sports Hub facilities",
    "Punggol Digital District vision",
    "Tuas Mega Port expansion",
    "Sungei Buloh Wetland Reserve",

    # Events & Historical Moments
    "Separation from Malaysia 1965",
    "Bukit Ho Swee Fire impact",
    "Japanese Occupation of Singapore",
    "Asian Financial Crisis effects on Singapore",
    "COVID-19 pandemic response Singapore",
    "Speak Mandarin Campaign milestones",
    "National Service conscription policy",
    "Singapore Bicentennial commemorations",

    # Science, Tech and Education
    "A*STAR research institutes Singapore",
    "Singapore University of Technology and Design SUTD",
    "National University of Singapore NUS ranking",
    "Singapore Science Centre attractions",
    "Smart Nation initiative Singapore",
    "Biopolis biomedical research hub",
    "Infocomm Media Development Authority IMDA Singapore",
    "Cyber Security Agency of Singapore projects",

    # Media & Literature
    "Singapore Literature Prize winners",
    "The Straits Times newspaper history",
    "Berita Harian Malay newspaper legacy",
    "Tamil Murasu Tamil community news",
    "Channel NewsAsia Singapore media landscape",
    "Mothership Singapore digital media",
    "Singapore films internationally recognized"
]


async def main():
    # Load pipeline configuration
    logger.info(f"Loading configurations from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    if config is None:
        logger.error("Could not load config. Exiting...")
        return

    crawler_cfg = config["crawler"]
    scraper_cfg = config["scraper"]
    topic_cfg = config["topic_extractor"]
    expand_cfg = config["query_expansion"]

    # Initialize crawler
    crawler = Crawler(crawler_cfg)

    for i in range(NUM_ITERATIONS):
        print("=" * 29)
        print(f"====    Iteration {i+1}/{NUM_ITERATIONS}    ====")
        print("=" * 29)

        # 1) Crawl: initial or batch
        if len(INITIAL_QUERY) == 1:
            q = INITIAL_QUERY[0]
            logger.info(f"Crawling initial query '{q}'...")
            run_links_file = crawler.search_and_store(q)
        else:
            logger.info(f"Crawling batch queries: {INITIAL_QUERY}...")
            run_links_file = crawler.search_and_store_batch(INITIAL_QUERY)

        # 2) Load discovered links
        logger.info("Loading links for scraping...")

        scraper_cfg.links_file_path = run_links_file
        

        links = load_links(scraper_cfg.links_file_path)

        # 3) Scrape content
        logger.info(f"Scraping {len(links)} links with concurrency={scraper_cfg.concurrency}...")
        async with Scraper(scraper_cfg) as scraper:
            out = await scraper.extract_all_content(links)
            images = out["images"]
            markdowns = out["markdowns"]
            
        # 4) Save scraped outputs
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = Path(scraper_cfg.images_dir) / f"images_metadata_{ts}.json"
        md_file = Path(scraper_cfg.markdown_dir) / f"text_markdown_{ts}.json"
        img_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(img_file, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)
        with open(md_file, "w", encoding="utf-8") as f:
            json.dump(markdowns, f, ensure_ascii=False, indent=2)

        # 5) Topic extraction
        logger.info("Extracting topics from markdown output...")
        topic_cfg.data_file = str(md_file)
        extractor = TopicExtractor(topic_cfg)
        stats = extractor.extract_from_file()
        logger.info(f"Total unique entities: {stats['total_entities']}")
        for label, count in sorted(stats['counts_by_label'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label}: {count}")

        # 6) Query expansion
        logger.info("Generating new queries via expansion...")
        expander = QueryExpansion(expand_cfg)
        current_queries = expander.get_queries(4)
        logger.info(f"Generated {len(current_queries)} queries: {current_queries}")

    logger.info("Pipeline run complete.")

if __name__ == "__main__":
    asyncio.run(main())
