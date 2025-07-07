from .logger import logger
import json

def load_links(file_path):
    links = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    links.append(data.get("href"))
                except json.JSONDecodeError as e:
                    logger.error(f"{e}: Invalid JSON line in: {file_path} at {line}")
    except FileNotFoundError as e:
        logger.error(f"{e}: JSONL File not found: {file_path}")
    return links
