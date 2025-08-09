import re
import json
import logging
import sqlite3
from pathlib import Path
from typing import Set, List, Dict, Any
from collections import Counter, defaultdict
import torch
from tqdm import tqdm
from gliner import GLiNER
from ..configs.config import TopicExtractorConfig
from ..utils.logger import logger
import nltk
from nltk.corpus import stopwords
from rapidfuzz import fuzz
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


############################
####      Helpers     ######
############################

try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS: Set[str] = set(stopwords.words('english'))

_SG_VARIANTS = ["singapore", "singaporean", "s'pore"]
_SG_REGEX = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in _SG_VARIANTS + ["sg"]) + r")\b",
    flags=re.IGNORECASE,
)

def mentions_singapore(text: str, fuzzy_threshold: int = 75) -> bool:
    if not text:
        return False
    if _SG_REGEX.search(text):
        return True
    normalized_variants = {normalize_text(v) for v in _SG_VARIANTS}
    words = set(re.findall(r"[A-Za-z']+", text))
    for word in words:
        word_norm = normalize_text(word)
        for var_norm in normalized_variants:
            if fuzz.ratio(word_norm, var_norm) >= fuzzy_threshold:
                return True
    return False


def normalize_text(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


class TopicExtractor:
    def __init__(self, config: TopicExtractorConfig):
        self.config = config
        self.data_file = Path(self.config.data_file)
        self.output_path_json = Path(self.config.output_path)
        # Expect config.sqlite_db_path for SQLite file
        self.db_path = Path(self.config.db_path) / "kb.sqlite"

        self.model = GLiNER.from_pretrained(self.config.gliner_model_name, device="cuda")
        self.embedder = SentenceTransformer(self.config.embedding_model, device="cuda")

        self.gliner_threshold = self.config.gliner_threshold
        self.concurrency = self.config.concurrency

        # Load labels from JSONL
        self.labels_path = Path(self.config.gliner_labels_path)
        with self.labels_path.open(encoding="utf-8") as f:
            raw_labels = [json.loads(line) for line in f]
        self.labels = self._process_labels(raw_labels)
        logger.info(f"Loaded %d unique labels", len(self.labels))

        # Load abbrev map
        with open(self.config.abbrev_map_path, encoding='utf-8') as f:
            raw_map = json.load(f)
            self.abbrev_map: Dict[str, str] = {k.lower(): v for k, v in raw_map.items()}


        # Initialize or connect to SQLite
        self._init_db()

        # clients for llm
        self.cleaner_client = OpenAI(
            base_url=self.config.cleaner_base_url,
            api_key="dummy"
        )

        self.validator_client = OpenAI(
            base_url=self.config.validator_base_url,
            api_key="dummy"
        )

    ########################################
    #          database functions          #
    ########################################

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Create tables
        c.execute("""
        CREATE TABLE IF NOT EXISTS labels (
          label_id    INTEGER PRIMARY KEY AUTOINCREMENT,
          label       TEXT NOT NULL UNIQUE
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS entities (
          entity_id      INTEGER PRIMARY KEY AUTOINCREMENT,
          label_id       INTEGER NOT NULL,
          entity_text    TEXT NOT NULL,
          total_count    INTEGER NOT NULL DEFAULT 0,
          UNIQUE(label_id, entity_text),
          FOREIGN KEY(label_id) REFERENCES labels(label_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
          entity_id   INTEGER PRIMARY KEY,
          vector      TEXT NOT NULL,
          FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS cooccurrence (
          entity_a     INTEGER NOT NULL,
          entity_b     INTEGER NOT NULL,
          count        INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (entity_a, entity_b),
          FOREIGN KEY(entity_a) REFERENCES entities(entity_id),
          FOREIGN KEY(entity_b) REFERENCES entities(entity_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS label_coverage (
          label_id        INTEGER PRIMARY KEY,
          docs_with_label INTEGER NOT NULL DEFAULT 0,
          FOREIGN KEY(label_id) REFERENCES labels(label_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
          key   TEXT PRIMARY KEY,
          value TEXT
        )""")
        conn.commit()
        conn.close()


    def _save_to_db(self, data: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Metadata
        c.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", ('total_docs', str(data['total_docs'])))
        # Labels
        for label, ents in data['counts'].items():
            c.execute("INSERT OR IGNORE INTO labels(label) VALUES(?)", (label,))
        # Entities and embeddings
        for label, ents in data['counts'].items():
            c.execute("SELECT label_id FROM labels WHERE label=?", (label,))
            label_id = c.fetchone()[0]
            for ent, cnt in ents.items():
                c.execute(
                    "INSERT OR IGNORE INTO entities(label_id,entity_text,total_count) VALUES(?,?,?)", 
                    (label_id, ent, cnt)
                )
                c.execute(
                    "UPDATE entities SET total_count=? WHERE label_id=? AND entity_text=?", 
                    (cnt, label_id, ent)
                )
                c.execute("SELECT entity_id FROM entities WHERE label_id=? AND entity_text=?", (label_id, ent))
                entity_id = c.fetchone()[0]
                vec = json.dumps(data['embeddings'][label][ent])
                c.execute(
                    "INSERT OR REPLACE INTO embeddings(entity_id,vector) VALUES(?,?)", 
                    (entity_id, vec)
                )
        # Co-occurrence
        for e1, neigh in data['co_occurrence'].items():
            c.execute("SELECT entity_id FROM entities WHERE entity_text=?", (e1,))
            row = c.fetchone()
            if not row: continue
            id1 = row[0]
            for e2, cnt in neigh.items():
                c.execute("SELECT entity_id FROM entities WHERE entity_text=?", (e2,))
                row2 = c.fetchone()
                if not row2: continue
                id2 = row2[0]
                a, b = (id1, id2) if id1 < id2 else (id2, id1)
                c.execute(
                    "INSERT OR IGNORE INTO cooccurrence(entity_a,entity_b,count) VALUES(?,?,?)", 
                    (a, b, cnt)
                )
                c.execute(
                    "UPDATE cooccurrence SET count=? WHERE entity_a=? AND entity_b=?", 
                    (cnt, a, b)
                )
        # Coverage
        for label, cov in data['coverage_by_label'].items():
            c.execute("SELECT label_id FROM labels WHERE label=?", (label,))
            lid = c.fetchone()[0]
            docs = int(cov * data['total_docs'])
            c.execute(
                "INSERT OR REPLACE INTO label_coverage(label_id,docs_with_label) VALUES(?,?)", 
                (lid, docs)
            )
        conn.commit()
        conn.close()

    ########################################
    #            core functions            #
    ########################################

    def extract_from_file(self, max_words: int = 275, overlap_words: int = 15, fuzzy_threshold: int = 75) -> Dict[str, Any]:
        raw_by_label       = defaultdict(list)
        co_occurrence      = defaultdict(lambda: defaultdict(int))
        docs_with_label    = defaultdict(int)
        total_docs         = 0

        files = list(self.data_file.glob('*.json')) \
                if self.data_file.is_dir() else [self.data_file]
        logger.info("Processing %d file(s) for extraction", len(files))

        relevance_cache = {}

        for filepath in tqdm(files, desc="Files processed"):
            try:
                data = json.load(filepath.open(encoding='utf-8'))
                entries = data if isinstance(data, list) else [data]

                # skip file if no SG mention
                if not any(mentions_singapore(e.get("text_content",""), fuzzy_threshold)
                           for e in entries):
                    continue

                for entry in entries:
                    text = entry.get('text_content','') or ''
                    if len(text) < 10 or not mentions_singapore(text, fuzzy_threshold):
                        continue

                    entry_entities          = set()
                    entry_entities_by_label = defaultdict(set)
                    chunks = self._chunk_text_for_gliner(text, max_words, overlap_words)

                    # --- begin parallel chunk processing ---
                    futures = {}
                    with ThreadPoolExecutor(max_workers=self.concurrency) as exe:
                        for chunk in chunks:
                            futures[ exe.submit(self._process_chunk, chunk) ] = chunk

                        for fut in tqdm(as_completed(futures),
                                        total=len(futures),
                                        desc=f"Chunks in {filepath.name}",
                                        leave=False):
                            try:
                                preds = fut.result()
                                for p in preds:
                                    label   = p.get("label")
                                    text_val= p.get("text","").strip()
                                    if not label or not text_val:
                                        continue

                                    # map to long form if it's a known abbreviation
                                    mapped = self.abbrev_map.get(text_val.lower(), text_val)

                                    # check relevance before adding
                                    if mapped not in relevance_cache:
                                        relevance_cache[mapped] = self._check_relevance(mapped)
                                    if relevance_cache[mapped]:
                                        raw_by_label[label].append(mapped)
                                        entry_entities.add(mapped)
                                        entry_entities_by_label[label].add(mapped)

                            except Exception as e:
                                logger.warning(f"Error in parallel chunk: {e}")
                    # --- end parallel chunk processing ---

                    if entry_entities:
                        total_docs += 1
                        from itertools import combinations
                        for e1, e2 in combinations(entry_entities, 2):
                            co_occurrence[e1][e2] += 1
                            co_occurrence[e2][e1] += 1

                        for label, ents in entry_entities_by_label.items():
                            if ents:
                                docs_with_label[label] += 1

            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")

        # Cluster and count
        fuzzy_counts = {
            label: self._fuzzy_cluster_counts(items, threshold=fuzzy_threshold)
            for label, items in raw_by_label.items()
        }

        # Merge with existing JSON (optional)
        out_json = self.output_path_json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if out_json.exists():
            with out_json.open('r', encoding='utf-8') as f:
                existing = json.load(f)
        merged_counts = self._merge_fuzzy_counts(existing.get('counts', {}), fuzzy_counts, threshold=fuzzy_threshold)

        # Compute embeddings
        embeddings = {}
        for label, ents in merged_counts.items():
            embeddings[label] = {}
            for ent in ents:
                embeddings[label][ent] = self.embedder.encode(ent).tolist()

        # Compute per-label distribution summing to 100%
        docs_count_by_label = {label: docs_with_label.get(label, 0) for label in merged_counts}
        coverage_by_label = {
            label: docs_with_label.get(label, 0) / total_docs
            for label in merged_counts
        }

        final_output = {
            'counts': merged_counts,
            'embeddings': embeddings,
            'co_occurrence': {e: dict(neighbors) for e, neighbors in co_occurrence.items()},
            'total_docs': total_docs,
            'coverage_by_label': coverage_by_label,
        }

        # Save to JSON
        with out_json.open('w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        # Save to SQLite
        self._save_to_db(final_output)

        # Summary
        unique_entities = sum(len(ents) for ents in merged_counts.values())
        return {
            'total_entities': unique_entities,
            'total_docs': total_docs,
            'counts_by_label': {l: len(v) for l, v in merged_counts.items()},
            'coverage_by_label': coverage_by_label
        }

    ########################################
    #           helper functions           #
    ########################################
    
    def _check_relevance(self, entity: str) -> bool:
        messages = [
            {"role": "system", "content": "You are given an entity in the form of a phrase, topic or word. Respond with 'yes' or 'no' depending on whether the entity is related to Singapore in any way."},
            {"role": "user", "content": entity}
        ]
        # resp = self.cleaner_client.chat.completions.create(
        #     model=self.config.cleaner_model_name,
        #     messages=messages,
        #     temperature=0  # Deterministic
        # )
        # return resp.choices[0].message.content.strip().lower() == "yes"
        resp = self.cleaner_client.chat.completions.create(
            model=self.config.cleaner_model_name,
            messages=messages,
            n=3,
            temperature=0.7  # or even 1.0 for max variation
        )
        responses = [choice.message.content.strip().lower() for choice in resp.choices]
        return responses.count("yes") > responses.count("no")


    def _process_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        """Helper to clean and then run GLiNER on a single chunk."""
        cleaned = self._clean_text_llm(chunk)
        return self.model.predict_entities(cleaned, self.labels, threshold=self.gliner_threshold)


    def _clean_text_llm(self, text):
        messages = [
            {"role": "system", "content": "You are given a text scraped from a website, clean it to be suitable for entity extraction. Give the cleaned text directly with no explanation"},
            {"role": "user", "content": text}
        ]
        resp = self.cleaner_client.chat.completions.create(
            model=self.config.cleaner_model_name,
            messages=messages,
        )
        return resp.choices[0].message.content


    def _process_labels(self, raw_labels: List[Dict]) -> List[str]:
        processed_labels: List[str] = []
        seen_labels: Set[str] = set()
        for label_dict in raw_labels:
            if isinstance(label_dict, dict):
                label_name = label_dict.get('label') or label_dict.get('name') or str(label_dict)
            else:
                label_name = str(label_dict)
            if label_name not in seen_labels:
                processed_labels.append(label_name)
                seen_labels.add(label_name)
        return processed_labels


    def _fuzzy_cluster_counts(self, items: List[str], threshold: int = 75) -> Dict[str, int]:
        clusters: List[str] = []
        counts: List[int] = []

        for item in items:
            item_norm = normalize_text(item)
            matched = False
            for idx, rep in enumerate(clusters):
                rep_norm = normalize_text(rep)
                if fuzz.ratio(item_norm, rep_norm) >= threshold:
                    counts[idx] += 1
                    matched = True
                    break
            if not matched:
                clusters.append(item)
                counts.append(1)

        return dict(zip(clusters, counts))


    def _merge_fuzzy_counts(self, existing: Dict[str, Dict[str, int]], new_counts: Dict[str, Dict[str, int]], threshold: int) -> Dict[str, Dict[str, int]]:
        merged = deepcopy(existing)
        for label, new_map in new_counts.items():
            if label not in merged:
                merged[label] = new_map.copy()
                continue

            for ent, cnt in new_map.items():
                ent_norm = normalize_text(ent)
                best_rep, best_score = None, 0
                for rep in merged[label]:
                    rep_norm = normalize_text(rep)
                    score = fuzz.ratio(ent_norm, rep_norm)
                    if score > best_score:
                        best_score, best_rep = score, rep

                if best_score >= threshold:
                    merged[label][best_rep] += cnt
                else:
                    merged[label][ent] = merged[label].get(ent, 0) + cnt

        return merged
    

    def _chunk_text_for_gliner(self, text: str, max_words: int, overlap_words: int) -> List[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap_words
        return chunks


def main() -> None:
    config = TopicExtractorConfig()
    extractor = TopicExtractor(config)
    stats = extractor.extract_from_file()
    print(f"Total unique entities found: {stats['total_entities']}")
    print(f"Processed documents: {stats['total_docs']}")
    for label, cnt in sorted(stats['counts_by_label'].items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {cnt}")

if __name__ == "__main__":
    main()
