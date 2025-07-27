import re
import json
import logging
from pathlib import Path
from typing import Set, List, Dict, Any
from collections import Counter
import torch
from tqdm import tqdm
from gliner import GLiNER
from ..configs.config import TopicExtractorConfig
from ..utils.logger import logger
import nltk
from nltk.corpus import stopwords
from rapidfuzz import fuzz
from copy import deepcopy

# Ensure NLTK stopwords are available
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS: Set[str] = set(stopwords.words('english'))

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class TopicExtractor:
    def __init__(self, config: TopicExtractorConfig):
        self.config = config
        self.data_file = Path(self.config.data_file)
        device = getattr(self.config, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(self.config.gliner_model_name, device=device)
        self.threshold = self.config.gliner_threshold
        self.concurrency = self.config.concurrency
        self.output_path = self.config.output_path

        # Load labels from JSONL
        self.labels_path = Path(self.config.gliner_labels_path)
        with self.labels_path.open(encoding="utf-8") as f:
            raw_labels = [json.loads(line) for line in f]

        # Process labels - GLiNER expects simple strings, not dicts
        self.labels = self._process_labels(raw_labels)
        logger.info(f"Loaded %d unique labels", len(self.labels))

    ##################################
    #    main extraction function    #
    ##################################
    def extract_from_file(
        self,
        max_words: int = 200,
        overlap_words: int = 5,
        fuzzy_threshold: int = 90
    ) -> Dict[str, Any]:
        raw_by_label: Dict[str, List[str]] = {}
        files = list(self.data_file.glob('*.json')) if self.data_file.is_dir() else [self.data_file]
        logger.info("Processing %d file(s) for extraction", len(files))
        for filepath in tqdm(files, desc="Files processed"):
            logger.info("Processing file: %s", filepath)
            try:
                with filepath.open(encoding='utf-8') as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else [data]
                for entry in tqdm(entries, desc="Entries", leave=False):
                    text = entry.get('text_content', '')
                    cleaned = self._clean_text(text)
                    if len(cleaned) < 10:
                        continue
                    chunks = self._chunk_text_for_gliner(cleaned, max_words, overlap_words)
                    for chunk in chunks:
                        try:
                            preds = self.model.predict_entities(chunk, self.labels, threshold=self.threshold)
                            for p in preds:
                                label = p.get('label')
                                text_val = p.get('text', '').strip()
                                if label and text_val:
                                    raw_by_label.setdefault(label, []).append(text_val)
                        except Exception as e:
                            logger.warning(f"Error processing chunk: {e}")
            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")

        # Apply fuzzy clustering to get entity→count maps
        fuzzy_counts: Dict[str, Dict[str, int]] = {
            label: self._fuzzy_cluster_counts(items, threshold=fuzzy_threshold)
            for label, items in raw_by_label.items()
        }

        # Merge into existing output JSON instead of overwriting
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists():
            with out.open('r', encoding='utf-8') as f:
                existing = json.load(f)
        else:
            existing = {}

        merged = self._merge_fuzzy_counts(existing, fuzzy_counts, threshold=fuzzy_threshold)

        with out.open('w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        logger.info(f"Merged entity counts to {self.output_path}")

        # Prepare summary counts
        counts_by_label = {label: len(entities) for label, entities in merged.items()}
        total_entities = sum(counts_by_label.values())

        return {
            "total_entities": total_entities,
            "counts_by_label": counts_by_label
        }

    ########################################
    #           helper functions           #
    ########################################

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

    def _fuzzy_cluster_counts(self, items: List[str], threshold: int = 85) -> Dict[str, int]:
        clusters: List[str] = []
        counts: List[int] = []

        for item in items:
            matched = False
            for idx, rep in enumerate(clusters):
                if fuzz.ratio(item, rep) >= threshold:
                    counts[idx] += 1
                    matched = True
                    break
            if not matched:
                clusters.append(item)
                counts.append(1)

        return dict(zip(clusters, counts))

    def _merge_fuzzy_counts(
        self,
        existing: Dict[str, Dict[str, int]],
        new_counts: Dict[str, Dict[str, int]],
        threshold: int
    ) -> Dict[str, Dict[str, int]]:
        merged = deepcopy(existing)
        for label, new_map in new_counts.items():
            if label not in merged:
                merged[label] = new_map.copy()
                continue

            for ent, cnt in new_map.items():
                best_rep, best_score = None, 0
                for rep in merged[label]:
                    score = fuzz.ratio(ent, rep)
                    if score > best_score:
                        best_score, best_rep = score, rep

                if best_score >= threshold:
                    merged[label][best_rep] += cnt
                else:
                    merged[label][ent] = merged[label].get(ent, 0) + cnt

        return merged

    def extract_entities(self, text: str, max_words: int = 200, overlap_words: int = 5) -> Set[str]:
        cleaned = self._clean_text(text)
        if len(cleaned) < 5:
            return set()
        chunks = self._chunk_text_for_gliner(cleaned, max_words, overlap_words)
        entities: Set[str] = set()
        logger.info("Extracting entities from text with %d chunks", len(chunks))
        for chunk in tqdm(chunks, desc="Entity extraction chunks"):
            try:
                chunk_entities = self._predict_chunk(chunk)
                entities.update(chunk_entities)
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
        logger.info("Found %d unique entities", len(entities))
        return entities

    def analyze_entities(self, text: str, max_words: int = 200, overlap_words: int = 20) -> Dict[str, Any]:
        cleaned = self._clean_text(text)
        if len(cleaned) < 10:
            return {"total_entities": 0, "counts_by_label": {}}
        chunks = self._chunk_text_for_gliner(cleaned, max_words, overlap_words)
        all_predictions: List[Dict[str, Any]] = []
        logger.info("Analyzing entities in %d chunks", len(chunks))
        for chunk in tqdm(chunks, desc="Entity analysis chunks"):
            try:
                preds = self.model.predict_entities(chunk, self.labels, threshold=self.threshold)
                for p in preds:
                    label = p.get("label")
                    text_val = p.get("text", "").strip()
                    if label and text_val:
                        all_predictions.append({"label": label, "text": text_val})
            except Exception as e:
                logger.warning(f"Error analyzing chunk: {e}")
        total = len(all_predictions)
        counts = Counter(item["label"] for item in all_predictions)
        logger.info("Total entity occurrences: %d", total)
        return {"total_entities": total, "counts_by_label": dict(counts)}

    def save_entities_by_label(self, entities_by_label: Dict[str, Set[str]], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        hierarchical_entities = self._build_hierarchical_structure(entities_by_label)
        total_entities = sum(len(ents) for ents in entities_by_label.values())
        logger.info(f"Saved {total_entities} unique entities in hierarchical format to {output_path}")

    def load_entities_by_label(self, entities_path: str) -> Dict[str, Dict[str, int]]:
        with open(entities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = sum(len(ents) for ents in data.values())
        logger.info(f"Loaded {total} entities across {len(data)} labels from {entities_path}")
        return data

    def analyze_entities_from_file(self, entities_path: str) -> Dict[str, Any]:
        with open(entities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        counts_by_label = {label: len(ents) for label, ents in data.items()}
        total_entities = sum(counts_by_label.values())
        all_entities = set().union(*(ents.keys() for ents in data.values()))
        analytics = {
            'total_entities': total_entities,
            'unique_entities_total': len(all_entities),
            'counts_by_label': counts_by_label,
            'labels_count': len(data),
            'entities_by_label': data
        }
        logger.info(f"Analytics: {analytics['total_entities']} total entities across {analytics['labels_count']} labels")
        return analytics

    def get_entities_for_label(self, entities_path: str, label: str) -> List[str]:
        data = self.load_entities_by_label(entities_path)
        entities = list(data.get(label, {}).keys())
        logger.info(f"Found {len(entities)} entities for label '{label}'")
        return entities

    def _predict_chunk(self, chunk: str) -> Set[str]:
        try:
            preds = self.model.predict_entities(chunk, self.labels, threshold=self.threshold)
            return {p["text"].strip() for p in preds if len(p.get("text", "").strip()) > 2}
        except Exception as e:
            logger.warning(f"Error in _predict_chunk: {e}")
            return set()

    @staticmethod
    def _chunk_text_for_gliner(text: str, max_words: int, overlap_words: int) -> List[str]:
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

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.,;:!?&()\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w.lower() not in STOP_WORDS]
        return " ".join(words)

def main() -> None:
    config = TopicExtractorConfig()
    extractor = TopicExtractor(config)

    stats = extractor.extract_from_file()

    print(f"Total unique entities found: {stats['total_entities']}")
    print("Unique entities per label:")
    for label, cnt in sorted(stats['counts_by_label'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {cnt}")

if __name__ == "__main__":
    main()
