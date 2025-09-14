import re
import json
import sqlite3
from pathlib import Path
from typing import Set, List, Dict, Any, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
from ..configs.config import TopicExtractorConfig
from ..utils.logger import logger
import numpy as np
from rapidfuzz import fuzz
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.utils import mentions_singapore, normalize_text


class TopicExtractor:
    def __init__(self, config: TopicExtractorConfig):
        self.config = config
        self.data_file = Path(self.config.data_file)
        self.output_path_json = Path(self.config.output_path)

        ## ------------ inits ------------- ##
        self.db_path = Path(self.config.db_path) / "kb.sqlite"                                   ## database
                
        self.ner_client = OpenAI(base_url = self.config.ner_base_url, api_key = "dummy")         ## ner models
        self.ner_model_name = self.config.ner_model_name                                         ## ner model 
        self.embedder = SentenceTransformer(self.config.embedding_model, device="cuda")          ## embeddings models
        self.concurrency = self.config.concurrency                                               ## concurrency
        self.cleaner_client = OpenAI(base_url=self.config.cleaner_base_url, api_key="dummy")     ## cleaner model
        self.validator_client = OpenAI(base_url=self.config.validator_base_url, api_key="dummy") ## validator model

        ## ----------- seeds -------------- ##
        with open(self.config.abbrev_map_path, encoding='utf-8') as f:                           ## abbreviations
            raw_map = json.load(f)
            self.abbrev_map: Dict[str, str] = {k.lower(): v for k, v in raw_map.items()}

        with Path(self.config.labels_path).open(encoding="utf-8") as f:                          ## labels
            raw_labels = [json.loads(line) for line in f]

        # UNPACK the tuple returned by _process_labels into two variables
        self.labels_for_validation, self.labels_for_prompt = self._process_labels(raw_labels)

        self.seed_entities_file = Path(self.config.seed_entities_file)
        logger.info(f"Loaded %d unique labels and %d abbreviations", len(self.labels_for_validation), len(self.abbrev_map))

    ########################################
    #          database functions          #
    ########################################

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # ----------------------------------------------------------------------
        # 1. Core tables (labels, entities, embeddings, co-occurrence, coverage)
        # ----------------------------------------------------------------------
        c.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            label_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label    TEXT NOT NULL UNIQUE
        )""")

        # NOTE: the column `is_seed` is new; for fresh DBs it is created here.
        c.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            label_id    INTEGER NOT NULL,
            entity_text TEXT NOT NULL,
            total_count INTEGER NOT NULL DEFAULT 0,
            is_seed     INTEGER NOT NULL DEFAULT 0,
            UNIQUE(label_id, entity_text),
            FOREIGN KEY(label_id) REFERENCES labels(label_id)
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            entity_id INTEGER PRIMARY KEY,
            vector    TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS cooccurrence (
            entity_a INTEGER NOT NULL,
            entity_b INTEGER NOT NULL,
            count    INTEGER NOT NULL DEFAULT 0,
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

        # ----------------------------------------------------------------------
        # 2. One-time schema-upgrade helper: add `is_seed` to old DBs
        # ----------------------------------------------------------------------
        try:
            c.execute("SELECT is_seed FROM entities LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE entities ADD COLUMN is_seed INTEGER NOT NULL DEFAULT 0")

        # ----------------------------------------------------------------------
        # 3. Load seed entities (if file provided)
        # ----------------------------------------------------------------------
        conn.commit()  # commit schema changes before starting seed load

        if self.seed_entities_file and self.seed_entities_file.exists():
            logger.info("Loading seed entities from %s...", self.seed_entities_file)
            try:
                seed_data = self._load_seed_entities(self.seed_entities_file)
                self._insert_seed_entities(c, seed_data)
                conn.commit()
                logger.info("Seed entities loaded successfully.")
            except Exception as e:
                logger.error("Error loading seed entities: %s", e)
                conn.rollback()
        else:
            logger.info("No seed entities file provided or file does not exist.")

        conn.close()

    def _load_seed_entities(self, seed_path: Path) -> List[Dict[str, str]]:
        """
        Reads a JSONL file containing seed entities.
        Each line should be a JSON object with 'entity' and 'label' keys.
        Returns a list of dictionaries { 'entity': str, 'label': str }.
        """
        seed_entities = []
        try:
            with seed_path.open(encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines
                    try:
                        obj = json.loads(line)
                        entity = obj.get('entity')
                        label = obj.get('label')
                        if entity is not None and label is not None:
                            seed_entities.append({'entity': entity.strip(), 'label': label.strip()})
                        else:
                            logger.warning(f"Skipping invalid line {line_num} in seed file: {line}")
                    except json.JSONDecodeError as je:
                        logger.warning(f"Invalid JSON on line {line_num} in seed file: {line}. Error: {je}")
        except Exception as e:
            raise RuntimeError(f"Error reading seed entities file '{seed_path}': {e}")
        return seed_entities

    def _insert_seed_entities(self, cursor, seed_data: List[Dict[str, str]]) -> None:
        """
        Bulk-insert seed entities with efficient batch operations.
        Assumes seed_data is list of {'entity': str, 'label': str}
        """
        if not seed_data:
            logger.info("No seed entities to insert.")
            return

        logger.info(f"Inserting {len(seed_data)} seed entities in bulk...")

        # Step 1: Extract unique labels and entities
        unique_labels = set(item['label'].strip() for item in seed_data)
        label_to_id = {}

        # Step 2: Insert all unique labels in one go
        cursor.executemany(
            "INSERT OR IGNORE INTO labels(label) VALUES(?)",
            [(label,) for label in unique_labels]
        )

        # Step 3: Fetch all label_ids in one query
        cursor.execute("SELECT label_id, label FROM labels WHERE label IN ({})".format(
            ",".join("?" * len(unique_labels))
        ), list(unique_labels))
        label_to_id = {label: lid for lid, label in cursor.fetchall()}

        # Step 4: Prepare entity data (label_id, entity_text, is_seed=1)
        entity_data = []
        skipped_invalid = 0
        for item in seed_data:
            label_name = item['label'].strip()
            entity_text = item['entity'].strip()
            if not label_name or not entity_text:
                skipped_invalid += 1
                continue
            if label_name in label_to_id:
                entity_data.append((label_to_id[label_name], entity_text))

        if skipped_invalid:
            logger.warning(f"Skipped {skipped_invalid} invalid seed entries (empty label/entity).")

        if not entity_data:
            logger.warning("No valid seed entities after filtering.")
            return

        # Step 5: Bulk insert entities (ignore duplicates)
        cursor.executemany("""
            INSERT OR IGNORE INTO entities(label_id, entity_text, total_count, is_seed)
            VALUES (?, ?, 0, 1)
        """, [(lid, text) for lid, text in entity_data])

        # Step 6: Get all inserted entity_ids (by label_id + entity_text)
        entity_keys = entity_data  # list of (label_id, entity_text)
        cursor.execute("""
            SELECT e.entity_id, e.label_id, e.entity_text
            FROM entities e
            WHERE (e.label_id, e.entity_text) IN ({})
        """.format(",".join(" (?,?)" for _ in entity_keys)),
            [item for pair in entity_keys for item in pair]  # flatten list of tuples
        )
        entity_map = {(lid, text): eid for eid, lid, text in cursor.fetchall()}

        # Step 7: Precompute ALL embeddings in one batch (FAST!)
        entity_texts = [text for _, text in entity_keys]
        logger.info(f"Encoding {len(entity_texts)} seed entity texts in batch...")
        
        embedding_data = []  # <-- DEFINE IT HERE, OUTSIDE TRY BLOCK!
        try:
            # Encode all at once — this is MUCH faster than looping
            embeddings = self.embedder.encode(
                entity_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embedding_data = [
                (entity_map[(lid, text)], json.dumps(vec.tolist()))
                for (lid, text), vec in zip(entity_keys, embeddings)
            ]
            logger.info(f"✅ Successfully encoded {len(embedding_data)} embeddings.")

        except Exception as e:
            logger.error(f"Failed to encode seed entities: {e}")
            raise  # Re-raise to fail fast — better than silent failure

        # Step 8: Bulk insert embeddings (ignore if already exist)
        if embedding_data:  # Only insert if we have embeddings
            cursor.executemany("""
                INSERT OR IGNORE INTO embeddings(entity_id, vector)
                VALUES (?, ?)
            """, embedding_data)
            logger.info(f"✅ Successfully inserted {len(embedding_data)} seed entities with embeddings.")
        else:
            logger.warning("No embeddings to insert.")

    def _save_to_db(self, data: Dict[str, Any]) -> None:
        """
        Saves extracted topic data to the SQLite database using bulk operations for performance.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # --- PRE-CACHE ALL SEED ENTITIES (TEXT + LABEL) FOR FAST CHECK ---
        c.execute("""
            SELECT e.entity_text, l.label
            FROM entities e
            JOIN labels l ON e.label_id = l.label_id
            WHERE e.is_seed = 1
        """)
        seed_entities = {(ent.lower().strip(), label.lower().strip()) for ent, label in c.fetchall()}
        conn.close()
        # Reopen connection for writing
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Start a single transaction for all operations
        conn.execute("BEGIN")
        try:
            # Metadata
            c.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", ('total_docs', str(data['total_docs'])))
            # --- BULK OPERATIONS START ---
            # Prepare lists for bulk inserts/updates
            label_inserts = []          # For new labels
            entity_inserts = []         # For new entities (INSERT OR IGNORE)
            entity_updates = []         # For updating entity counts
            embedding_inserts = []      # For new/updated embeddings
            cooccurrence_inserts = []   # For new co-occurrences (INSERT OR IGNORE)
            cooccurrence_updates = []   # For updating co-occurrence counts
            coverage_inserts = []       # For label coverage (INSERT OR REPLACE)
            # Process Labels and Entities
            for label, ents in data['counts'].items():
                # Add label for bulk insert (duplicates will be ignored)
                label_inserts.append((label,))
                # Get label_id (we'll fetch them all after bulk insert)
                # We'll handle this in a separate step after inserting labels
                for ent, cnt in ents.items():
                    ent_clean = ent.strip()
                    label_clean = label.strip()
                    # Skip if this exact entity+label is already a seed
                    if (ent_clean.lower(), label_clean.lower()) in seed_entities:
                        logger.debug(f"Skipping insertion of '{ent}' (label: '{label}') — already a seed entity.")
                        continue
                    # Optional: Semantic/Fuzzy check against ALL seeds
                    seed_texts, seed_vecs = self._get_seed_entity_vectors()
                    if seed_texts:
                        cand_vec = self.embedder.encode([ent_clean]).astype(np.float32)
                        sims = util.cos_sim(cand_vec, seed_vecs.astype(np.float32))[0]
                        best_sim = float(sims.max())
                        if best_sim >= 0.85:
                            matched_seed = seed_texts[int(sims.argmax())]
                            logger.debug(f"Skipping '{ent}' (label: '{label}') — semantically matches seed: '{matched_seed}' (sim: {best_sim:.3f})")
                            continue
                        for seed_text in seed_texts:
                            if fuzz.ratio(ent_clean.lower(), seed_text.lower()) >= 85:
                                logger.debug(f"Skipping '{ent}' (label: '{label}') — fuzzy matches seed: '{seed_text}'")
                                continue
                    # Prepare data for bulk operations
                    # We need label_id, so we'll collect (label, ent_clean, cnt) and resolve label_id later
                    entity_inserts.append((label, ent_clean, cnt))  # Temporarily store label text
                    entity_updates.append((cnt, label, ent_clean))  # For UPDATE
            # Bulk insert labels first
            if label_inserts:
                c.executemany("INSERT OR IGNORE INTO labels(label) VALUES(?)", label_inserts)
            # Now, fetch all label_ids in one go
            unique_labels = list(set(label for label, _ in data['counts'].items()))
            label_to_id = {}
            if unique_labels:
                placeholders = ','.join('?' * len(unique_labels))
                c.execute(f"SELECT label_id, label FROM labels WHERE label IN ({placeholders})", unique_labels)
                label_to_id = {label: lid for lid, label in c.fetchall()}
            # Now, resolve label_id for entities and prepare final lists
            final_entity_inserts = []
            final_entity_updates = []
            final_embedding_inserts = []
            for label, ent_clean, cnt in entity_inserts:
                label_id = label_to_id.get(label)
                if label_id is None:
                    logger.warning(f"Could not find label_id for label '{label}' during entity insert. Skipping entity '{ent_clean}'.")
                    continue
                final_entity_inserts.append((label_id, ent_clean, cnt))
            for cnt, label, ent_clean in entity_updates:
                label_id = label_to_id.get(label)
                if label_id is None:
                    continue
                final_entity_updates.append((cnt, label_id, ent_clean))
            # Bulk insert entities
            if final_entity_inserts:
                c.executemany(
                    "INSERT OR IGNORE INTO entities(label_id, entity_text, total_count, is_seed) VALUES(?,?,?,0)",
                    final_entity_inserts
                )
            # Bulk update entity counts
            if final_entity_updates:
                c.executemany(
                    "UPDATE entities SET total_count=? WHERE label_id=? AND entity_text=?",
                    final_entity_updates
                )

            # Create a set of all (label_id, entity_text) keys for which we need embeddings
            all_entity_keys = set()
            for label, ents in data['counts'].items():
                for ent, cnt in ents.items():
                    ent_clean = ent.strip()
                    label_clean = label.strip()
                    if (ent_clean.lower(), label_clean.lower()) in seed_entities:
                        continue
                    label_id = label_to_id.get(label)
                    if label_id is None:
                        continue
                    all_entity_keys.add((label_id, ent_clean))
            # Now, fetch entity_ids for ALL these keys
            entity_keys = list(all_entity_keys) # <-- Use the comprehensive set built above
            entity_map = {}
            if entity_keys:
                # Create a list of placeholders for the query
                placeholders = ','.join('(?,?)' for _ in entity_keys)
                params = [item for pair in entity_keys for item in pair]  # Flatten the list
                c.execute(f"""
                    SELECT e.entity_id, e.label_id, e.entity_text
                    FROM entities e
                    WHERE (e.label_id, e.entity_text) IN ({placeholders})
                """, params)
                entity_map = {(lid, text): eid for eid, lid, text in c.fetchall()}  

            # Prepare embedding data
            for label, ents in data['counts'].items():
                for ent, cnt in ents.items():
                    ent_clean = ent.strip()
                    label_clean = label.strip()
                    if (ent_clean.lower(), label_clean.lower()) in seed_entities:
                        continue
                    # Re-check semantic/fuzzy if needed (you might want to cache this result from earlier)
                    # For simplicity, assuming we passed the checks above.
                    label_id = label_to_id.get(label)
                    if label_id is None:
                        continue
                    entity_id = entity_map.get((label_id, ent_clean))
                    if entity_id is None:
                        logger.warning(f"Could not find entity_id for '{ent_clean}' (label: '{label}') during embedding insert.")
                        continue
                    vec = json.dumps(data['embeddings'][label][ent])
                    final_embedding_inserts.append((entity_id, vec))
            # Bulk insert embeddings
            if final_embedding_inserts:
                c.executemany(
                    "INSERT OR REPLACE INTO embeddings(entity_id,vector) VALUES(?,?)",
                    final_embedding_inserts
                )
            # Co-occurrence (Bulk Operations)
            cooccurrence_data = []
            for e1, neigh in data['co_occurrence'].items():
                c.execute("SELECT entity_id FROM entities WHERE entity_text=?", (e1,))
                row = c.fetchone()
                if not row:
                    continue
                id1 = row[0]
                for e2, cnt in neigh.items():
                    c.execute("SELECT entity_id FROM entities WHERE entity_text=?", (e2,))
                    row2 = c.fetchone()
                    if not row2:
                        continue
                    id2 = row2[0]
                    a, b = (id1, id2) if id1 < id2 else (id2, id1)
                    cooccurrence_data.append((a, b, cnt))
            # Separate into inserts and updates
            if cooccurrence_data:
                # For simplicity, we do INSERT OR IGNORE followed by UPDATE.
                # A more advanced approach would be to use INSERT ... ON CONFLICT, but SQLite's support varies.
                c.executemany(
                    "INSERT OR IGNORE INTO cooccurrence(entity_a,entity_b,count) VALUES(?,?,?)",
                    cooccurrence_data
                )
                c.executemany(
                    "UPDATE cooccurrence SET count=? WHERE entity_a=? AND entity_b=?",
                    [(cnt, a, b) for a, b, cnt in cooccurrence_data]
                )
            # Coverage (Bulk Operations)
            for label, cov in data['coverage_by_label'].items():
                c.execute("SELECT label_id FROM labels WHERE label=?", (label,))
                lid = c.fetchone()
                if lid:
                    docs = int(cov * data['total_docs'])
                    coverage_inserts.append((lid[0], docs))
            if coverage_inserts:
                c.executemany(
                    "INSERT OR REPLACE INTO label_coverage(label_id,docs_with_label) VALUES(?,?)",
                    coverage_inserts
                )
            # Commit the transaction
            conn.commit()
            logger.info("Successfully saved data to database with bulk operations.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error during bulk save to DB: {e}")
            raise
        finally:
            conn.close()

    @lru_cache(maxsize=1)
    def _get_seed_entity_vectors(self) -> Tuple[List[str], np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT e.entity_text, em.vector
            FROM entities e
            JOIN embeddings em ON e.entity_id = em.entity_id
            WHERE e.is_seed = 1
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            dim = self.embedder.get_sentence_embedding_dimension()
            return [], np.empty((0, dim))

        texts, vecs = zip(*rows)
        return list(texts), np.array([json.loads(v) for v in vecs])
    
    def _get_all_entity_vectors(self) -> Dict[str, np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT e.entity_text, em.vector
            FROM entities e
            JOIN embeddings em ON e.entity_id = em.entity_id
        """)
        rows = cur.fetchall()
        conn.close()
        return {t: np.array(json.loads(v)) for t, v in rows}

    def _get_all_seed_entities(self) -> List[str]:
        """Get all seed entities from the seed entities file"""
        if not self.seed_entities_file or not self.seed_entities_file.exists():
            return []
        
        seed_entities = []
        try:
            with self.seed_entities_file.open(encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        entity = obj.get('entity')
                        if entity is not None:
                            seed_entities.append(entity.strip())
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return seed_entities


    ########################################
    #           helper functions           #
    ########################################
    
    def _check_relevance(self, entity: str, *, sem_threshold: float = .90) -> str | None:
        entity_lower = entity.lower().strip()

        # 1. fuzzy against seed
        for seed_text in self._get_all_seed_entities():   
            if fuzz.ratio(entity_lower, seed_text.lower()) >= 85:
                # logger.debug(f"Fuzzy match found for '{entity}' against seed: '{seed_text}'")
                return entity  # <-- Return the candidate

        # 2. semantic similarity using cached vectors
        try:
            # encode the candidate only once
            cand_vec = self.embedder.encode([entity])
            cand_vec = cand_vec.astype(np.float32)

            # start with seed vectors (fast, already cached)
            seed_texts, seed_vecs = self._get_seed_entity_vectors()
            seed_vecs = seed_vecs.astype(np.float32)
            if seed_vecs.shape[0]:
                sims = util.cos_sim(cand_vec, seed_vecs)[0]
                best = float(sims.max())
                if best >= sem_threshold:
                    matched_text = seed_texts[int(sims.argmax())]
                    # logger.info(f"Semantic match found for '{entity}' against seed: '{matched_text}' (similarity: {best:.3f})")
                    return entity  # <-- Return the candidate instead of matched_text

            # widen to *all* entities (seed + extracted)
            all_vecs = self._get_all_entity_vectors()
            if not all_vecs:
                raise ValueError("No vectors in DB")

            texts, vecs = zip(*all_vecs.items())
            vecs = np.stack(vecs).astype(np.float32)
            sims = util.cos_sim(cand_vec, np.stack(vecs))[0]
            best = float(sims.max())
            if best >= sem_threshold:
                matched_text = texts[int(sims.argmax())]
                # logger.info(f"Semantic match found for '{entity}' against DB entity: '{matched_text}' (similarity: {best:.3f})")
                return entity  # <-- Return the candidate

        except Exception as e:
            logger.warning("Semantic check failed for '%s': %s", entity, e)

        # 4. LLM fallback
        context = (
            "You are given an entity phrase. Reply 'yes' or 'no' depending on whether it is related to Singapore."
            "Only reply no if you are certain it is unrelated to Singapore, be it directly or indirectly."
            "If you are unsure, say yes."
        )

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": entity}
        ]
        resp = self.validator_client.chat.completions.create(
            model=self.config.validator_model_name,
            messages=messages,
            n=3,
            temperature=0.6,
            max_completion_tokens=3
        )
        yes_votes = sum(1 for c in resp.choices if c.message.content.strip().lower().startswith("yes"))
        result = entity if yes_votes > 1 else None  # <-- Already returning entity
        if result is not None:
            logger.debug(f"LLM fallback returned positive match for '{entity}'")
        else:
            logger.debug(f"LLM fallback returned no match for '{entity}'")
        return result

    def _process_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        """Helper to clean and then run GLiNER on a single chunk."""
        cleaned = self._clean_text_llm(chunk)
        output = self._perform_ner_with_llm(cleaned)
        return output

    def _clean_text_llm(self, text):
        messages = [
            {"role": "system", "content": "You are given a text scraped from a website, clean the text by forming readable sentences as well as removing malformed text. Focus on extracting and forming the actual content, so that it can be be suitable for entity extraction. Give the cleaned text directly with no explanation"},
            {"role": "user", "content": text}
        ]
        resp = self.cleaner_client.chat.completions.create(
            model=self.config.cleaner_model_name,
            messages=messages,
        )
        return resp.choices[0].message.content

    def _process_labels(self, raw_labels: List[Dict]):
        try:
            # Read the file content as a single string for the prompt
            with Path(self.config.labels_path).open(encoding='utf-8') as f:
                prompt_string = ''.join(f.readlines()).strip()

            # Create a list of label names for validation from the loaded `raw_labels`
            validation_list = [item.get("label", "").strip() for item in raw_labels if item.get("label")]

            return (validation_list, prompt_string)

        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.config.labels_path}")
            return ([], "")
        except Exception as e:
            logger.error(f"Error processing labels file: {e}")
            return ([], "")

    def _fuzzy_cluster_counts(self, items: List[str], threshold: int) -> Dict[str, int]:
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
    
    def _chunk_text_for_NER(self, text: str, max_words: int, overlap_words: int) -> List[str]:
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

    def _perform_ner_with_llm(self, text_chunk: str) -> List[Dict[str, Any]]:
        ## Uses an LLM to perform Named Entity Recognition on a text chunk, returns a list of dictionaries similar to GLiNER output.
        if not text_chunk.strip():
            return []

        # Handle labels formatting
        formatted_labels = self.labels_for_prompt

        system_prompt = """
        ## Context
        You are an expert assistant helping to build a knowledge base about Singapore. 
        Your task is to read a provided text and identify specific named entities that are directly related to Singapore. 
        This includes people, places, organizations, events, landmarks, policies, historical periods, cultural aspects, and other concrete nouns that contribute to understanding Singapore's context.

        You will be given a list of entity type labels. For each entity you identify, you MUST assign it the SINGLE most appropriate label from this list. 
        You MUST ONLY use labels specified below; do not invent, modify, or substitute any labels. 
        The labels are: {labels}.

        ## Instructions:
        1. Focus ONLY on entities explicitly mentioned in the text that are relevant to Singapore.
        2. Do NOT include generic terms, pronouns, or abstract concepts unless they are part of a specific named entity (e.g., 'Singaporean cuisine' is acceptable if 'cultural aspect' is a label, but 'policy' alone is not).
        3. Extract the entity text EXACTLY as it appears in the input text, including capitalization, punctuation, and spacing.
        4. Assign the MOST SPECIFIC and ACCURATE label from the provided list to each entity. If an entity could fit multiple labels, choose the one that best captures its primary context (e.g., 'National University of Singapore' → 'Organization', not 'Location').
        5. Respond ONLY with a valid JSON array. Each item in the array must be a JSON object with two keys:
        - 'label': The assigned label from the list.
        - 'text': The exact entity text found in the input.
        6. If no relevant Singapore entities are found according to these criteria, return an empty JSON array [].
        7. Do not include any explanations, markdown formatting (like ```json), comments, or additional text outside the JSON array.
        8. Only use English

        ## Example Output Format (if entities are found):
        [{{"label": "Person", "text": "Lee Kuan Yew"}}, {{"label": "Location", "text": "Marina Bay Sands"}}, {{"label": "Organization", "text": "Housing and Development Board"}}]
        
        ## Example Output Format (if no entities are found):
        []
        """
        
        # Use format with escaped braces
        system_prompt = system_prompt.format(labels=formatted_labels)

        user_prompt = "Perform entity extraction for this text chunk:\n" + text_chunk
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.ner_client.chat.completions.create(
            model=self.ner_model_name,
            messages=messages,
            temperature=0,
            top_p=0.8,
            max_completion_tokens=4096,
            frequency_penalty=1.0,
            presence_penalty=0.6
        )
        raw_response = response.choices[0].message.content.strip()
        entities_found = json.loads(raw_response)
        if not isinstance(entities_found, list):
            logger.warning(f"LLM NER response is not a list: {raw_response}")
            return []

        validated_entities = []
        for item in entities_found:
            if isinstance(item, dict) and 'label' in item and 'text' in item:
                if item['label'] in self.labels_for_validation:
                    validated_entities.append(item)
                else:
                    logger.debug(f"LLM returned entity with unexpected label '{item.get('label')}', ignoring. Entity: {item}")
            else:
                logger.warning(f"LLM NER response item is invalid: {item}")

        logger.debug(f"LLM NER found {len(validated_entities)} entities in chunk.")
        return validated_entities
    
    ########################################
    #            core functions            #
    ########################################

    def extract_from_file(self, max_words: int = 450, overlap_words: int = 50, fuzzy_threshold: int = 85) -> Dict[str, Any]:
        raw_by_label       = defaultdict(list)
        co_occurrence      = defaultdict(lambda: defaultdict(int))
        docs_with_label    = defaultdict(int)
        total_docs         = 0

        files = list(self.data_file.glob('*.json')) if self.data_file.is_dir() else [self.data_file]
        logger.info("Processing %d file(s) for extraction", len(files))

        # === Step 1: Estimate total number of chunks across all files ===
        total_chunks = 0
        for filepath in files:
            try:
                data = json.load(filepath.open(encoding='utf-8'))
                entries = data if isinstance(data, list) else [data]
                for entry in entries:
                    text = entry.get('text_content','') or ''
                    if len(text) < 10 or not mentions_singapore(text, fuzzy_threshold):
                        continue
                    chunks = self._chunk_text_for_NER(text, max_words, overlap_words)
                    total_chunks += len(chunks)
            except Exception as e:
                logger.warning(f"Skipping file due to error during chunk estimation: {filepath} - {e}")

        logger.info(f"Estimated total chunks to process: {total_chunks}")

        # === Step 2: Initialize global progress bar ===
        pbar = tqdm(total=total_chunks, desc="Chunks processed", unit="chunk")

        relevance_cache = getattr(self, '_relevance_cache', {})
        self._relevance_cache = relevance_cache

        for filepath in files:
            try:
                data = json.load(filepath.open(encoding='utf-8'))
                entries = data if isinstance(data, list) else [data]

                if not any(mentions_singapore(e.get("text_content",""), fuzzy_threshold) for e in entries):
                    continue

                for entry in entries:
                    text = entry.get('text_content','') or ''
                    if len(text) < 10 or not mentions_singapore(text, fuzzy_threshold):
                        continue

                    entry_entities          = set()
                    entry_entities_by_label = defaultdict(set)
                    chunks = self._chunk_text_for_NER(text, max_words, overlap_words)

                    # --- Begin parallel chunk processing ---
                    futures = {}
                    with ThreadPoolExecutor(max_workers=self.concurrency) as exe:
                        for chunk in chunks:
                            futures[exe.submit(self._process_chunk, chunk)] = chunk

                        for fut in as_completed(futures):
                            try:
                                preds = fut.result()
                                for p in preds:
                                    label   = p.get("label")
                                    text_val= p.get("text","").strip()
                                    if not label or not text_val:
                                        continue
                                    mapped = self.abbrev_map.get(text_val.lower(), text_val)
                                    matched_entity = self._check_relevance(mapped)
                                    if mapped not in relevance_cache:
                                        relevance_cache[mapped] = matched_entity is not None
                                    if matched_entity is not None:
                                        raw_by_label[label].append(matched_entity)
                                        entry_entities.add(matched_entity)
                                        entry_entities_by_label[label].add(matched_entity)
                            except Exception as e:
                                logger.warning(f"Error in parallel chunk: {e}")
                            finally:
                                pbar.update(1)  # Update progress per chunk

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

        pbar.close()


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
        # Step 1: Collect ALL unique entities across all labels
        all_unique_entities = set()
        for ents in merged_counts.values():
            all_unique_entities.update(ents.keys())
        all_unique_entities = list(all_unique_entities)  # Convert to list for indexing

        if all_unique_entities:
            # Step 2: Encode ALL entities in ONE batch call (uses GPU efficiently)
            logger.info(f"Encoding {len(all_unique_entities)} unique entities in a single batch...")
            all_vectors = self.embedder.encode(all_unique_entities, batch_size=32, show_progress_bar=False)

            # Step 3: Assign vectors back to their respective labels/entities
            for label, ents in merged_counts.items():
                embeddings[label] = {}
                for ent in ents:
                    # Find the index of this entity in the master list
                    idx = all_unique_entities.index(ent)
                    embeddings[label][ent] = all_vectors[idx].tolist()
        else:
            logger.info("No entities found to encode.")

        # Compute per-label distribution summing to 100%
        docs_count_by_label = {label: docs_with_label.get(label, 0) for label in merged_counts}

        if total_docs == 0:
            coverage_by_label = {label: 0.0 for label in merged_counts}
        else:
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


def main() -> None:
    config = TopicExtractorConfig()
    extractor = TopicExtractor(config)
    extractor._init_db()

    # print("Seed entities in DB:", extractor._get_all_seed_entities())
    stats = extractor.extract_from_file()
    print(f"Total unique entities found: {stats['total_entities']}")
    print(f"Processed documents: {stats['total_docs']}")
    for label, cnt in sorted(stats['counts_by_label'].items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {cnt}")

if __name__ == "__main__":
    main()
