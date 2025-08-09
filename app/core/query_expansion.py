from ..utils.logger import logger
from ..utils.prompts import depth_prompt, width_prompt
from ..configs.config import QueryExpansionConfig
from pathlib import Path
import json
import re
from openai import OpenAI
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import sqlite3
import numpy as np
from numpy.linalg import norm

class QueryExpansion:
    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.db_path = Path(self.config.db_path) / "kb.sqlite"

        # Initialize OpenAI client once
        self.client = OpenAI(base_url=self.config.base_url, api_key="dummy")

        # Prompt templates
        self.depth_prompt = depth_prompt
        self.width_prompt = width_prompt

    ##########################################
    ##          SQL/Helper Functions        ##
    ##########################################
    
    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _get_total_docs(self) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE key='total_docs'")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def get_bottom_labels(self, n: Optional[int] = None):
        """Rank labels by doc coverage then by entity count."""
        n = n or self.config.bottom_n_labels
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT l.label_id, l.label,
                       COALESCE(lc.docs_with_label,0) AS docs,
                       COUNT(e.entity_id) AS n_entities,
                       COALESCE(SUM(e.total_count),0) AS mentions
                FROM labels l
                LEFT JOIN label_coverage lc ON lc.label_id = l.label_id
                LEFT JOIN entities e ON e.label_id = l.label_id
                GROUP BY l.label_id
                ORDER BY docs ASC, n_entities ASC, mentions ASC
                LIMIT ?
            """, (n,))
            return cur.fetchall()  # (label_id, label, docs, n_entities, mentions)

    def get_bottom_entities(self, n: Optional[int] = None, max_count: Optional[int] = None):
        """Underexplored entities: lowest total_count; break ties by degree desc."""
        n = n or self.config.bottom_n_entities
        max_count = max_count or self.config.min_entity_count
        with self._conn() as conn:
            cur = conn.cursor()
            # degree = number of unique neighbors
            cur.execute("""
                WITH deg AS (
                  SELECT entity_id, COUNT(*) AS degree FROM (
                    SELECT entity_a AS entity_id FROM cooccurrence
                    UNION ALL
                    SELECT entity_b AS entity_id FROM cooccurrence
                  ) t GROUP BY entity_id
                )
                SELECT e.entity_id, l.label, e.entity_text, e.total_count,
                       COALESCE(d.degree,0) AS degree
                FROM entities e
                JOIN labels l ON l.label_id = e.label_id
                LEFT JOIN deg d ON d.entity_id = e.entity_id
                WHERE e.total_count <= ?
                ORDER BY e.total_count ASC, d.degree DESC, e.entity_text ASC
                LIMIT ?
            """, (max_count, n))
            return cur.fetchall()  # (entity_id, label, text, count, degree)

    def top_neighbors(self, entity_id: int, k: int = 5):
        with self._conn() as conn:
            cur = conn.cursor()
            # both directions
            cur.execute("""
                SELECT e2.entity_text, c.count
                FROM cooccurrence c
                JOIN entities e2 ON e2.entity_id = c.entity_b
                WHERE c.entity_a = ?
                UNION ALL
                SELECT e1.entity_text, c.count
                FROM cooccurrence c
                JOIN entities e1 ON e1.entity_id = c.entity_a
                WHERE c.entity_b = ?
                ORDER BY count DESC
                LIMIT ?
            """, (entity_id, entity_id, k))
            return cur.fetchall()  # [(text, count), ...]

    def _load_embeddings(self):
        """Return list of (entity_id, label, text, count, vector[np.ndarray])."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
              SELECT e.entity_id, l.label, e.entity_text, e.total_count, emb.vector
              FROM embeddings emb
              JOIN entities e ON e.entity_id = emb.entity_id
              JOIN labels l ON l.label_id = e.label_id
            """)
            items = []
            for eid, label, text, cnt, vec_json in cur.fetchall():
                v = np.asarray(json.loads(vec_json), dtype=np.float32)
                items.append((eid, label, text, cnt, v))
            return items

    @staticmethod
    def _cos(a, b):
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(a @ b) / (na * nb)

    def embedding_neighbors(self, target_id: int, k: int = 8):
        items = getattr(self, "_emb_cache", None)
        if items is None:
            items = self._load_embeddings()
            self._emb_cache = items
        id_to_idx = {eid: i for i, (eid, *_rest) in enumerate(items)}
        if target_id not in id_to_idx: return []
        qi = id_to_idx[target_id]
        qv = items[qi][4]
        sims = []
        for i, (eid, _lab, text, _cnt, v) in enumerate(items):
            if i == qi: continue
            sims.append((text, self._cos(qv, v)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in sims[:k]]

    def mmr(self, candidates: List[str], embed_lookup: Dict[str, np.ndarray], k: int = 10, lam: float = 0.7):
        if not candidates: return []
        selected = [candidates[0]]
        rest = candidates[1:]
        def sim(a,b): 
            va, vb = embed_lookup.get(a), embed_lookup.get(b)
            return self._cos(va, vb) if (va is not None and vb is not None) else 0.0
        while rest and len(selected) < k:
            best, best_score = None, -1e9
            for c in rest:
                rel = max(sim(c, s) for s in selected) if selected else 0.0
                relevance = 0.0  # if you want, plug a query vector here
                score = lam*relevance - (1-lam)*rel
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            rest.remove(best)
        return selected
    
    def _parse_queries(self, text: str) -> List[str]:
        """
        Parse numbered or quoted lines from the LLM response into a list of query strings.
        """
        queries: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            # Match lines like '1. "query"' or '1) query'
            m = re.match(r'^\d+[\)\.\-\s]+["“]?(.*?)["”]?$', line)
            if m:
                queries.append(m.group(1).strip())
        return queries

    #####################################################
    ##          Core Query Expansion Functions         ##
    #####################################################

    def generate_depth_query(self, base_query: str, extra_hints: List[str] = None) -> List[str]:
        n = self.config.expansion_depth
        # rows: (label_id, label, docs, n_entities, mentions)
        bottom_labels = [row[1] for row in self.get_bottom_labels()]
        labels_str = ", ".join(bottom_labels)
        hints = ""
        if extra_hints:
            hints = "\nWhen helpful, consider these related entities: " + ", ".join(extra_hints[:8])

        prompt = self.depth_prompt.format(n=n, bottom_labels=labels_str) + hints
        messages = [
            {"role": "system", "content": "You are a Singapore-focused query expansion assistant that creates Google search queries for research and knowledge base expansion"},
            {"role": "user", "content": f"{prompt}\nOriginal Query: \"{base_query}\""}
        ]
        resp = self.client.chat.completions.create(model=self.config.model_name, messages=messages)
        return self._parse_queries(resp.choices[0].message.content)



    def generate_width_query(self, base_query: str, extra_hints: List[str] = None) -> List[str]:
        n = self.config.expansion_width
        # rows: (entity_id, label, entity_text, total_count, degree)
        bottom_entities = [row[2] for row in self.get_bottom_entities()]
        entities_str = ", ".join(bottom_entities)
        hints = ""
        if extra_hints:
            hints = "\nConsider these related/neighbor entities too: " + ", ".join(extra_hints[:8])

        prompt = self.width_prompt.format(n=n, bottom_entities=entities_str) + hints
        messages = [
            {"role": "system", "content": "You are a Singapore-focused query expansion assistant that creates Google search queries for research and knowledge base expansion"},
            {"role": "user", "content": f"{prompt}\nOriginal Query: \"{base_query}\""}
        ]
        resp = self.client.chat.completions.create(model=self.config.model_name, messages=messages)
        return self._parse_queries(resp.choices[0].message.content)


    def _expand_one_entity(self, entity: str, entity_id: Optional[int] = None) -> Tuple[List[str], List[str]]:
        # collect neighbors for context
        neighbor_texts = []
        if entity_id is not None:
            neighbor_texts += [row[0] for row in self.top_neighbors(entity_id, k=5)]
            if self.config.use_embeddings:
                neighbor_texts += self.embedding_neighbors(entity_id, k=5)

        depth_qs = self.generate_depth_query(entity, extra_hints=neighbor_texts)
        width_qs = self.generate_width_query(entity, extra_hints=neighbor_texts)
        return depth_qs, width_qs

    def auto_expand_entities(self) -> Dict[str, Dict[str, List[str]]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite KB not found at {self.db_path}")
        bottoms = self.get_bottom_entities(n=self.config.bottom_n_entities,
                                            max_count=self.config.min_entity_count)
        pairs = [(row[0], row[2]) for row in bottoms]  # (entity_id, entity_text)
        if not pairs:
            logger.warning("No bottom entities found for auto expansion")
            return {}

        expansions = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut2ent = {ex.submit(self._expand_one_entity, text, eid): (eid, text) for eid, text in pairs}
            for fut in as_completed(fut2ent):
                eid, text = fut2ent[fut]
                try:
                    depth_qs, width_qs = fut.result()
                    expansions[text] = {"depth": depth_qs, "width": width_qs}
                except Exception as exc:
                    logger.error(f"Expansion failed for '{text}': {exc}")
                    expansions[text] = {"depth": [], "width": []}
        return expansions


    def auto_expand_labels(self) -> Dict[str, List[str]]:
        """
        Automatically generate width expansions for bottom labels.
        Fallback to empty dict if no bottom labels are available.
        """
        bottoms = self.get_bottom_labels()
        if not bottoms:
            logger.warning("No bottom labels found for auto label expansion")
            return {}

        expansions: Dict[str, List[str]] = {}
        for row in bottoms:
            label = row[1]
            expansions[label] = self.generate_width_query(label)

        return expansions
    
    def auto_expand_all(self) -> Dict[str, Dict]:
        tasks = {
            "entities": self.auto_expand_entities,
            "labels": self.auto_expand_labels,
        }
        results: Dict[str, Dict] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_key = {
                executor.submit(func): key for key, func in tasks.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"auto_expand_{key} failed: {e}")
                    results[key] = {}

        return results
    
    def flatten_all_expansions(self, all_expansions: Dict[str, Dict], dedupe: bool = True) -> List[str]:

        flat: List[str] = []

        for parts in all_expansions.get("entities", {}).values():
            flat.extend(parts.get("depth", []))
            flat.extend(parts.get("width", []))

        for queries in all_expansions.get("labels", {}).values():
            flat.extend(queries)

        if dedupe:
            seen = set()
            unique = []
            for q in flat:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique

        return flat


    def get_new_queries(self):
        all = self.auto_expand_all()
        print(all)
        return self.flatten_all_expansions(all)


def main():
    config = QueryExpansionConfig()
    qe = QueryExpansion(config)

    print(qe.get_new_queries())

if __name__ == "__main__":
    main()