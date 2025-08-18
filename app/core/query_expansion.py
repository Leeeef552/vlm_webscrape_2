import json
import math
import random
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from openai import OpenAI
import time
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

# project-local
from ..utils.logger import logger
from ..utils.prompts import (
    entity_prompt,
    labels_prompt,
    filter_prompt
)
from ..configs.config import QueryExpansionConfig


# =========================
# Internal caching helpers
# =========================

class _GraphCache:
    """Build once, reuse often. Holds graphs, embeddings and derived metrics."""
    __slots__ = (
        "p",
        "_entity_graph",
        "_label_graph",
        "_metrics_ready",
        "_emb_list",
        "_emb_eids",
        "_emb_texts",
        "_emb_vecs",
        "_emb_id2idx",
    )

    def __init__(self, parent):
        self.p = parent
        self._entity_graph: Optional[nx.Graph] = None
        self._label_graph: Optional[nx.Graph] = None
        self._metrics_ready: bool = False

        # Embedding cache (both list view and array view)
        self._emb_list: Optional[List[Tuple[int, str, str, int, np.ndarray]]] = None
        self._emb_eids = None
        self._emb_texts = None
        self._emb_vecs = None
        self._emb_id2idx = None

    # ---------- read-only properties ----------
    @property
    def entity_graph(self) -> nx.Graph:
        if self._entity_graph is None:
            self._entity_graph = self._build_entity_graph()
        return self._entity_graph

    @property
    def label_graph(self) -> nx.Graph:
        if self._label_graph is None:
            self._label_graph = self._build_label_graph()
        return self._label_graph

    @property
    def embeddings_list(self):
        if self._emb_list is None:
            self._load_embeddings()
        return self._emb_list

    @staticmethod
    def _norm01(vals: Dict[int, float]) -> Dict[int, float]:
        if not vals:
            return {}
        vmin = min(vals.values())
        vmax = max(vals.values())
        if vmax == vmin:
            return {k: 0.0 for k in vals}
        r = vmax - vmin
        return {k: (v - vmin) / r for k, v in vals.items()}

    def embeddings_arrays(self):
        """Return (eids[np.int64], texts[List[str]], vecs[np.float32 normalized], id2idx[dict])."""
        if self._emb_vecs is None:
            self._load_embeddings()
        return self._emb_eids, self._emb_texts, self._emb_vecs, self._emb_id2idx

    # ---------- construction ----------
    def _build_entity_graph(self) -> nx.Graph:
        G = nx.Graph()
        with self.p._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.entity_id, e.entity_text, e.total_count, l.label
                FROM entities e
                JOIN labels l ON l.label_id = e.label_id
                """
            )
            for eid, txt, cnt, lbl in cur.fetchall():
                G.add_node(eid, text=txt, count=cnt, label=lbl)

            cur.execute("SELECT entity_a, entity_b, count FROM cooccurrence")
            for a, b, w in cur.fetchall():
                if a in G and b in G:
                    G.add_edge(a, b, weight=float(w))
        return G


    def _build_label_graph(self, weighting: str = "npmi", min_pair: int = 2, only_pos: bool = True) -> nx.Graph:
        G = nx.Graph()
        with self.p._conn() as conn:
            cur = conn.cursor()
            # nodes
            cur.execute(
                """
                SELECT l.label_id, l.label, COUNT(*), COALESCE(SUM(e.total_count),0)
                FROM labels l LEFT JOIN entities e ON e.label_id = l.label_id
                GROUP BY l.label_id
                """
            )
            for lid, lab, n_ent, n_ment in cur.fetchall():
                G.add_node(
                    lid, label=lab, entity_count=int(n_ent), mention_count=int(n_ment)
                )

            # raw co-occurrence between labels (via entity pairs)
            cur.execute(
                """
                SELECT ea.label_id, eb.label_id, SUM(c.count) AS w
                FROM cooccurrence c
                JOIN entities ea ON ea.entity_id = c.entity_a
                JOIN entities eb ON eb.entity_id = c.entity_b
                WHERE ea.label_id != eb.label_id
                GROUP BY ea.label_id, eb.label_id
                """
            )
            raw = cur.fetchall()

        pair_count: Dict[Tuple[int, int], float] = {}
        label_pair_sum = defaultdict(float)
        total_w = 0.0

        for la, lb, w in raw:
            if la == lb:
                continue
            a, b = (la, lb) if la < lb else (lb, la)
            pair_count[(a, b)] = pair_count.get((a, b), 0.0) + float(w)

        for (a, b), w in pair_count.items():
            label_pair_sum[a] += w
            label_pair_sum[b] += w
            total_w += w
        total_w = float(total_w) or 1.0

        def transform(a: int, b: int, w: float) -> float:
            if weighting.lower() in ("cosine", "association", "assoc"):
                denom = math.sqrt(max(label_pair_sum[a], 1e-9) * max(label_pair_sum[b], 1e-9))
                return float(w) / denom if denom > 0 else 0.0
            # default nPMI
            p_ab = w / total_w
            p_a = label_pair_sum[a] / total_w
            p_b = label_pair_sum[b] / total_w
            if p_ab <= 0 or p_a <= 0 or p_b <= 0:
                return 0.0
            pmi = math.log(p_ab / (p_a * p_b), 2)
            npmi = pmi / (-math.log(p_ab, 2))
            return float(npmi)

        for (a, b), w in pair_count.items():
            if w < min_pair:
                continue
            if a not in G or b not in G:
                continue
            wt = transform(a, b, w)
            if only_pos and weighting.lower() == "npmi" and wt <= 0:
                continue
            G.add_edge(a, b, weight=wt, raw=float(w))

        # distances for weighted shortest paths
        if G.number_of_edges() > 0:
            invw = {(u, v): 1.0 / max(ed.get("weight", 1e-9), 1e-9) for u, v, ed in G.edges(data=True)}
            nx.set_edge_attributes(G, invw, name="invw")

        # compute "strength" = sum of positive weights
        for n in G.nodes():
            G.nodes[n]["strength"] = sum(
                max(G[n][m].get("weight", 0.0), 0.0) for m in G.neighbors(n)
            )
        return G

    def ensure_metrics(self):
        """Compute heavy metrics once for the entity graph (and a kb_strength using labels)."""
        if self._metrics_ready:
            return
        start = time.time()
        G = self.entity_graph
        if G.number_of_edges() > 0:
            invw = {(u, v): 1.0 / max(d.get("weight", 1.0), 1e-6) for u, v, d in G.edges(data=True)}
            nx.set_edge_attributes(G, invw, "invw")
        else:
            # degenerate empty graph
            for n in G.nodes():
                G.nodes[n]["invw"] = {}

        # Core centralities (weighted)
        try:
            if G.number_of_nodes() > 5000:
                betw = nx.betweenness_centrality(G, k=min(1000, int(0.3*G.number_of_nodes())), weight="invw", seed=42)
            else:
                betw = nx.betweenness_centrality(G, weight="invw")

        except Exception:
            betw = {n: 0.0 for n in G}
        try:
            clos = nx.closeness_centrality(G, distance="invw")
        except Exception:
            clos = {n: 0.0 for n in G}
        try:
            pr = nx.pagerank(G, weight="weight")
        except Exception:
            pr = {n: 0.0 for n in G}
        try:
            clus = nx.clustering(G, weight="weight")
        except Exception:
            clus = {n: 0.0 for n in G}

        for n in G.nodes():
            d = G.nodes[n]
            d["betweenness"] = float(betw.get(n, 0.0))
            d["closeness"] = float(clos.get(n, 0.0))
            d["pagerank"] = float(pr.get(n, 0.0))
            d["clustering"] = float(clus.get(n, 0.0))

        # kb_strength (combines degree/PR/count/label prior/closeness)
        label_strength_raw: Dict[str, float] = {}
        LG = self.label_graph
        for lid, d in LG.nodes(data=True):
            lab = str(d.get("label", "")).lower()
            label_strength_raw[lab] = float(d.get("strength", 0.0))
        max_ls = max(label_strength_raw.values(), default=0.0) or 1.0

        deg_w = {n: sum(float(G[n][nb].get("weight", 0.0)) for nb in G.neighbors(n)) for n in G.nodes()}
        counts = {n: math.log1p(float(G.nodes[n].get("count", 0.0))) for n in G.nodes()}
        pr_map = {n: float(G.nodes[n].get("pagerank", 0.0)) for n in G.nodes()}
        close_map = {n: float(G.nodes[n].get("closeness", 0.0)) for n in G.nodes()}

        n_counts = self._norm01(counts)
        n_deg = self._norm01(deg_w)
        n_pr = self._norm01(pr_map)
        n_close = self._norm01(close_map)

        W = getattr(self.p.config, "kb_strength_weights", None) or {
            "count": 0.35,
            "deg": 0.20,
            "pr": 0.25,
            "label": 0.15,
            "closeness": 0.05,
        }

        for n in G.nodes():
            labtxt = str(G.nodes[n].get("label", "")).lower()
            lab_prior = label_strength_raw.get(labtxt, 0.0) / max_ls
            score = (
                W["count"] * n_counts.get(n, 0.0)
                + W["deg"] * n_deg.get(n, 0.0)
                + W["pr"] * n_pr.get(n, 0.0)
                + W["label"] * lab_prior
                + W["closeness"] * n_close.get(n, 0.0)
            )
            G.nodes[n]["kb_strength"] = float(score)

        logger.info(f"All Graph Metrics Computed: Took {time.time() - start:.2f}s")
        self._metrics_ready = True

    # ---------- embeddings ----------
    def _load_embeddings(self):
        """Loads embeddings once; keeps both a list of tuples and normalized arrays for fast cosine ops."""
        with self.p._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.entity_id, l.label, e.entity_text, e.total_count, emb.vector
                FROM embeddings emb
                JOIN entities e ON e.entity_id = emb.entity_id
                JOIN labels l ON l.label_id = e.label_id
                """
            )
            items = []
            for eid, label, text, cnt, vec_json in cur.fetchall():
                v = np.asarray(json.loads(vec_json), dtype=np.float32)
                items.append((int(eid), str(label), str(text), int(cnt), v))

        if not items:
            # initialize empty structures
            self._emb_list = []
            self._emb_eids = np.empty((0,), dtype=np.int64)
            self._emb_texts = []
            self._emb_vecs = np.empty((0, 0), dtype=np.float32)
            self._emb_id2idx = {}
            return

        self._emb_list = items
        eids = np.array([eid for eid, *_ in items], dtype=np.int64)
        texts = [text for _, _, text, _, _ in items]
        vecs = np.stack([vec for *_, vec in items]).astype(np.float32)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs = vecs / norms

        self._emb_eids = eids
        self._emb_texts = texts
        self._emb_vecs = vecs
        self._emb_id2idx = {int(e): i for i, e in enumerate(eids)}


class QueryExpansion:
    """Single-source-of-truth implementation with aggressive caching.
    - No duplicated build methods
    - Heavy metrics/embeddings computed once and reused
    """
    __slots__ = ("config", "db_path", "client", "_cache")

    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.db_path = Path(self.config.db_path) / "kb.sqlite"
        self.client = OpenAI(base_url=self.config.base_url, api_key="dummy")
        self._cache = _GraphCache(self)
        self._cache.ensure_metrics()

    # ---------- lightweight accessors ----------
    @property
    def entity_graph(self) -> nx.Graph:
        return self._cache.entity_graph

    @property
    def label_graph(self) -> nx.Graph:
        return self._cache.label_graph

    @property
    def embeddings(self) -> List[Tuple[int, str, str, int, np.ndarray]]:
        return self._cache.embeddings_list

    @staticmethod
    def _parse_queries(text: str) -> List[str]:
        """Extract queries from numbered/bulleted lines."""
        out = []
        for line in text.splitlines():
            s = line.strip()
            # matches: 1) foo / 1. "foo" / - foo / • foo
            m = re.match(r'^(?:[-•]|\d+[\)\.\-\s]+)\s*["“”]?(.*?)["“”]?\s*$', s)
            if m:
                q = m.group(1).strip()
                if q:
                    out.append(q)
        return out

    def _get_embedding_arrays(self):
        return self._cache.embeddings_arrays()

    @staticmethod
    def _norm01(vals: Dict[int, float]) -> Dict[int, float]:
        if not vals:
            return {}
        vmin = min(vals.values())
        vmax = max(vals.values())
        if vmax == vmin:
            return {k: 0.0 for k in vals}
        r = vmax - vmin
        return {k: (v - vmin) / r for k, v in vals.items()}
    
    def _llm_generate(self, prompt_template: str, base_query: str, n: Optional[int] = 4) -> List[str]:
        prompt = prompt_template.format(n=n)
        messages = [
            {"role": "system", "content": "You are a Singapore-focused knowledge-expansion assistant."},
            {"role": "user", "content": f"{prompt}\nOriginal Query: \"{base_query}\""},
        ]
        resp = self.client.chat.completions.create(model=self.config.model_name, messages=messages)
        return self._parse_queries(resp.choices[0].message.content or "")
    
    # ---------- DB ----------
    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ==========================================
    # Graph extraction
    # ==========================================
    
    # ----- entities exploit: high-impact entities -----
    def _find_exploit_entities(self, k: int = 20):
        """Find high-impact entities using occurrence count, degree centrality, and PageRank."""
        start = time.time()
        G = self.entity_graph
        
        items = []
        
        # Get raw metrics for normalization
        counts = {n: math.log1p(float(G.nodes[n].get("count", 0.0))) for n in G.nodes()}
        degrees = {n: float(G.degree(n, weight="weight")) for n in G.nodes()}
        pageranks = {n: float(G.nodes[n].get("pagerank", 0.0)) for n in G.nodes()}
        # Normalize each metric to [0, 1]
        n_counts = self._norm01(counts)
        n_degrees = self._norm01(degrees)
        n_pageranks = self._norm01(pageranks)
        # Weighted combination for exploit score
        exploit_weights = getattr(self.config, "exploit_weights", None) or {
            "count": 0.4,      # High occurrence indicates importance
            "degree": 0.35,    # Well-connected entities
            "pagerank": 0.25   # Global importance
        }
        for n in G.nodes():
            exploit_score = (
                exploit_weights["count"] * n_counts.get(n, 0.0) +
                exploit_weights["degree"] * n_degrees.get(n, 0.0) +
                exploit_weights["pagerank"] * n_pageranks.get(n, 0.0)
            )
            items.append((n, G.nodes[n]["text"], float(exploit_score)))
        items.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Found exploit entities in {time.time() - start:.2f}s")
        return items[:int(k)]

    # ----- entities explore: low-coverage, bridge entities -----
    def _find_explore_entities(self, k: int = 20):
        """Find low-coverage entities with high betweenness (bridge entities)."""
        start = time.time()
        G = self.entity_graph
        
        items = []
        # Get raw metrics
        counts = {n: float(G.nodes[n].get("count", 1.0)) for n in G.nodes()}  # Use 1.0 as min
        betweenness = {n: float(G.nodes[n].get("betweenness", 0.0)) for n in G.nodes()}
        # For exploration, we want LOW frequency (inverse count) and HIGH betweenness
        inv_counts = {n: 1.0 / max(counts[n], 1.0) for n in counts}
        # Normalize metrics
        n_inv_counts = self._norm01(inv_counts)
        n_betweenness = self._norm01(betweenness)
        # Weighted combination for explore score
        explore_weights = getattr(self.config, "explore_weights", None) or {
            "low_frequency": 0.6,  # Emphasize rare/underexplored entities
            "betweenness": 0.4     # Bridge entities that connect different parts
        }
        for n in G.nodes():
            # Skip extremely isolated nodes (degree 0)
            if G.degree(n) == 0:
                continue
            explore_score = (
                explore_weights["low_frequency"] * n_inv_counts.get(n, 0.0) +
                explore_weights["betweenness"] * n_betweenness.get(n, 0.0)
            )
            items.append((n, G.nodes[n]["text"], float(explore_score)))
        items.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Found explore entities in {time.time() - start:.2f}s")
        return items[:int(k)]
    
    # ----- labels: core labels with low entity coverage -----
    def _find_underrepresented_labels(self, k: int = 15):
        """Find important labels that have very few entities (underrepresented categories)."""
        start = time.time()
        LG = self.label_graph
        items = []
        # Get label metrics
        strengths = {n: float(LG.nodes[n].get("strength", 0.0)) for n in LG.nodes()}
        entity_counts = {n: float(LG.nodes[n].get("entity_count", 0.0)) for n in LG.nodes()}
        mention_counts = {n: float(LG.nodes[n].get("mention_count", 0.0)) for n in LG.nodes()}
        # For underrepresented labels, we want high importance but low entity coverage
        # Invert entity counts (fewer entities = higher score)
        max_entities = max(entity_counts.values()) or 1.0
        inv_entity_counts = {n: (max_entities - entity_counts[n]) / max_entities 
                            for n in entity_counts}
        # Normalize metrics
        n_strengths = self._norm01(strengths)
        n_inv_entities = self._norm01(inv_entity_counts)
        n_mentions = self._norm01(mention_counts)
        # Weighted combination for underrepresented score
        underrep_weights = getattr(self.config, "underrep_weights", None) or {
            "strength": 0.4,        # Label importance in the graph
            "low_entities": 0.45,   # Few entities (main signal)
            "mentions": 0.15        # Some mention activity (not completely dead)
        }
        for n in LG.nodes():
            # Skip labels with no entities at all or too many entities
            if entity_counts[n] == 0 or entity_counts[n] > max_entities * 0.3:
                continue
            underrep_score = (
                underrep_weights["strength"] * n_strengths.get(n, 0.0) +
                underrep_weights["low_entities"] * n_inv_entities.get(n, 0.0) +
                underrep_weights["mentions"] * n_mentions.get(n, 0.0)
            )
            label_text = str(LG.nodes[n].get("label", ""))
            items.append((n, label_text, float(underrep_score)))
        items.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Found underrepresented labels in {time.time() - start:.2f}s")
        return items[:int(k)]
    
    # ----- exploit entities semantics -----
    def _find_semantic_holes_hdbscan(self, k: int = 20, min_cluster_size: int = 5):
        """
        Returns the top-k entities that HDBSCAN labelled as noise (-1).
        These are your 'semantic holes'.
        """
        start = time.time()
        eids, texts, vecs, _ = self._get_embedding_arrays()
        if len(eids) == 0:
            return []

        # ---- 1. Build cosine distance matrix (O(n²d) time, O(n²) memory) ----
        D = cosine_distances(vecs).astype(np.float64)   # <─ ensure double precision

        # ---- 2. Run HDBSCAN on the pre-computed distances ----
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='precomputed',
            cluster_selection_epsilon=0.05
        )
        labels = clusterer.fit_predict(D)   # -1 == noise

        # ---- 3. Collect noise points and rank by outlier score ----
        noise_mask = labels == -1
        if not noise_mask.any():
            return []

        noise_eids   = eids[noise_mask]
        noise_texts  = np.array(texts)[noise_mask]
        scores       = clusterer.outlier_scores_[noise_mask]

        idx = np.argsort(scores)[::-1][:k]      # higher score → stronger hole
        logger.info(f"Found semantic holes in {time.time() - start:.2f}s")
        return [(int(noise_eids[i]), str(noise_texts[i]), float(scores[i]))
                for i in idx]
    
    # ----- gap entities : through labels -----
    def _find_gap_entities(self, labels: List[Tuple[int, str, float]], k: int = 20,) -> List[Tuple[int, str, str]]:
        start = time.time()
        if not labels:
            return []
        # Build a fast membership set for label text
        target_labels = {txt.lower() for _, txt, _ in labels}
        matches = [
            (eid, text, label)
            for eid, label, text, *_ in self.embeddings
            if label.lower() in target_labels
        ]
        # Deterministic shuffle for variety (optional)
        random.seed(42)
        random.shuffle(matches)
        logger.info(f"Found gap entities in {time.time() - start:.2f}s")
        return matches[: int(k)]

    def get_entities_and_labels(self) -> Tuple[List[Tuple], List[Tuple]]:
        start = time.time()
        total_nodes = self.entity_graph.number_of_nodes()
        max_entities = min(
            max(1, int(self.config.n_entities * total_nodes)),
            self.config.entities_cap,
        )
        k_each = max(1, max_entities // 4)

        with ThreadPoolExecutor() as pool:
            futures = {
                "exploit":  pool.submit(self._find_exploit_entities, k_each),
                "explore":  pool.submit(self._find_explore_entities, k_each),
                "holes":    pool.submit(self._find_semantic_holes_hdbscan, k_each),
                "underrep": pool.submit(self._find_underrepresented_labels, k_each),
            }
            results = {name: fut.result() for name, fut in futures.items()}
        # labels
        labels_raw = [(lid, lab, score) for lid, lab, score in results["underrep"]]
        # keep best n_labels
        label_ids = list(dict.fromkeys(lid for lid, *_ in labels_raw))[: self.config.n_labels]
        labels = [(lid, lab, sc) for lid, lab, sc in labels_raw if lid in set(label_ids)]
        # entitites
        entities = []
        # exploit, explore, holes
        for name in ("exploit", "explore", "holes"):
            entities.extend([(eid, txt, score) for eid, txt, score in results[name]])
        # gap entities
        gap_entities = self._find_gap_entities(results["underrep"], k=k_each)
        entities.extend([(eid, txt, lbl) for eid, txt, lbl in gap_entities])
        # deduplicate by entity_id while preserving order
        seen: set = set()
        entities = [t for t in entities if not (t[0] in seen or seen.add(t[0]))]
        # final cut to respect absolute cap
        entities = entities[:max_entities]
        logger.info(f"Found all entities in {time.time() - start:.2f}s")
        logger.info(f"total {len(entities)} unique entities found and {len(labels)} labels found")
        return entities, labels

    # ==========================================
    # Query Generation
    # ==========================================

    def generate_queries(self, entities: List[Tuple], labels: List[Tuple]) -> Dict[str, List[str]]:
        start = time.time()
        all_queries = {"entity_queries": [], "label_queries": []}
        queries_per_entity = self.config.num_queries_per_entity
        queries_per_label = self.config.num_queries_per_labels
        def generate_entity_queries(entity_tuple):
            entity_id, entity_text, meta = entity_tuple
            try:
                queries = self._llm_generate(
                    prompt_template=entity_prompt,
                    base_query=entity_text,
                    n=queries_per_entity
                )
                return ("entity", entity_id, entity_text, queries)
            except Exception as e:
                logger.error(f"Failed to generate queries for entity {entity_id} ({entity_text}): {e}")
                return ("entity", entity_id, entity_text, [])
        
        def generate_label_queries(label_tuple):
            label_id, label_text, score = label_tuple
            try:
                queries = self._llm_generate(
                    prompt_template=labels_prompt,
                    base_query=label_text,
                    n=queries_per_label
                )
                return ("label", label_id, label_text, queries)
            except Exception as e:
                logger.error(f"Failed to generate queries for label {label_id} ({label_text}): {e}")
                return ("label", label_id, label_text, [])
        
        tasks = []
        for entity in entities:
            tasks.append(("entity", entity))
        for label in labels:
            tasks.append(("label", label))
        
        logger.info(f"Starting concurrent query generation for {len(entities)} entities and {len(labels)} labels")
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_type, item in tasks:
                if task_type == "entity":
                    future = executor.submit(generate_entity_queries, item)
                else:  # label
                    future = executor.submit(generate_label_queries, item)
                future_to_task[future] = (task_type, item)
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_type, original_item = future_to_task[future]
                try:
                    result_type, item_id, item_text, queries = future.result()
                    if result_type == "entity":
                        all_queries["entity_queries"].extend([
                            {
                                "entity_id": item_id,
                                "entity_text": item_text,
                                "query": query,
                                "source": "entity_expansion"
                            }
                            for query in queries
                        ])
                    else:  # label
                        all_queries["label_queries"].extend([
                            {
                                "label_id": item_id,
                                "label_text": item_text,
                                "query": query,
                                "source": "label_expansion"
                            }
                            for query in queries
                        ])
                        
                except Exception as e:
                    logger.error(f"Error processing {task_type} task: {e}")
        
        total_entity_queries = len(all_queries["entity_queries"])
        total_label_queries = len(all_queries["label_queries"])
        logger.info(f"Query generation completed in {time.time() - start:.2f}s")
        logger.info(f"Generated {total_entity_queries} entity queries and {total_label_queries} label queries")
        return all_queries

    def _filter_one(self, query: str):
        """Return the query dict if it passes the filter, else None."""
        base_query = query["query"]
        template = filter_prompt
        messages = [
            {"role": "system", "content": "You are a Singapore-focused knowledge-expansion assistant."},
            {"role": "user", "content": f"{template}\nCandidate Query: \"{base_query}\""},
        ]
        resp = self.client.chat.completions.create(model=self.config.model_name, messages=messages)
        if resp.choices[0].message.content == "PASS":
            return base_query

    def get_queries(self, max_workers: int = 4) -> List[str]:
        start = time.time()
        
        entities, labels = self.get_entities_and_labels()
        raw = self.generate_queries(entities, labels) 
        all_queries = raw["entity_queries"] + raw["label_queries"]
        logger.info(f"Raw expansion queries: {len(all_queries)}")
        if not all_queries:
            logger.warning("No expansion queries generated.")
            return []

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(self._filter_one, q) for q in all_queries]
            for f in as_completed(futs):
                kept = f.result()
                if kept:
                    results.append(kept)

        logger.info(f"Queries after filtering: {len(results)}")
        logger.info(f"Original number of queries: {len(all_queries)}")
        return results

    # ==========================================
    # Graph Analysis
    # ==========================================
    def analyze_graph_structure(self) -> Dict:
        G = self.entity_graph
        if G.number_of_nodes() == 0:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "density": 0.0,
                "connected_components": 0,
                "largest_component_size": 0,
                "average_clustering": 0.0,
                "average_degree": 0.0,
            }
        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(G),
            "largest_component_size": len(max(nx.connected_components(G), key=len)),
            "average_clustering": nx.average_clustering(G, weight="weight"),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
    


def main():
    start = time.time()
    config = QueryExpansionConfig()
    qe = QueryExpansion(config)
    logger.info("Graphs & metrics warmed.")

    # 3. Query generation -----------------------------------------------------
    print("\n=== Generating expansion queries ===")
    queries = qe.get_queries(4)
    print(f"\n\n#################### Total {len(queries)} queries ###################### \n\n")

    # 4. Persist queries to file (JSONL) -------------------------------------
    out_dir = Path("/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/temp") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "expansion_queries.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"\nSaved all filtered queries to {out_file}")

    # 5. Timings --------------------------------------------------------------
    logger.info(f"All tests completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()