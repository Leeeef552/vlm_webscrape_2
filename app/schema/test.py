#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer


# ----------------------------
# I/O
# ----------------------------
def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "entity" not in obj or "label" not in obj:
                raise ValueError("Each line must contain 'entity' and 'label'.")
            rows.append({"entity": str(obj["entity"]).strip(), "label": str(obj["label"]).strip()})
    df = pd.DataFrame(rows)
    df = df[df["entity"].astype(str).str.len() > 0].copy()
    df.drop_duplicates(subset=["entity", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ----------------------------
# Embeddings
# ----------------------------
def prefix_texts(texts: List[str], mode: str = "passage") -> List[str]:
    mode = mode.lower()
    if mode not in {"passage", "query"}:
        mode = "passage"
    return [f"{mode}: " + t for t in texts]


def embed(df: pd.DataFrame, model_name: str, device: str | None, batch_size: int = 64, mode: str = "passage") -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    texts = prefix_texts(df["entity"].tolist(), mode=mode)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-ready
    )
    return embs


# ----------------------------
# Clustering + evaluation
# ----------------------------
def kmeans_sweep(embs: np.ndarray, k_values: List[int], random_state: int = 42) -> Tuple[int, Dict[int, float]]:
    scores = {}
    for k in k_values:
        if k < 2 or k > len(embs):
            continue
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(embs)
        # embeddings are L2-normalized, euclidean ~ cosine
        score = silhouette_score(embs, labels, metric="euclidean")
        scores[k] = float(score)
    if not scores:
        return 2, {}
    best_k = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_k, scores


def run_kmeans(embs: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(embs)
    return labels, km.cluster_centers_


def run_dbscan(embs: np.ndarray, eps: float = 0.25, min_samples: int = 5) -> np.ndarray:
    # cosine works well on normalized vectors
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(embs)


def cluster_purity(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    dfc = df.copy()
    dfc["cluster"] = cluster_labels
    stats = (
        dfc.groupby("cluster")["label"].agg(["count"]).rename(columns={"count": "size"}).reset_index()
    )
    dom = (
        dfc.groupby(["cluster", "label"]).size().reset_index(name="n")
        .sort_values(["cluster", "n"], ascending=[True, False])
        .groupby("cluster").head(1)
        .rename(columns={"label": "dominant_label", "n": "dominant_count"})
    )
    stats = stats.merge(dom, on="cluster", how="left")
    stats["purity"] = stats["dominant_count"] / stats["size"]
    return stats.sort_values("size", ascending=False)


def majority_vote_mapping(df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, str]:
    dfc = df.copy()
    dfc["cluster"] = cluster_labels
    mapping = {}
    for c, g in dfc.groupby("cluster"):
        if len(g) == 0:
            continue
        label_counts = g["label"].value_counts()
        mapping[int(c)] = label_counts.idxmax()
    return mapping


def map_predict_labels(cluster_labels: np.ndarray, mapping: Dict[int, str]) -> List[str]:
    preds = []
    for c in cluster_labels:
        c = int(c)
        preds.append(mapping.get(c, "UNASSIGNED"))
    return preds


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to JSONL with keys 'entity','label'")
    ap.add_argument("--model", default="BAAI/bge-m3", help="HF model name")
    ap.add_argument("--device", default=None, help="Force device, e.g., 'cuda' or 'cpu'")
    ap.add_argument("--mode", default="passage", choices=["passage", "query"], help="BGE prefix mode")
    ap.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    ap.add_argument("--kmax", type=int, default=20, help="Max k to try in KMeans sweep")
    ap.add_argument("--eps", type=float, default=0.25, help="DBSCAN eps (cosine)")
    ap.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    ap.add_argument("--do_dbscan", action="store_true", help="Also run DBSCAN for reference")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_jsonl(Path(args.file))

    # 2) Embed
    embs = embed(df, args.model, device=args.device, mode=args.mode)
    emb_cols = [f"emb_{i}" for i in range(embs.shape[1])]
    pd.concat([df.reset_index(drop=True), pd.DataFrame(embs, columns=emb_cols)], axis=1)\
      .to_parquet(outdir / "embeddings.parquet", index=False)

    # 3) KMeans sweep -> best k
    n_labels = df["label"].nunique()
    # sensible range: from max(2, n_labels-2) .. min(kmax, n_labels+10, N)
    N = len(df)
    sweep_hi = min(args.kmax, max(2, n_labels + 10), N)
    k_values = list(range(max(2, n_labels - 2), sweep_hi + 1))
    best_k, sil_scores = kmeans_sweep(embs, k_values, random_state=args.random_state)

    # 4) KMeans with best k
    km_labels, _ = run_kmeans(embs, best_k, random_state=args.random_state)

    # 5) Evaluate clustering vs. ground truth
    le = LabelEncoder()
    true_y = le.fit_transform(df["label"].values)
    ari = float(adjusted_rand_score(true_y, km_labels))
    nmi = float(normalized_mutual_info_score(true_y, km_labels))
    sil_best = float(silhouette_score(embs, km_labels, metric="euclidean"))

    # 6) Majority-vote mapping → predicted labels
    mapping = majority_vote_mapping(df, km_labels)
    pred_labels = map_predict_labels(km_labels, mapping)

    # 7) Classification-style scores from the mapping
    acc = float(accuracy_score(df["label"], pred_labels))
    macro_f1 = float(f1_score(df["label"], pred_labels, average="macro"))
    micro_f1 = float(f1_score(df["label"], pred_labels, average="micro"))

    # 8) Save reports
    # Per-cluster stats/purity
    purity_df = cluster_purity(df, km_labels)
    purity_df.to_csv(outdir / "cluster_report.csv", index=False)

    # Predictions per row
    pred_df = df.copy()
    pred_df["cluster"] = km_labels
    pred_df["pred_label"] = pred_labels
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    metrics = {
        "n_examples": int(N),
        "n_labels_true": int(n_labels),
        "kmeans": {
            "best_k": int(best_k),
            "silhouette_scores": sil_scores,      # per-k sweep
            "silhouette_best": sil_best,          # at best_k
            "ari": ari,
            "nmi": nmi,
            "mapping_accuracy": acc,
            "mapping_macro_f1": macro_f1,
            "mapping_micro_f1": micro_f1,
        },
    }

    if args.do_dbscan:
        db_labels = run_dbscan(embs, eps=args.eps, min_samples=args.min_samples)
        # Ignore noise (-1) in mapping/purity; still compute ARI/NMI with all points
        metrics["dbscan"] = {
            "clusters_excluding_noise": int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
            "noise_points": int(np.sum(db_labels == -1)),
            "ari": float(adjusted_rand_score(true_y, db_labels)),
            "nmi": float(normalized_mutual_info_score(true_y, db_labels)),
        }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 9) Console summary
    print("\n=== SUMMARY ===")
    print(f"Rows: {N} | Unique labels: {n_labels}")
    print(f"KMeans best k: {best_k} | Silhouette(best): {sil_best:.3f} | ARI: {ari:.3f} | NMI: {nmi:.3f}")
    print(f"Label mapping — Acc: {acc:.3f} | Macro-F1: {macro_f1:.3f} | Micro-F1: {micro_f1:.3f}")
    if args.do_dbscan:
        db = metrics['dbscan']
        print(f"DBSCAN clusters (ex noise): {db['clusters_excluding_noise']} | noise: {db['noise_points']}")
        print(f"DBSCAN ARI: {db['ari']:.3f} | NMI: {db['nmi']:.3f}")
    print("Saved:", (outdir / "embeddings.parquet").resolve())
    print("Saved:", (outdir / "cluster_report.csv").resolve())
    print("Saved:", (outdir / "predictions.csv").resolve())
    print("Saved:", (outdir / "metrics.json").resolve())


if __name__ == "__main__":
    main()
