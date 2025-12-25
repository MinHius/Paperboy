import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

from typing import TypedDict
import numpy as np
import json
from umap import UMAP
from numpy.typing import NDArray
from datetime import datetime
from sklearn.cluster import HDBSCAN  # type:ignore
from sklearn.metrics.pairwise import cosine_similarity
from database.parade.database import load_articles
from .config import (
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    UMAP_RANDOM_STATE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    VECTOR_CLUSTER_MIN_SIZE,
    VECTOR_CLUSTER_THRESHOLD
)

from .utils import articles_centroid


def reduce_dimensions(
    embeddings: NDArray[np.float32], seeding: bool
) -> NDArray[np.float32]:
    """Reduce embedding dimensions using UMAP."""
    reducer = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        n_jobs=-1,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        verbose=False,
        random_state=UMAP_RANDOM_STATE,
    )
    reduced = reducer.fit_transform(embeddings)
    return np.array(reduced, dtype=np.float32)


def _perform_clustering(
    reduced_dim_articles: NDArray[np.float32],
    articles: list,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
) -> dict:
    """Internal function to perform HDBSCAN clustering with configurable parameters."""
    
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        n_jobs=-1,
    )
    
    reduced_embeddings = [d["embedding"] for d in reduced_dim_articles]

    labels: list[int] = clusterer.fit_predict(reduced_embeddings)

    # Group articles by cluster label
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        key = str(label)

        if key not in clusters:
            clusters[key] = {
                "articles": [],
                "centroid": np.empty(0, dtype=np.float32),
            }
        clusters[key]["articles"].append(articles[i])

    # Calculate centroids from original embeddings (1024D)
    for label, cluster in clusters.items():
        centroid = articles_centroid(cluster["articles"])
        clusters[label]["centroid"] = centroid

    return clusters



def search_in_batch(query_emb, embeddings):
    sims = embeddings @ query_emb
    idx = np.flatnonzero(sims >= VECTOR_CLUSTER_THRESHOLD)
    return idx[np.argsort(-sims[idx])].tolist()


def leader_clustering(articles, embeddings, reference_time) -> None:
    if not reference_time:
        reference_time = datetime.now()

    N = len(articles)
    used = set()
    clusters = {}

    embeddings = np.stack(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    used = np.zeros(N, dtype=bool)
    for i in range(N):
        if used[i]:
            continue
        
        # 1) take article i as temporary “center”
        cluster_idx = search_in_batch(embeddings[i], embeddings) # 0.70

        # 2) filter out ones already used
        if len(cluster_idx) >= VECTOR_CLUSTER_MIN_SIZE:
            if i not in clusters:
                clusters[i] = {
                    "articles": [],
                    "centroid": np.empty(0, dtype=np.float32),
                }
            for j in cluster_idx:
                if not used[j]:
                    used[j] = True
                    clusters[i]["articles"].append(articles[j])

    for label, cluster in clusters.items():
        centroid = articles_centroid(cluster["articles"])
        clusters[label]["centroid"] = centroid

    if not clusters:
        print("No clusters created")
        return
    
    print(f"Final: Collected {len(clusters)} clusters")
    
    return clusters




