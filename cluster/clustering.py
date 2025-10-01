import logging
from typing import TypedDict
import numpy as np
import umap
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN  # type:ignore
from sklearn.metrics.pairwise import cosine_similarity
from .config_cluster import (
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    UMAP_RANDOM_STATE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    CLUSTER_MERGE_THRESHOLD,
)

from .utils_clstr import articles_centroid


def reduce_dimensions(
    embeddings: NDArray[np.float32], seeding: bool
) -> NDArray[np.float32]:
    """Reduce embedding dimensions using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        n_jobs=-1,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        verbose=False,
        random_state=UMAP_RANDOM_STATE if seeding else None,
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