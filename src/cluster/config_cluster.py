# Clustering parameters
UMAP_N_NEIGHBORS: int = 25
UMAP_N_COMPONENTS: int = 96
UMAP_MIN_DIST: float = 0.05
UMAP_METRIC: str = "cosine"
UMAP_RANDOM_STATE: int = 42

HDBSCAN_MIN_CLUSTER_SIZE: int = 8
HDBSCAN_MIN_SAMPLES: int = 10
HDBSCAN_METRIC: str = "cosine"