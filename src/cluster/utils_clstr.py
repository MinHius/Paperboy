from numpy.typing import NDArray
import numpy as np

def articles_centroid(articles: list) -> NDArray[np.float32]:
    """Calculate the centroid (mean embedding) of a story from its articles."""
    embeddings = []
    for article in articles:
        if article.embedding is not None:
            embeddings.append(article.embedding)

    if not embeddings:
        raise ValueError("No articles with embeddings found for centroid calculation")

    # Stack embeddings and calculate mean
    embeddings_array = np.vstack(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    return centroid.astype(np.float32)