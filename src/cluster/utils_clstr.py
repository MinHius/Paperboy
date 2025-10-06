from numpy.typing import NDArray
import numpy as np
import json

def articles_centroid(articles: list) -> NDArray[np.float32]:
    """Calculate the centroid (mean embedding) of a story from its articles."""
    embeddings = []
    for article in articles:
        tmp = article["embedding"]
        if tmp is not None:
            if isinstance(tmp, str):
                tmp = json.loads(tmp)   # turn string into list
                embeddings.append(np.array(tmp, dtype=np.float32))
            else:
                embeddings.append(article["embedding"])

    if not embeddings:
        raise ValueError("No articles with embeddings found for centroid calculation")

    # Stack embeddings and calculate mean
    embeddings_array = np.vstack(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    return centroid.astype(np.float32)