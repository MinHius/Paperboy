from numpy.typing import NDArray
import numpy as np
import json

def articles_centroid(articles: list) -> NDArray[np.float32]:
    """Calculate the centroid of a story from its articles."""
    embeddings = []
    for article in articles:
        tmp = article["embedding"]
        if tmp is not None:
            if isinstance(tmp, str):
                tmp = json.loads(tmp)
                embeddings.append(np.array(tmp, dtype=np.float32))
            else:
                embeddings.append(article["embedding"])

    if not embeddings:
        raise ValueError("No embeddings provided")


    embeddings_array = np.vstack(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    return centroid.astype(np.float32)


def update_centroid(
    old_centroid: NDArray[np.float32],
    new_embeddings: list[NDArray[np.float32]],
    old_cluster_size: int
) -> NDArray[np.float32]:
    
    if not new_embeddings:
        return old_centroid

    new_embeddings_array = np.vstack(new_embeddings)
    new_sum = np.sum(new_embeddings_array, axis=0)

    updated_centroid = (
        old_centroid * old_cluster_size + new_sum
    ) / (old_cluster_size + len(new_embeddings))

    return updated_centroid.astype(np.float32)