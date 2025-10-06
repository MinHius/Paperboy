import sys
sys.path.append("D:/DH/Senior/Paperboy") 

from datetime import datetime
from src.databases.database import load_articles, search_similar_articles
from src.cluster.clustering import reduce_dimensions, _perform_clustering
from src.api.LLM import validate_clusters
from pathlib import Path
import pickle
import json
import numpy as np
import copy


inference_time = datetime.now()

"""Step 1: Load articles"""
articles = load_articles(inference_time)
print("Step 1: Done!")


"""Step 2: Reduce embedding dimensions"""
reduced_dim_articles = copy.deepcopy(articles)
embeddings = [
    json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
    for d in articles
]
embeddings = np.array(embeddings, dtype=np.float32)
ROOT = Path("d:/DH/Senior/Paperboy/src")  # your project root
file_path = ROOT / "pickled_data" / "articles.pkl"  
reduced_embeddings = reduce_dimensions(embeddings, None)
for article, embedding in zip(reduced_dim_articles, reduced_embeddings):
    article["embedding"] = embedding
print("Step 2: Done!")


"""Step 3: Cluster articles"""
raw_clusters = _perform_clustering(reduced_dim_articles, articles)
similar_articles = search_similar_articles(raw_clusters)
print(similar_articles)

with open("src/pickled_data/similar_articles.pkl", "wb") as f:
    pickle.dump(similar_articles, f)
print("Step 3: Done!")


"""Step 4: Validate clusters"""
similar_articles = [
    [(a[0], a[1][:701]) for a in group]
    for group in similar_articles
]
        
validation_result = validate_clusters(similar_articles)
print(validation_result[0])