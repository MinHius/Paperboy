from datetime import datetime
from Paperboy.databases.database import load_articles, search_similar_articles
from Paperboy.cluster.clustering import reduce_dimensions, _perform_clustering
import pickle


inference_time = datetime.now()

"""Step 1: Load articles"""
articles = load_articles(inference_time)


"""Step 2: Reduce embedding dimensions"""
reduced_dim_articles = articles.copy()
embeddings = [d["embedding"] for d in articles]

with open("pickled_data/full_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
    
reduced_embeddings = reduce_dimensions(embeddings)
for article, embedding in zip(reduced_dim_articles, reduced_embeddings):
    reduced_dim_articles["embedding"] = reduce_dimensions

"""Step 3: Cluster articles"""
raw_clusters = _perform_clustering(reduced_dim_articles, articles)

similar_articles = search_similar_articles(raw_clusters)

print(similar_articles)

