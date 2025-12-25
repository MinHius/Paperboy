import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

from datetime import datetime
from database.parade.database import load_articles, search_similar_articles, upsert_stories, remove_used_articles, search_similar_stories
from cluster.clustering import reduce_dimensions, _perform_clustering, leader_clustering
from cluster.utils import articles_centroid
from api.llm.llm import validate_clusters, synthesize_stories
from api.video.video import fetch_youtube_videos
from pipeline.update.update import update_stories
from pathlib import Path
import pickle
import json
import numpy as np
import copy
import uuid
import os


inference_time = datetime.now()
marker = "18_12_25"
os.makedirs(f"src/pickled_data/{marker}", exist_ok=True)


"""Step 1: Load articles"""
articles = load_articles(inference_time)
LU_table = {}
for article in articles:
    LU_table[article["id"]] = article 
print("Step 1 - Load articles: Done!")


"""Step 2: Reduce embedding dimensions"""
embeddings = [
    json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
    for d in articles
]
embeddings = np.array(embeddings, dtype=np.float32)

# # =================================== UPDATES GOES HERE =====================================
# update_stories(inference_time, articles, embeddings, marker)
# # ===========================================================================================


"""Step 3: Cluster articles"""
raw_clusters = leader_clustering(articles, embeddings, inference_time)       

with open(f"src/pickled_data/{marker}/raw_clusters.pkl", "wb") as f:
    pickle.dump(raw_clusters, f)

similar_articles = search_similar_articles(raw_clusters)
for centroid, articles_found in similar_articles:
    for article in articles_found:
        LU_table[article["id"]] = article
    
with open(f"src/pickled_data/{marker}/LU_table.pkl", "wb") as f:
    pickle.dump(LU_table, f)
with open(f"src/pickled_data/{marker}/similar_articles.pkl", "wb") as f:
    pickle.dump(similar_articles, f)
print("Step 3 - Cluster articles: Done!")

for value in similar_articles:
    result = [art["title"] for art in value[1]]
    print(result)
    print()

"""Step 4: Validate clusters"""
similar_articles = [
    [str(group[0]), [(article["id"], article["body"][:701]) for article in group[1] if article["body"]]]
    for group in similar_articles
]        
validation_result = validate_clusters(similar_articles, marker)
with open(f"src/pickled_data/{marker}/validation_result.pkl", "wb") as f:
    pickle.dump(validation_result, f)

articles_clusters = []
qualified_groups = []
articles_to_remove = []
for result in validation_result:
    if result.get('score') != None:
        if result.get('score') >= 8:
            content = []
            image = []
            author = []
            remove = []
            embedding_list = []
            article_ids = result.get("articles")
            for d in article_ids:
                if d in LU_table:
                    remove.append(d)
                    content.append(LU_table[d]['body'][:701])
                    image.append(LU_table[d]['thumbnail'])
                    author.append(LU_table[d]['author'])
                    embedding_list.append(LU_table[d])
            qualified_groups.append({'centroid': articles_centroid(embedding_list),
                                    'content': content,
                                    'image': image,
                                    'keyword': result.get('keyword'),  
                                    'author': author, 
                                    'cluster_size': len(article_ids),                                   
            })
            articles_to_remove.append(article_ids)
            
with open(f"src/pickled_data/{marker}/articles_to_remove.pkl", "wb") as f:
    pickle.dump(articles_to_remove, f)
with open(f"src/pickled_data/{marker}/qualified_result.pkl", "wb") as f:
    pickle.dump(qualified_groups, f)
print("Step 4 - Validate clusters: Done!")


"""Step 4.5: Remove used articles"""
remove_used_articles(articles_to_remove)
print(f"Step 4.5 - Article removal: Pruned {len(articles_to_remove)} articles!")
    
    
"""Step 5: Synthesize stories"""
synthesized_story = synthesize_stories(qualified_groups, marker)

with open(f"src/pickled_data/{marker}/raw_synthesis_result.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
for i, group in enumerate(qualified_groups):
    synthesized_story[i]["centroid"] = group['centroid']
    synthesized_story[i]["thumbnail"] = group['image'] 
    
with open(f"src/pickled_data/{marker}/synthesis_result_img_cen.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
print("Step 5 - Synthesize story: Done!")


"""Step 5.5: Add video links, trending score and id."""
for i, (cluster, story) in enumerate(zip(articles_to_remove, synthesized_story)):
    if isinstance(cluster[0], str) and len(cluster[0]) == 36:
        try:
            videos = fetch_youtube_videos(str(LU_table[cluster[0]]['title']))
            # Save to JSON file
            save_path = f"3. videos/video_{i}_{marker}.json"
            story['video'] = videos if videos else []
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(videos, f, ensure_ascii=False, indent=2)
            story['trending'] = 10
            print(f"Story {i} video gathered")
        except:
            print(f"Story {i} video failed")
        story['id'] = str(uuid.uuid4())
    
with open(f"src/pickled_data/{marker}/complete_stories.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
print("Step 5.5 - Gather video links: Done!")


"""Step 6: Check story integrity."""
expected_fields = {
    "id", "title", "topic", "story", "thumbnail",
    "figure", "quotes", "keyword", "video", "location",
    "created_at", "centroid", "sentiment", "author", "cluster_size"
}

for i, story in enumerate(synthesized_story):
    present = set(story.keys())
    missing = expected_fields - present
    extra = present - expected_fields

    print(f"Story #{i+1}:")
    print("  ✅ Present:", sorted(present))
    print("  ⚠️ Missing:", sorted(missing) if missing else "None")
    print("  🌀 Extra:", sorted(extra) if extra else "None")
    print("-" * 40)  
print("Step 6 - Integrity check: Done!")
    
    
"""Step 7: Upsert stories to DB and remove used stories."""
upsert_stories(synthesized_story)
print("Step 7 - Upsert story: Done!")


"""Step 8: Audio generation"""       