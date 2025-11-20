import sys
sys.path.append("D:/DH/Senior/Paperboy") 

from datetime import datetime
from database.parade.database import load_articles, search_similar_articles, upsert_stories, load_stories
from src.cluster.clustering import reduce_dimensions, _perform_clustering
from api.llm.llm import validate_clusters, synthesize_stories
from api.video.video import fetch_youtube_videos
from pathlib import Path
import pickle
import json
import numpy as np
import copy
import os


inference_time = datetime.now()
marker = "10_11_25"
os.makedirs(f"src/pickled_data/{marker}", exist_ok=True)

"""Step 1: Load articles"""
articles = load_articles(inference_time)
LU_table = {}
for article in articles:
    LU_table[article["id"]] = article 
with open(f"src/pickled_data/{marker}/LU_table.pkl", "wb") as f:
    pickle.dump(LU_table, f)
print("Step 1 - Load articles: Done!")


"""Step 2: Reduce embedding dimensions"""
reduced_dim_articles = copy.deepcopy(articles)
embeddings = [
    json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
    for d in articles
]
embeddings = np.array(embeddings, dtype=np.float32)
reduced_embeddings = reduce_dimensions(embeddings, None)
for article, embedding in zip(reduced_dim_articles, reduced_embeddings):
    article["embedding"] = embedding
with open(f"src/pickled_data/{marker}/reduced_dim_articles.pkl", "wb") as f:
    pickle.dump(reduced_dim_articles, f)
print("Step 2 - Reduce embedding dimension: Done!")


"""Step 3: Cluster articles"""
raw_clusters = _perform_clustering(reduced_dim_articles, articles)
similar_articles = search_similar_articles(raw_clusters)
with open(f"src/pickled_data/{marker}/similar_articles.pkl", "wb") as f:
    pickle.dump(similar_articles, f)
print("Step 3 - Cluster articles: Done!")



"""Step 4: Validate clusters"""
similar_articles = [
    [str(group[0]), [(a[0], a[1][:701]) for a in group[1] if a[1]]]
    for group in similar_articles
]        
validation_result = validate_clusters(similar_articles)
with open(f"src/pickled_data/{marker}/validation_result.pkl", "wb") as f:
    pickle.dump(validation_result, f)

qualified_groups = []
for result in validation_result:
    if result.get('score') != None:
        if result.get('score') >= 8:
            content = []
            image = []
            author = []
            for d in result.get("articles"):
                if d in LU_table:
                    content.append(LU_table[d]['body'][:701])
                    image.append(LU_table[d]['thumbnail'])
                    author.append(LU_table[d]['author'])
            qualified_groups.append({'centroid': str(result.get('centroid')),
                                    'content': content,
                                    'image': image,
                                    'keyword': result.get('keyword'),  
                                    'author': author,           
            })            
with open(f"src/pickled_data/{marker}/qualified_result.pkl", "wb") as f:
    pickle.dump(qualified_groups, f)
print("Step 4 - Validate clusters: Done!")

    
"""Step 5: Synthesize stories"""
synthesized_story = synthesize_stories(qualified_groups)

with open(f"src/pickled_data/{marker}/raw_synthesis_result.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
for i, group in enumerate(qualified_groups):
    synthesized_story[i]["centroid"] = group['centroid']
    synthesized_story[i]["thumbnail"] = group['image'] 
    
with open(f"src/pickled_data/{marker}/synthesis_result_img_cen.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
print("Step 5 - Synthesize story: Done!")


"""Step 5.5: Re-evaluate video links for availability."""
for i, story in enumerate(synthesized_story):
    videos = fetch_youtube_videos(story['title'])
    # Save to JSON file
    save_path = f"3. videos/video_{i}_{marker}.json"
    story['video'] = videos if videos else []
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
with open(f"src/pickled_data/{marker}/complete_stories.pkl", "wb") as f:
    pickle.dump(synthesized_story, f)
print("Step 5.5 - Gather video links: Done!")
    


"""Step 6: Check story integrity."""
expected_fields = {
    "id", "title", "topic", "story", "thumbnail",
    "figure", "quotes", "keyword", "video", "location",
    "created_at", "centroid", "sentiment", "author"
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

    
"""Step 7: Upsert stories to DB."""
upsert_stories(synthesized_story)
print("Step 7 - Upsert story: Done!")


# ==============================================================================

# folder = "D:/DH/Senior/Paperboy/3. videos"

# for i, filename in enumerate(os.listdir(folder)):
#     if filename.endswith(".json"):
#         path = os.path.join(folder, filename)
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             stories[i]['video'] = data if data else []

# with open("src/pickled_data/synthesized_with_keyword.pkl", "wb") as f:
#     pickle.dump(stories, f)
