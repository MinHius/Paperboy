import sys
sys.path.append("D:/DH/Senior/Paperboy") 

# from datetime import datetime
# from src.databases.database import load_articles, search_similar_articles, upsert_stories, load_stories
# from src.cluster.clustering import reduce_dimensions, _perform_clustering
# from src.api.llm import validate_clusters, synthesize_stories, extract_keyword
# from pathlib import Path
import pickle
import json
# import numpy as np
# import copy
# import os


# inference_time = datetime.now()

# """Step 1: Load articles"""
# articles = load_articles(inference_time)
# LU_table = {}
# for article in articles:
#     LU_table[article["id"]] = article
    
# with open("src/pickled_data/LU_table.pkl", "wb") as f:
#     pickle.dump(LU_table, f)
# print("Step 1: Done!")


# """Step 2: Reduce embedding dimensions"""
# reduced_dim_articles = copy.deepcopy(articles)
# embeddings = [
#     json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
#     for d in articles
# ]
# embeddings = np.array(embeddings, dtype=np.float32)

# reduced_embeddings = reduce_dimensions(embeddings, None)

# for article, embedding in zip(reduced_dim_articles, reduced_embeddings):
#     article["embedding"] = embedding
    
# with open("src/pickled_data/reduced_dim_articles.pkl", "wb") as f:
#     pickle.dump(reduced_dim_articles, f)
# print("Step 2: Done!")


# """Step 3: Cluster articles"""
# raw_clusters = _perform_clustering(reduced_dim_articles, articles)
# similar_articles = search_similar_articles(raw_clusters)

# with open("src/pickled_data/similar_articles.pkl", "wb") as f:
#     pickle.dump(similar_articles, f)
# print("Step 3: Done!")


# with open("src/pickled_data/LU_table.pkl", "rb") as f:
#     LU_table = pickle.load(f)

# # with open("src/pickled_data/similar_articles.pkl", "rb") as f:
# #     similar_articles = pickle.load(f)


# """Step 4: Validate clusters"""
# similar_articles = [
#     [str(group[0]), [(a[0], a[1][:701]) for a in group[1] if a[1]]]
#     for group in similar_articles
# ]        
# validation_result = validate_clusters(similar_articles)
# with open("src/pickled_data/validation_result.pkl", "wb") as f:
#     pickle.dump(validation_result, f)

# with open("src/pickled_data/validation_result.pkl", "rb") as f:
#     validation_result = pickle.load(f)

# qualified_groups = []
# for result in validation_result:
#     if result.get('score') != None:
#         if result.get('score') >= 8:
#             content = []
#             image = []
#             for d in result.get("articles"):
#                 if d in LU_table:
#                     content.append(LU_table[d]['body'][:701])
#                     image.append(LU_table[d]['thumbnail'])
#             print(content)
#             qualified_groups.append({'centroid': str(result.get('centroid')),
#                                     'content': content,
#                                     'image': image              
#             })            

# with open("src/pickled_data/qualified_result.pkl", "wb") as f:
#     pickle.dump(qualified_groups, f)
# print("Step 4: Done!")


# with open("src/pickled_data/qualified_result.pkl", "rb") as f:
#     qualified_groups = pickle.load(f)

# """Step 5: Synthesize stories"""
# synthesized_story = synthesize_stories(qualified_groups)

# with open("src/pickled_data/synthesis_result.pkl", "wb") as f:
#     pickle.dump(synthesized_story, f)
    
# print("Step 5 done! Well done!")

# with open("src/pickled_data/qualified_result.pkl", "rb") as f:
#     qualified_groups = pickle.load(f)


# with open("src/pickled_data/synthesized_story.pkl", "rb") as f:
#     images = pickle.load(f)


# for i, group in enumerate(qualified_groups):
#     synthesized_story[i]["centroid"] = group['centroid']
#     synthesized_story[i]["image"] = group['image']
    
# with open("src/pickled_data/synthesis_result.pkl", "wb") as f:
#     pickle.dump(synthesized_story, f)

# """Step 5.5: Re-evaluate video links for availability."""
# stories = load_stories()
# print(stories[0])

# for image in images:
#     for story in stories:
#         if image['id'] == story['id']:
#             story['thumbnail'] = image['thumbnail']
#             print("Gotcha!")

# print(stories[0])
# # synthesized_story = extract_keyword(stories)

# with open("src/pickled_data/synthesized_with_keyword.pkl", "wb") as f:
#     pickle.dump(stories, f)


# with open("src/pickled_data/synthesized_with_keyword.pkl", "rb") as f:
#     synthesized_story = pickle.load(f)
    
# print(synthesized_story[0])

# upsert_stories(synthesized_story)

# ==============================================================================


# stories = load_stories()
# print(stories[0])
# with open("src/pickled_data/synthesized_story.pkl", "wb") as f:
#     pickle.dump(stories, f)
# import os, json

# folder = "D:/DH/Senior/Paperboy/4. keywords"
# all_data = []

# for i, filename in enumerate(os.listdir(folder)):
#     if filename.endswith(".json"):
#         path = os.path.join(folder, filename)
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             stories[i]['keyword'] = data

# print(f"Loaded {len(all_data)} total items from {len(os.listdir(folder))} files.")
# with open("src/pickled_data/synthesized_with_keyword.pkl", "wb") as f:
#     pickle.dump(stories, f)

# expected_fields = {
#     "id", "title", "topic", "story", "thumbnail",
#     "figure", "keyword", "video", "location",
#     "created_at", "centroid"
# }

# for i, story in enumerate(stories):
#     present = set(story.keys())
#     missing = expected_fields - present
#     extra = present - expected_fields

#     print(f"Story #{i+1}:")
#     print("  ✅ Present:", sorted(present))
#     print("  ⚠️ Missing:", sorted(missing) if missing else "None")
#     print("  🌀 Extra:", sorted(extra) if extra else "None")
#     print("-" * 40)

path="D:/DH/Senior/Paperboy/src/api/chat_context.pkl"
with open(path, "rb") as f:
    old_context = pickle.load(f)
    
print(old_context)