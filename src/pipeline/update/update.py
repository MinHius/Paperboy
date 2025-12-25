import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

from datetime import datetime
from database.parade.database import search_similar_articles, search_similar_stories, upsert_updated
from api.chat.utils import json_to_paragraph
from cluster.clustering import reduce_dimensions, _perform_clustering
from api.llm.llm import story_update
from pipeline.update.utils import prepare_validation_input
from src.cluster.clustering import leader_clustering
from pathlib import Path
import pickle
import json
import numpy as np
import copy
import os

def update_stories(inference_time, articles: list[dict], embeddings: list, marker: str):
    os.makedirs(f"src/pickled_data/{marker}", exist_ok=True)
    """Step 1: Cluster articles"""
    article_clusters = leader_clustering(articles, embeddings, inference_time)
    
    """Step 2: Search similar stories"""
    similar_stories = search_similar_stories(article_clusters)
    print("Update step 2: Done!")
    
    """Step 3: Prepare validation input"""
    processed_input = prepare_validation_input(similar_stories)
    print("Update step 3: Done!")
    
    with open(f"./vector_clustering_size_{marker}", 'w', encoding='utf-8') as f:
        json.dump(processed_input, f, default=lambda o: o.tolist(), ensure_ascii=False, indent=4)

    """Step 4: Update validation"""
    update_story = story_update(processed_input, marker)
    print("Update step 4: Done!")
    
    """Step 5: Extraction and Update"""
    upsert_updated(update_story)

    with open(f"src/pickled_data/{marker}/update_validation.pkl", "wb") as f:
        pickle.dump(update_story, f)
        

    
