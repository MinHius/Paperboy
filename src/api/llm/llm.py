import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

import json
import time
import random
import time
import requests
import numpy as np
from google import generativeai as genai
from prompt.story import (
    cluster_validation_prompt, 
    synthesis_prompt,
    update_prompt,
    )
from cluster.utils import update_centroid
from api.config import GEMINI_API_KEY, GEMINI_MODEL
import os

# Initialize Gemini
if not GEMINI_API_KEY or not GEMINI_MODEL:
    raise
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


def validate_clusters(article_clusters: dict[str, list], marker: str, max_retries: int = 20):
    results = []

    for count, (centroid, contents) in enumerate(article_clusters, start=0):
        cluster_text = "\n\n".join(f"--- ARTICLE {i+1} (Cluster {count}) ---\n\n{c}" for i, c in enumerate(contents))
        retries = 0

        while retries < max_retries:
            try:
                response = model.generate_content([
                    {"role": "user", "parts": [{"text": cluster_validation_prompt}]},
                    {"role": "user", "parts": [{"text": cluster_text}]}
                ])

                try:
                    evaluation = json.loads(response.text)
                except:
                    raise ValueError("Invalid JSON")

                evaluation["centroid"] = centroid
                with open(f"1. evaluations/evaluation_{count}_{marker}.json", "w", encoding="utf-8") as f:
                    json.dump(evaluation, f, ensure_ascii=False, indent=2)
                print(f"Cluster {count} validated.")
                results.append(evaluation)
                break

            except Exception as e:
                retries += 1
                wait = min(60, 2 ** retries + random.random())
                print(f"Error validating cluster {count} (attempt {retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            fail_data = {"error": f"Failed after {max_retries} retries", "centroid": centroid}
            results.append(fail_data)
            with open(f"1. evaluations/evaluation_{count}_{marker}_failed.json", "w", encoding="utf-8") as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=2)

    return results


def synthesize_stories(groups: dict, marker: str, max_retries: int = 20):
    results = []

    for i, group in enumerate(groups, start=1):
        input_data = {
            "articles": [f"--- ARTICLE {n+1} ---\n{text}" for n, text in enumerate(group["content"])]
        }

        for attempt in range(max_retries):
            try:
                response = model.generate_content([
                    {"role": "user", "parts": [
                        {"text": synthesis_prompt},
                        {"text": json.dumps(input_data, ensure_ascii=False)}
                    ]}
                ])

                try:
                    synthesis = json.loads(response.text)
                    synthesis.update({
                        "centroid": group["centroid"],
                        "image": group["image"],
                        "keyword": group["keyword"],
                        "author": group["author"],
                        "cluster_size": group['cluster_size'],
                        
                    })
                    with open(f"2. synthesis/synthesis_{i}_{marker}.json", "w", encoding="utf-8") as f:
                        json.dump(synthesis, f, ensure_ascii=False, indent=2)
                    results.append(synthesis)
                    print(f"Cluster {i} ✅ synthesized.")
                    break
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON")

            except Exception as e:
                wait = min(60, 2 ** attempt + random.random())
                print(f"Cluster {i} retry {attempt+1}/{max_retries}: {e}. Waiting {wait:.1f}s.")
                time.sleep(wait)
        else:
            print(f"Cluster {i} ❌ failed after {max_retries} retries.")
            fail_data = {"error": f"Failed after {max_retries} retries"}
            results.append(fail_data)

    return results


def story_update(groups: dict, marker: str, max_retries: int = 20):
    results = []
    for i, group in enumerate(groups, start=0):
        if i == 0:
            tmp_article = group["article"]
            thumbnail = [art["thumbnail"] for art in tmp_article]
            embeddings = [
                json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
                for d in tmp_article
            ]
            embeddings = np.array(embeddings, dtype=np.float32)

            subset_art = [{"id": a["id"], "body": a["body"]} for a in tmp_article]
            subset_str = {"id": group["story"]["id"], "story": group["story"]["story"]}
            old_centroid = group["story"]["centroid"]
            cluster_size = group["story"]["cluster_size"]
            stories_json = json.dumps(list(subset_str), ensure_ascii=False)
            
            for attempt in range(max_retries):
                
                try:
                    response = model.generate_content(f"""
                    SYSTEM INSTRUCTIONS:
                    {update_prompt}

                    ARTICLES (a list of the id-article that might be update materials):
                    {subset_art}

                    The story that needed to be updated
                    {stories_json}
                    """)

                    try:
                        update = json.loads(response.text)
                        with open(f"6. updates/update{i}_{marker}.json", "w", encoding="utf-8") as f:
                            json.dump(update, f, ensure_ascii=False, indent=2)
                        if update:
                            new_centroid = update_centroid(
                                old_centroid,
                                embeddings,
                                cluster_size
                            )

                            results.append({
                                "id": subset_str["id"],
                                "story": update,
                                "thumbnail": thumbnail,
                                "centroid": new_centroid,
                            })
                        print(f"Story {i} ✅ updated.")
                        break
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON")

                except Exception as e:
                    wait = min(60, 2 ** attempt + random.random())
                    print(f"Story {i} retry {attempt+1}/{max_retries}: {e}. Waiting {wait:.1f}s.")
                    time.sleep(wait)
            else:
                print(f"Story {i} ❌ failed after {max_retries} retries.")

    return results









