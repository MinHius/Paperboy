import sys
sys.path.append("D:/DH/Senior/Paperboy") 

import json
import time
import random
import requests
from google import generativeai as genai
from prompt.story import (
    cluster_validation_prompt, 
    synthesis_prompt, 
    keyword_extraction
    )
from api.config import GEMINI_API_KEY, GEMINI_MODEL
import os

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


def validate_clusters(article_clusters, max_retries=20):
    Time = "2"
    results = []

    for count, (centroid, contents) in enumerate(article_clusters, start=1):
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
                with open(f"1. evaluations/evaluation_{count}_{Time}.json", "w", encoding="utf-8") as f:
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
            with open(f"1. evaluations/evaluation_{count}_{Time}_failed.json", "w", encoding="utf-8") as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=2)

    return results


def synthesize_stories(groups, max_retries=5):
    import json, random, time
    results, Time = [], "2"

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
                        
                    })
                    with open(f"2. synthesis/synthesis_{i}_{Time}.json", "w", encoding="utf-8") as f:
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


def update_stories(groups, max_retries = 20):
    pass









