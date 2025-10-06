import sys
sys.path.append("D:/DH/Senior/Paperboy") 

import json
import time
import random
from google import generativeai as genai
from src.prompt.prompt import cluster_validation_prompt
from src.api.config_api import GEMINI_API_KEY, GEMINI_MODEL


# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)



def validate_clusters(article_clusters, max_retries=6):
    results = []

    for count, group in enumerate(article_clusters, start=1):
        retries = 0
        cluster_text = "\n\n".join(
            f"--- ARTICLE {i+1} (ID: {cid}) ---\n{text}"
            for i, (cid, text) in enumerate(group)
        )

        while retries < max_retries:
            try:
                # --- call with timeout ---
                response = model.generate_content(
                    [
                        {"role": "user", "parts": [{"text": cluster_validation_prompt}]},
                        {"role": "user", "parts": [{"text": cluster_text}]}
                    ]
                )

                # --- try parsing ---
                try:
                    evaluation = json.loads(response.text)
                except Exception:
                    evaluation = {"error": "Failed to parse JSON", "raw": response.text}

                # --- save result ---
                with open(f"evaluations/evaluation_{count}.json", "w", encoding="utf-8") as f:
                    json.dump(evaluation, f, ensure_ascii=False, indent=2)

                print(f"Cluster {count} validated.")
                results.append(evaluation)
                break  # success, exit retry loop

            except Exception as e:
                retries += 1
                wait = min(60, 2 ** retries + random.random())
                print(f"Error validating cluster {count} (attempt {retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            # ran out of retries
            fail_data = {"error": f"Failed after {max_retries} retries"}
            results.append(fail_data)
            with open(f"evaluations/evaluation_{count}_failed.json", "w", encoding="utf-8") as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=2)

    return results



def synthesize_stories(validation_results):
    results = []

    for count, group in enumerate(validation_results, start=1):
        retries = 0
        cluster_text = "\n\n".join(
            f"--- ARTICLE {i+1} (ID: {cid}) ---\n{text}"
            for i, (cid, text) in enumerate(group)
        )

        while retries < max_retries:
            try:
                # --- call with timeout ---
                response = model.generate_content(
                    [
                        {"role": "user", "parts": [{"text": cluster_validation_prompt}]},
                        {"role": "user", "parts": [{"text": cluster_text}]}
                    ]
                )

                # --- try parsing ---
                try:
                    evaluation = json.loads(response.text)
                except Exception:
                    evaluation = {"error": "Failed to parse JSON", "raw": response.text}

                # --- save result ---
                with open(f"evaluations/evaluation_{count}.json", "w", encoding="utf-8") as f:
                    json.dump(evaluation, f, ensure_ascii=False, indent=2)

                print(f"Cluster {count} validated.")
                results.append(evaluation)
                break  # success, exit retry loop

            except Exception as e:
                retries += 1
                wait = min(60, 2 ** retries + random.random())
                print(f"Error validating cluster {count} (attempt {retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            # ran out of retries
            fail_data = {"error": f"Failed after {max_retries} retries"}
            results.append(fail_data)
            with open(f"evaluations/evaluation_{count}_failed.json", "w", encoding="utf-8") as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=2)

    return results