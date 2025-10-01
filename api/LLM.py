import json
from google import generativeai as genai
from prompt.prompt import cluster_validation_prompt
from config_api import GEMINI_API_KEY, GEMINI_MODEL


# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


def validate_clusters(article_clusters):
    # Run evaluation for each pair
    results = []
    for cluster in article_clusters:

        response = model.generate_content([
            {"role": "user", "parts": [{"text": cluster_validation_prompt}]},
            {"role": "user", "parts": [{"list": cluster}]}
        ])


        # Parse model response as JSON
        try:
            evaluation = json.loads(response.text)
        except:
            evaluation = {"error": "Failed to validate", "raw": response.text}

        results.append({
            "question": q,
            "ground_truth": gt_answer,
            "generated_answer": ga,
            "evaluation": evaluation
        })
