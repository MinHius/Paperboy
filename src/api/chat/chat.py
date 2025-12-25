import time
import json
import random
import requests
from google import generativeai as genai
from prompt.chat import chat_prompt
from api.config import GEMINI_API_KEY_VNU, GEMINI_MODEL
import os

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY_VNU)
model = genai.GenerativeModel(GEMINI_MODEL)



def generate_chat(message: str, context: str, max_retries: int = 20):
    retries = 0
    result = []
    for attempt in range(max_retries):
        try:
            response = model.generate_content(f"""
                SYSTEM INSTRUCTIONS:
                {chat_prompt}

                STORY (a combination of the target story and conversation progress):
                {context}

                USER MESSAGE:
                {message}
                """
                )
            try:
                parsed = json.loads(response.text)
                result = parsed
                with open(f"5. chats/chat.json", "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                break
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON")
        except Exception as e:
            wait = min(6, attempt)
            print(f"Chat retry {attempt+1}/{max_retries}: {e}. Waiting {wait:.1f}s.")
            time.sleep(wait)


    return result
