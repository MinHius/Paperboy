import sys
sys.path.append("D:/DH/Senior/Paperboy") 
import os
import json
import pickle
import redis
import uuid

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def new_session_id() -> str:
    sid = str(uuid.uuid4())
    return sid


def load_context(session_id: str, content: str) -> str:
    context = r.get(session_id)
    if not context:
        r.set(session_id, content)
        r.expire(session_id, 86400)
    
    return context

def update_context(session_id: str, content: str) -> str:
    context = r.get(session_id)
    new_context = context + "\n" + content
    print(new_context)

    r.set(session_id, new_context)
    r.expire(session_id, 86400)
    

def json_to_paragraph(story_json: dict) -> str:
    # Handle if input is a list or dict
    segments = story_json.get("story", story_json) if isinstance(story_json, dict) else story_json
    parts = []

    for segment in segments:
        seg_type = segment.get("type")
        if isinstance(seg_type, dict):
            continue
        else:
            text = str(seg_type).strip()

        if seg_type == "header":
            parts.append(f"📰 {text.upper()}")
        elif seg_type == "paragraph":
            parts.append(text)
        elif seg_type == "quote":
            parts.append(f"❝ {text} ❞")
        elif seg_type == "list":
            items = segment.get("items", [])
            list_text = "\n".join(f" • {i}" for i in items)
            parts.append(f"{text}\n{list_text}" if text else list_text)
        else:
            parts.append(text)

    return "\n\n".join(p for p in parts if p)


def json_to_paragraph_audio(story_json: dict) -> str:
    # Handle if input is a list or dict
    segments = story_json.get("story", story_json) if isinstance(story_json, dict) else story_json
    content = ""

    for segment in segments:
        seg_type = segment.get("type")
        text = segment.get("text", "").strip()
        
        if seg_type == "paragraph":
            content += text + " "

    return content