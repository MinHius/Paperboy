import sys
sys.path.append("D:/DH/Senior/Paperboy") 

def json_to_paragraph(story_json):
    # Handle if input is a list or dict
    segments = story_json.get("story", story_json) if isinstance(story_json, dict) else story_json
    parts = []

    for segment in segments:
        seg_type = segment.get("type")
        text = segment.get("text", "").strip()

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