import markdown

def story_to_markdown(story_json):
    md_segments = []
    for seg in story_json:
        if seg["type"] == "header":
            md_segments.append(f"## {seg['text']}")
        elif seg["type"] == "paragraph":
            md_segments.append(seg["text"])
        elif seg["type"] == "quote":
            md_segments.append(f"> {seg['text']}")
        elif seg["type"] == "list":
            md_segments.append("\n".join([f"- {item}" for item in seg["items"]]))
    
    # Join with double newlines to separate paragraphs
    return "\n\n".join(md_segments)


def render_story(story_json):
    markdown_text = story_to_markdown(story_json)
    html = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    return html


