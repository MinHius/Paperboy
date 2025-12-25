from api.chat.utils import json_to_paragraph

def prepare_validation_input(similar_stories: list[dict]) -> list[dict]:
    update_validation_input = [{
        "article": [{"id": a['id'], 
                     "body": a['body'], 
                     "thumbnail": a.get("thumbnail", []),
                     "embedding": a["embedding"],
                    } for j, a in enumerate(group[0]) if a['body']],
        "story": {"id": group[1]["id"], 
                  "centroid": group[1]['centroid'], 
                  "cluster_size": group[1]['cluster_size'], 
                  "story": json_to_paragraph(group[1]['story'])}
    } for i, group in enumerate(similar_stories)]
    
    return update_validation_input