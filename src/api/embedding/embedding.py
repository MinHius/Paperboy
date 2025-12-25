import requests
from ..config import ACCOUNT_ID, API_TOKEN, BGEM3_URL



def embed_bgem3(text: str) -> dict:
    # Texts to embed
    payload = {
        "text": [text]
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(BGEM3_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        # print(data)
        
        # print("Embeddings:", data["result"])
    else:
        print("Error:", response.status_code, response.text)
    
    return data["result"] if response.status_code == 200 else None
