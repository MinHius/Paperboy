import requests
from .config_api import ACCOUNT_ID, API_TOKEN


url = f"https://api.cloudflare.com/client/v4/accounts/b1165c4f01cd15f55afac8553545b30b/ai/run/@cf/baai/bge-m3"


def embed_bgem3(text: str):
    # Texts to embed
    payload = {
        "text": [text]
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        # print(data)
        
        # print("Embeddings:", data["result"])
    else:
        print("Error:", response.status_code, response.text)
    
    return data["result"] if response.status_code == 200 else None
