import sys
sys.path.append("D:/DH/Senior/Paperboy") 

import json
import time
import requests
from api.config import YOUTUBE_API_KEY
from googleapiclient.discovery import build


def fetch_youtube_videos(title: str, max_results=4, max_retries=20) -> list[dict]:
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    results = []
    if not isinstance(title, str):
        title = str(title)
    
    for attempt in range(1, max_retries + 1):
        try:
            request = youtube.search().list(
                q=f"{title} news OR report OR documentary",
                part="id,snippet",
                maxResults=max_results,
                type="video"
            )
            response = request.execute()

            if response.get("items"):
                for item in response["items"]:
                    video_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    results.append({
                        "name": title,
                        "link": f"https://www.youtube.com/embed/{video_id}"
                    })
            break  # stop retry loop if success

        except Exception as e:
            print(f"Error fetching '{title}' (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(attempt)  # delay = current attempt in seconds
            else:
                print(f"Failed to fetch after {max_retries} attempts.")
    
    return results


