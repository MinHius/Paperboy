import psycopg2
import numpy as np
from .config import HOST, PORT, DB, USER, PASSWORD, INTERVAL_TIME, SIMILARITY_THRESHOLD
import json
from datetime import datetime

# CONNECT
def get_connection():
    return psycopg2.connect(
        host = HOST,
        port = PORT,
        dbname = DB,
        user = USER,
        password = PASSWORD
    )


# UPSERT
def upsert_stories(stories):
    sql = """
    INSERT INTO stories_test (title, topic, story, thumbnail, figure, quotes, keyword, video, location, created_at, centroid, sentiment, author)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
      id = EXCLUDED.id,
      title = EXCLUDED.title,
      topic = EXCLUDED.topic,
      story = EXCLUDED.story,
      thumbnail = EXCLUDED.thumbnail,
      figure = EXCLUDED.figure,
      quotes = EXCLUDED.quotes,
      keyword = EXCLUDED.keyword,
      video = EXCLUDED.video,
      location = EXCLUDED.location,
      created_at = EXCLUDED.created_at,
      centroid = EXCLUDED.centroid,
      sentiment = EXCLUDED.sentiment,
      author = EXCLUDED.author;
    """

    rows = []
    for story in stories:
        if story.get("title"):
            rows.append((
                story["title"],
                story.get("topic"),
                json.dumps(story.get("story")),
                json.dumps(story.get("thumbnail")),
                json.dumps(story.get("figure")),
                json.dumps(story.get("quotes")),
                json.dumps(story.get("keyword")),
                json.dumps(story.get("video")),
                json.dumps(story.get("location")),
                datetime.now(),
                story.get("centroid"),
                story["sentiment"],
                json.dumps([story.get("author")])
            ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    print(f"Upserted {len(rows)} articles.")


# LOAD
def load_articles(inference_time):
    sql = """
    SELECT id, title, headline, url, thumbnail, section, published_at, description, body, author, embedding
    FROM articles 
    WHERE published_at BETWEEN %s::date - %s AND %s::date
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (inference_time, INTERVAL_TIME, inference_time))
            colnames = [desc[0] for desc in cur.description]  # get column names
            rows = cur.fetchall()
    
    # turn each row into a dict
    return [dict(zip(colnames, row)) for row in rows]


# LOAD
def load_stories():
    sql = """
    SELECT *
    FROM stories_test 
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            colnames = [desc[0] for desc in cur.description]  # get column names
            rows = cur.fetchall()
    
    # turn each row into a dict
    stories = [dict(zip(colnames, row)) for row in rows]   
    return stories


# GATHER
def search_similar_articles(clusters, threshold=SIMILARITY_THRESHOLD) -> list:
    article_groups = []
    sql = """
    SELECT id, body,
           1 - (embedding <#> %s::vector) AS similarity
    FROM articles
    WHERE 1 - (embedding <#> %s::vector) >= %s
    ORDER BY similarity DESC
    LIMIT 10;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for label, cluster in clusters.items():   # <-- fix here
                centroid = cluster["centroid"]
                if isinstance(centroid, np.ndarray):
                    centroid = centroid.astype(float).tolist()

                cur.execute(sql, (centroid, centroid, threshold))
                rows = cur.fetchall()
                article_groups.append([centroid, rows])
                print(f"Cluster {label}: Found {len(rows)} articles.")
        conn.commit()
    return article_groups


        