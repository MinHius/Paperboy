import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from .config import HOST, PORT, DB, USER, PASSWORD, INTERVAL_TIME, SIMILARITY_THRESHOLD, UPDATE_THRESHOLD
import json
from datetime import datetime

# CONNECT
def get_connection():
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DB,
        user=USER,
        password=PASSWORD
    )
    register_vector(conn)   # <-- REQUIRED
    return conn


# UPSERT
def upsert_stories(stories) -> None:
    sql = """
    INSERT INTO stories (id, title, topic, story, thumbnail, figure, quotes, keyword, video, location, created_at, centroid, sentiment, author, trending, cluster_size)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
      author = EXCLUDED.author,
      trending = EXCLUDED.trending,
      cluster_size = EXCLUDED.cluster_size;
    """

    rows = []
    for story in stories:
        if story.get("title"):
            rows.append((
                story.get("id"),
                story.get("title"),
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
                story.get("sentiment"),
                json.dumps([story.get("author")]),
                10,
                story.get("cluster_size"),
            ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    print(f"Upserted {len(rows)} articles.")


# LOAD
def load_articles(inference_time) -> list[dict]:
    sql = """
    SELECT id, title, headline, url, thumbnail, section, published_at, description, body, author, embedding
    FROM articles
    WHERE published_at BETWEEN %s::date - %s AND %s::date
    AND embedding IS NOT NULL;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (inference_time, INTERVAL_TIME, inference_time))
            colnames = [desc[0] for desc in cur.description]  # get column names
            rows = cur.fetchall()
    
    # turn each row into a dict
    return [dict(zip(colnames, row)) for row in rows]


# LOAD
def load_stories() -> list[dict]:
    sql = """
    SELECT *
    FROM stories;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            colnames = [desc[0] for desc in cur.description]  # get column names
            rows = cur.fetchall()
    
    # turn each row into a dict
    stories = [dict(zip(colnames, row)) for row in rows]   
    return stories


# LOAD DISPLAY
def load_stories_display() -> list[dict]:
    sql = """
    SELECT *
    FROM stories
    WHERE video IS NOT NULL 
    AND video::text NOT IN ('[]', '{}', 'null')
    AND title != %s
    ORDER BY created_at DESC
    LIMIT 50;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, ("No News Articles Provided for Synthesis",))
            colnames = [desc[0] for desc in cur.description]  # get column names
            rows = cur.fetchall()
    
    # turn each row into a dict
    stories = [dict(zip(colnames, row)) for row in rows]   
    return stories


# GATHER ARTICLES
def search_similar_articles(clusters: dict[str, list], threshold: int = SIMILARITY_THRESHOLD) -> list[str, list]:
    article_groups = []
    sql = """
    SELECT id, title, headline, url, thumbnail, section, published_at, description, body, author, embedding
    FROM articles
    WHERE 1.0 - (embedding <#> %s::vector) >= %s
    ORDER BY 1.0 - (embedding <#> %s::vector) DESC
    LIMIT 10;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for label, cluster in clusters.items(): 
                centroid = cluster["centroid"]
                if isinstance(centroid, np.ndarray):
                    centroid = centroid.astype(float).tolist()

                cur.execute(sql, (centroid, threshold, centroid))
                colnames = [desc[0] for desc in cur.description]  # get column names
                rows = cur.fetchall()
                articles = [dict(zip(colnames, row)) for row in rows]  
                
                article_groups.append([centroid, articles])
                print(f"Cluster {label}: Found {len(articles)} articles.")
        conn.commit()
    
    return article_groups


# REMOVE
def remove_used_articles(article_clusters: list[str]) -> None:
    sql_delete = """
        DELETE FROM articles
        WHERE id = %s
        RETURNING
            id,
            title,
            headline,
            url,
            thumbnail,
            section,
            published_at,
            description,
            body,
            author,
            text,
            embedding;
    """
    
    sql_assert = """
        INSERT INTO article_story (
        id, title, headline, url, thumbnail, section,
        published_at, description, body, author, text, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for article_ids in article_clusters:
                for id in article_ids:
                    try:
                        cur.execute(sql_delete, (id,))
                        row = cur.fetchone()
                        if not row:
                            print(f"ID {id} not found")
                            continue

                        cur.execute(sql_assert, row)
                    except:
                        print(f"Error id: {id}")
                        continue
        conn.commit()


# GATHER STORIES
def search_similar_stories(article_clusters: dict[str, list], threshold: int = UPDATE_THRESHOLD) -> list[str, list, list]:
    story_groups = []
    sql = """
    SELECT id, story, centroid, cluster_size,
           1 - (centroid <#> %s::vector) AS similarity
    FROM stories
    WHERE 1 - (centroid <#> %s::vector) >= %s
    ORDER BY created_at DESC, similarity DESC
    LIMIT 1;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for label, cluster in article_clusters.items(): 
                centroid = cluster["centroid"]
                articles = [
                    {
                        "id": a["id"],
                        "body": a["body"],
                        "embedding": a["embedding"],
                    } for a in cluster["articles"]
                ]
                if isinstance(centroid, np.ndarray):
                    centroid = centroid.astype(float).tolist()

                cur.execute(sql, (centroid, centroid, threshold))
                colnames = [desc[0] for desc in cur.description] 
                rows = cur.fetchall()
                stories = [dict(zip(colnames, row)) for row in rows]
                
                story_groups.append([articles, stories[0]])
                print(f"Cluster {label}: Found {len(rows)} stories.")
        conn.commit()
        
    return story_groups


# UPDATE
def upsert_updated(stories):
    sql = """
    UPDATE stories
    SET
        story = %s,
        thumbnail = thumbnail || %s::jsonb,
        updated_at = %s,
        centroid = %s
    WHERE id = %s;
        """

    rows = []
    for story in stories:
        if story.get("id"):
            rows.append((
                json.dumps(story.get("story")),
                json.dumps(story.get("thumbnail")),
                datetime.now(),
                story.get("centroid"),
                story.get("id"),
            ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    print(f"Updated {len(rows)} stories.")







