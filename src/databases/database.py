import psycopg2
import numpy as np
from .config_database import HOST, PORT, DB, USER, PASSWORD, INTERVAL_TIME, SIMILARITY_THRESHOLD

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
def upsert_articles(all_articles):
    sql = """
    INSERT INTO articles (id, title, url, section, published_at, trail_text, body, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, all_articles)  # batch insert
        conn.commit()
    print(f"Upserted {len(all_articles)} articles.")


# LOAD
def load_articles(inference_time):
    sql = """
    SELECT id, title, url, section, published_at, trail_text, body, embedding
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
                article_groups.append(rows)
                print(f"Cluster {label}: Found {len(rows)} articles.")
        conn.commit()
    return article_groups


        