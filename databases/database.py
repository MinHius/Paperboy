import psycopg2
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
    SELECT * FROM articles WHERE published_at BETWEEN %s::date - %s AND %s::date
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (inference_time, INTERVAL_TIME, inference_time))   
            rows = cur.fetchall()                 
    return rows
    

# GATHER
def search_similar_articles(clusters, threshold=SIMILARITY_THRESHOLD) -> list:
    article_groups = []
    sql = """
    SELECT id, title, section, trail_text, body, 1 - (embedding <#> %s) AS similarity
    FROM articles
    WHERE 1 - (embedding <#> %s) >= %s
    ORDER BY similarity DESC
    LIMIT 15;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for cluster in clusters:
                cur.execute(sql, (cluster["centroid"], cluster["centroid"], threshold))
                rows = cur.fetchall()
                article_groups.append(rows)
                print(f"Cluster {cluster}: Found {len(rows)} articles.")
        conn.commit()
    return article_groups

        