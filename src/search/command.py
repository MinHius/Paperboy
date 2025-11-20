SEARCH_BM25 = """
SELECT id, title, topic, story, thumbnail, figure, video, location, created_at, centroid
FROM stories
WHERE title @@@ %s OR topic @@@ %s
ORDER BY paradedb.score(id) DESC
LIMIT %s;
"""

SEARCH_BGEM3 = """
SELECT id, title, topic, story, thumbnail, figure, video, location, created_at, centroid <=> %s::vector AS distance
FROM stories
ORDER BY distance DESC
LIMIT %s;
"""

SEARCH_HYBRID = """
WITH bm25_ranked AS (
    SELECT id, title, section, description, RANK() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT id, title, section, description, paradedb.score(id) AS score
        FROM stories
        WHERE title @@@ %s OR topic @@@ %s
        ORDER BY paradedb.score(id) DESC
        LIMIT %s
    ) AS bm25_score
),
semantic_search AS (
    SELECT id, title, section, description, RANK() OVER (ORDER BY centroid <=> %s::vector) AS rank
    FROM stories
    ORDER BY centroid <=> %s::vector
    LIMIT %s
)
SELECT
    COALESCE(semantic_search.id, bm25_ranked.id) AS id,
    (
        %s * (
            %s * COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
            %s * COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0)
        ) +
        %s * (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - stories.created_at)) / (86400 * 3)))
    ) AS score,
    stories.title,
    stories.topic,
    stories.created_at
FROM semantic_search
FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
JOIN stories ON stories.id = COALESCE(semantic_search.id, bm25_ranked.id)
ORDER BY score DESC
LIMIT %s;
"""
