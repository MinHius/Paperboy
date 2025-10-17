SEARCH_BM25 = """
SELECT *
FROM articles
WHERE trail_text @@@ %s
LIMIT %s;
"""

SEARCH_BGEM3 = """
SELECT *
FROM articles
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

SEARCH_HYBRID = """
WITH bm25_ranked AS (
    SELECT id, title, section, trail_text, RANK() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT id, title, section, trail_text, paradedb.score(id) AS score
        FROM articles
        WHERE trail_text @@@ %s
        ORDER BY paradedb.score(id) DESC
        LIMIT %s
    ) AS bm25_score
),
semantic_search AS (
    SELECT id, title, section, trail_text, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM articles
    ORDER BY embedding <=> %s::vector
    LIMIT %s
)
SELECT
    COALESCE(semantic_search.id, bm25_ranked.id) AS id,
    (
        %s * (
            %s * COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
            %s * COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0)
        ) +
        %s * (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - articles.published_at)) / (86400 * 3)))
    ) AS score,

    articles.title,
    articles.section,
    articles.trail_text
FROM semantic_search
FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
JOIN articles ON articles.id = COALESCE(semantic_search.id, bm25_ranked.id)
ORDER BY score DESC
LIMIT %s;
"""
