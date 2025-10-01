CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    section TEXT,
    published_at TIMESTAMP,
    trail_text TEXT,
    body TEXT,
    text_field TEXT GENERATED ALWAYS AS (title || '\n' || COALESCE(trail_text, '')) STORED,
    embedding VECTOR(1024)  -- semantic embedding size (OpenAI default)
);
