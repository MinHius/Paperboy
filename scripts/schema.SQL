CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    headline TEXT,
    url TEXT,
    thumbnail TEXT,
    section TEXT,
    published_at TIMESTAMP,
    description TEXT,
    body TEXT,
    author TEXT,
    text TEXT GENERATED ALWAYS AS (title || '\n\n' || COALESCE(description, '')) STORED,
    embedding VECTOR(1024)  -- semantic embedding size (OpenAI default)
);


CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE stories (
    id UUID PRIMARY KEY,
    title TEXT,
    topic TEXT,
    story JSONB,
    thumbnail JSONB,
    figure JSONB,
    quotes JSONB,
    keyword JSONB,
    video JSONB,
    location JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    centroid vector(1024),
    sentiment TEXT,
    author JSONB,
    trending DECIMAL DEFAULT 10,
    cluster_size INT
);


CREATE TABLE article-story (
    id TEXT PRIMARY KEY,
    title TEXT,
    headline TEXT,
    url TEXT,
    thumbnail TEXT,
    section TEXT,
    published_at TIMESTAMP,
    description TEXT,
    body TEXT,
    author TEXT,
    text TEXT GENERATED ALWAYS AS (title || '\n\n' || COALESCE(description, '')) STORED,
    embedding VECTOR(1024)  -- semantic embedding size (OpenAI default)
);


CREATE TABLE stories_test (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    topic TEXT,
    story JSONB,
    thumbnail JSONB,
    figure JSONB,
    quotes JSONB,
    keyword JSONB,
    video JSONB,
    location JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    centroid vector(1024),
    sentiment TEXT,
    author JSONB,
    trending DECIMAL DEFAULT 10,
    cluster_size INT
);


CREATE TABLE article_story (
    id UUID PRIMARY KEY,
    title TEXT,
    headline TEXT,
    url TEXT,
    thumbnail TEXT,
    section TEXT,
    published_at TIMESTAMP,
    description TEXT,
    body TEXT,
    author TEXT,
    text TEXT,
    embedding VECTOR(1024)  -- semantic embedding size (OpenAI default)
);