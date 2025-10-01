import re
import time
import json
import heapq
import pickle
import logging
import psycopg2
import numpy as np
import unicodedata
from rapidfuzz import fuzz, process

from rank_bm25 import BM25Okapi
from databases.database import DatabaseService
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone


from utils import setup_logging
from underthesea import word_tokenize
from pipeline.search.BGEM3_API import embed_bgem3
from pipeline.main.database import load_articles, load_recent_stories
from pipeline.search.config import (
    TOKENIZED_FILE, TOKENIZED_STR,
    BM25, BM25_STR,
    ARTICLES, STORIES,
    QUERY_LEN_MAX, QUERY_LEN_MIN,
    FUZZY_WEIGHT, TOP_K,
    TIME_TAU, RELEVANCE_W, TIME_W,
)


# Set up logging
setup_logging()
logger = logging.getLogger(__name__)



def get_embedding_materials_art(db: DatabaseService, reference_time: datetime) -> list:
    """ 
    Fetch articles for BM25 embedding. 
    """
    
    seeding = bool(reference_time)
    if not seeding:
        reference_time = datetime.now()

    logger.info("Loading articles and embeddings...")
    _, articles = load_articles(
        db,
        reference_time - timedelta(days=300)
    )
    logger.info(f"Loaded {len(articles)} articles")
    
    
    list_of_text = []
    for article in articles:
        list_of_text.append({
            "id": article.id,
            "title": article.title,
            "description": article.description,
            "url": article.url,
            "text": article.title + "  " + article.description,
            "publish_time": article.published_at,
            "embedding": article.embedding 
            })

    logger.debug(f"Found {len(list_of_text)} articles.")
    print(len(list_of_text))

    with open(ARTICLES, "wb") as f:
        pickle.dump(list_of_text, f)  
    logger.debug(f"Pickled {len(list_of_text)} articles.")
     
def get_embedding_materials_str(db: DatabaseService, reference_time: datetime) -> list:
    """
    Fetch stories for BM25 embedding.
    """
    
    seeding = bool(reference_time)
    if not seeding:
        reference_time = datetime.now()

    logger.info("Loading stories and embeddings...")
    stories = load_recent_stories(
        db,
        reference_time - timedelta(days=300)
    )
    logger.info(f"Loaded {len(stories)} stories")
    
    list_of_text = []
    for story in stories:
        list_of_text.append({
            "id": story.id,
            "title": story.title,
            "overview": story.overview,
            "text": story.title + "  " + story.overview,
            "figures": story.key_figures,
            "stats": story.key_stats,
            "keywords": story.keywords,
            "embedding": story.centroid,
            "updated_at": story.updated_at,
            })

    logger.debug(f"Found {len(list_of_text)} stories.")
    print(len(list_of_text))
    
    with open(STORIES, "wb") as f:
        pickle.dump(list_of_text, f)  
    logger.debug(f"Pickled {len(list_of_text)} stories.")



def embed_articles_bm25() -> list[dict]:
    """
    Embed articles using BM25.
    """
    
    with open(ARTICLES, "rb") as f:
        list_of_text = pickle.load(f)
    
    tokenized_docs = []
    # Tokenize the text for BM25
    if not list_of_text:
        return []
    
    for article in list_of_text:
        tokens = word_tokenize(article["text"], format="text").lower().split()
        raw_text = remove_diacritics(article["text"])
        tokens_raw = word_tokenize(raw_text, format="text").lower().split()
        
        combined_tokens = tokens + tokens_raw
        tokenized_docs.append(combined_tokens)
        
        raw_title = remove_diacritics(article["title"])
        article["title_token"] = word_tokenize(raw_title, format="text").lower().split()
        
    # Pickle the tokenized documents for later use
    with open(TOKENIZED_FILE, "wb") as f:
        pickle.dump(tokenized_docs, f)  
    logger.debug(f"Tokenized {len(tokenized_docs)} articles.")
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)
    logger.debug("BM25 model created.")
    
    # Precompute scores for all terms at once
    all_terms = set(term for doc in tokenized_docs for term in doc)
    term_scores = {term: bm25.get_scores([term]) for term in all_terms}

    sparse_docs = []
    for i, doc_tokens in enumerate(tokenized_docs):
        doc_score_dict = {term: term_scores[term][i] for term in set(doc_tokens) if term_scores[term][i] != 0}
        sparse_docs.append({
            "id": list_of_text[i]["id"],
            "title": list_of_text[i]["title"],
            "text": list_of_text[i]["text"],
            "bm25_dict": doc_score_dict,
            "date": list_of_text[i]["updated_time"],
            "title_token": list_of_text[i]["title_token"]
        })
    logger.info(f"Embedded {len(sparse_docs)} articles with BM25.") 
    
    
    # Pickle sparse vector dict for quick score computing.
    with open(BM25, "wb") as f:
        pickle.dump(sparse_docs, f)  
    logger.debug(f"Pickled sparse vector dict.")

def embed_stories_bm25() -> list[dict]:
    """
    Embed stories using BM25.
    """
    
    with open(STORIES, "rb") as f:
        list_of_text = pickle.load(f)
    
    tokenized_docs = []
    # Tokenize the text for BM25
    if not list_of_text:
        return []
    for story in list_of_text:
        tokens = word_tokenize(story["text"], format="text").lower().split()
        raw_text = remove_diacritics(story["text"])
        tokens_raw = word_tokenize(raw_text, format="text").lower().split()
        
        combined_tokens = tokens + tokens_raw
        tokenized_docs.append(combined_tokens)
        
        raw_title = remove_diacritics(story["title"])
        story["title_token"] = word_tokenize(raw_title, format="text").lower().split()
        
    # Pickle the tokenized documents for later use
    with open(TOKENIZED_STR, "wb") as f:
        pickle.dump(tokenized_docs, f)  
    logger.debug(f"Tokenized {len(tokenized_docs)} stories.")
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)
    logger.debug("BM25 model created.")
    
    # Precompute scores for all terms at once
    all_terms = set(term for doc in tokenized_docs for term in doc)
    term_scores = {term: bm25.get_scores([term]) for term in all_terms}

    sparse_docs = []
    for i, doc_tokens in enumerate(tokenized_docs):
        doc_score_dict = {term: term_scores[term][i] for term in set(doc_tokens) if term_scores[term][i] != 0}
        sparse_docs.append({
            "id": list_of_text[i]["id"],
            "title": list_of_text[i]["title"],
            "text": list_of_text[i]["text"],
            "keywords": list_of_text[i]["keywords"],
            "figures": list_of_text[i]["figures"],
            "stats": list_of_text[i]["stats"],
            "bm25_dict": doc_score_dict,
            "date": list_of_text[i]["updated_at"],
            "title_token": list_of_text[i]["title_token"]
        })
    logger.info(f"Embedded {len(sparse_docs)} stories with BM25.") 
    
    
    # Pickle sparse vector dict for quick score computing.
    with open(BM25_STR, "wb") as f:
        pickle.dump(sparse_docs, f)  
    logger.debug(f"Pickled sparse vector dict.")

def embed_non_diacritics_bgem3():
    count = 1
    
    # with open("articles_stories.pkl", "rb") as f: 
    #     current = pickle.load(f)
    
    with open(STORIES, "rb") as f:
        list_of_text = pickle.load(f)
    
    for doc in list_of_text:
        raw_vietnamese = remove_diacritics(doc["text"])
        raw_viet_embedded = embed_bgem3(raw_vietnamese)
        data = raw_viet_embedded.get("data", None)
        doc["raw_embedding"] = data[0]
        print(f"Embedded {count} raw articles.")
        count += 1
    
    # turn list into id→dict lookup
    # lookup = {item["id"]: item for item in current}
    
    # for article in new: 
    #     article_id = article["id"]
    #     article["raw_embedding"] = lookup[article_id]["embedding"]
    #     print(f"Embedded {count} raw articles.") 
    #     count += 1 
        
    # with open(ARTICLES, "wb") as f: 
    #     pickle.dump(new, f) 
    # logger.debug(f"Embedded {len(new)} raw articles.") 
    
    
    with open(STORIES, "wb") as f:
        pickle.dump(list_of_text, f)  
        logger.debug(f"Embedded {len(list_of_text)} raw data.")
   
           

def upsert_embeddings_to_db(embedded_articles: list[dict]) -> None:
    """ Upsert the BM25 embeddings to the database. """
    conn = psycopg2.connect(
        dbname="bm25test",
        user="postgres",
        password="minhhieu888",
        host="localhost",
        port=5434
    )
    cur = conn.cursor()

    # Insert raw docs and tokens
    for _, article in enumerate(embedded_articles, start=1):
        embedding_bm25 = "[" + ",".join(map(str, article["embedding_bm25"].tolist())) + "]"
        embedding = "[" + ",".join(map(str, article["embedding"].tolist())) + "]"
        cur.execute(
            "INSERT INTO documents (id, title, description, url, bm25_vector, embedding) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (
                article["id"], 
                article["title"], 
                article["description"], 
                article["url"], 
                json.dumps(article["embedding_bm25"].tolist()),  # as list, no json.dumps
                article["embedding"].tolist()      
            )
        )

    conn.commit()
    cur.close()
    conn.close()



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))    
        
def score_doc(doc, fuzzy_2_gram):
    if not doc["title_token"]:
        return 0.0
    token_scores = [
        process.extractOne(q, doc["title_token"], scorer=fuzz.ratio)[1]
        for q in fuzzy_2_gram
    ]
    return sum(token_scores) / len(token_scores) / 100.0
    
    

def compute_weights(query_tokens, 
                    sparse_max=1.0, sparse_min=0.0, 
                    len_min=QUERY_LEN_MIN, len_max=QUERY_LEN_MAX):
    """
    Compute continuous sparse/dense weights based on query length.

    Args:
        query_tokens: list of tokens in the query
        sparse_max: max weight for sparse (shortest queries)
        sparse_min: min weight for sparse (longest queries)
        len_min: query length to start decreasing sparse weight
        len_max: query length where sparse weight hits sparse_min
    """
    
    qlen = len(query_tokens)
    # Linear interpolation
    slope = (sparse_min - sparse_max) / (len_max - len_min)
    sparse_weight = sparse_max + slope * (qlen - len_min)
    
    dense_weight = 1 - sparse_weight
    return sparse_weight, dense_weight

def time_score(publish_time: datetime, now: datetime = None, tau: float = TIME_TAU) -> float:
    if now is None:
        now = datetime.now(timezone.utc)

    # Normalize publish_time
    if publish_time.tzinfo is None:
        publish_time = publish_time.replace(tzinfo=timezone.utc)

    delta_hours = (now - publish_time).total_seconds() / 3600.0
    weight = pow(2.71828, -delta_hours / tau)
    return max(0.0, min(1.0, weight))

def is_raw_vietnamese(text: str) -> bool:
    """
    Returns True if the text looks like 'raw' Vietnamese (no diacritics),
    False if it has Vietnamese diacritic vowels or tone marks.
    Works for both precomposed (NFC) and decomposed (NFD) Unicode.
    """
    # Vietnamese diacritic tone marks in combining form
    VIETNAMESE_COMBINING_MARKS = {
        "\u0300",  # grave ( ` )  → à
        "\u0301",  # acute ( ´ ) → á
        "\u0303",  # tilde ( ~ ) → ã
        "\u0309",  # hook above ( ̉ ) → ả
        "\u0323",  # dot below ( . ) → ạ
    }

    # Special Vietnamese base letters with diacritics
    VIETNAMESE_SPECIALS = "ăâđêôơưĂÂĐÊÔƠƯ"

    # Normalize for consistency
    text_nfc = unicodedata.normalize("NFC", text)  # precomposed
    text_nfd = unicodedata.normalize("NFD", text)  # decomposed

    # 1. Check precomposed letters with diacritics (e.g., ũ, ế, ộ)
    for ch in text_nfc:
        name = unicodedata.name(ch, "")
        if "WITH" in name or ch in VIETNAMESE_SPECIALS:
            return False

    # 2. Check decomposed combining marks (e.g., u + ́ → ú)
    for ch in text_nfd:
        if ch in VIETNAMESE_COMBINING_MARKS:
            return False

    return True

def remove_diacritics(text: str) -> str:
    # Normalize to decomposed form
    nfkd = unicodedata.normalize("NFD", text)
    # Remove combining diacritics
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def smart_link(tokens):
    linked = []
    word_re = re.compile(r"^\w+$", re.UNICODE)  # covers accented chars too
    
    for i, tok in enumerate(tokens):
        # Case 1: range or fraction (e.g. 2/9, 1-3)
        if re.match(r"^\d+[-/]\d+$", tok):
            linked.append(tok)
        # Case 2: number after a word → merge
        elif tok.isdigit() and i > 0 and word_re.match(tokens[i-1]):
            linked[-1] = linked[-1] + "_" + tok
        else:
            linked.append(tok)
    print(linked)
    return linked



def query_bm25(query: str, top_k: int, sparse_docs: dict, fuzzy_weight: int = FUZZY_WEIGHT) -> list[dict]:
    """ Query the BM25 database for the top K articles matching the query. """ 
    
    # Tokenize the query
    tokenized_query = word_tokenize(query, format="text").lower().split()
    if len(tokenized_query) == 2:
        tokenized_query = tokenized_query + ["_".join(tokenized_query)] 
    else: 
        tokenized_query = smart_link(tokenized_query)
    print(tokenized_query)
    results = []
    
    if is_raw_vietnamese(query):
        for doc in sparse_docs:
            # BM25 score (fast dictionary lookup)
            score_bm25 = sum(doc["bm25_dict"].get(term, 0) for term in tokenized_query)
            
            # Fuzzy score (only if fuzzy_weight > 0)
            if fuzzy_weight > 0:
                doc_tokens = doc["title_token"]  # precomputed once at index time
                if not doc_tokens:
                    fuzzy_scores = 0.0
                else:
                    token_scores = [
                        process.extractOne(q, doc_tokens, scorer=fuzz.partial_ratio)[1]
                        for q in [query]
                    ]
                    if all(score == 100 for score in token_scores):
                        fuzzy_scores = sum(token_scores) / len(token_scores) / 100.0
                    else:
                        fuzzy_scores = 0.0
            else:
                fuzzy_scores = 0.0
                

            # Combine weighted score
            score = (1 - fuzzy_weight) * score_bm25 + fuzzy_weight * fuzzy_scores
            results.append((doc["id"], doc["text"], score, doc["date"]))
    else:
        for doc in sparse_docs:
            # BM25 score (fast dictionary lookup)
            score = sum(doc["bm25_dict"].get(term, 0) for term in tokenized_query)
            results.append((doc["id"], doc["text"], score, doc["date"]))

    # Sort
    results = heapq.nlargest(top_k, results, key=lambda x: x[2])
    
    return results, tokenized_query

def query_bm25_backup(query: str, top_k: int, sparse_docs, fuzzy_weight=FUZZY_WEIGHT) -> list[dict]:
    # """ Query the BM25 database for the top K articles matching the query. """ 
    
    # # Tokenize the query
    # tokenized_query = word_tokenize(query, format="text").lower().split()
    # query_tokens = query.lower().split()
    # print(tokenized_query)
    # results = []
    
    # for doc in sparse_docs:
    #     # BM25 score (fast dictionary lookup)
    #     score_bm25 = sum(doc["bm25_dict"].get(term, 0) for term in tokenized_query)

    #     # Fuzzy score (only if fuzzy_weight > 0)
    #     if fuzzy_weight > 0:
    #         doc_tokens = doc["title_token"]  # precomputed once at index time
    #         if not doc_tokens:
    #             fuzzy_scores = 0.0
    #         else:
    #             token_scores = [
    #                 process.extractOne(q, doc_tokens, scorer=fuzz.ratio)[1]
    #                 for q in tokenized_query
    #             ]
    #             fuzzy_scores = sum(token_scores) / len(token_scores) / 100.0
    #     else:
    #         fuzzy_scores = 0.0

    #     # Combine weighted score
    #     score = (1 - fuzzy_weight) * score_bm25 + fuzzy_weight * fuzzy_scores

    #     results.append((doc["id"], doc["text"], score, doc["date"]))
#     # Sort
#     results = heapq.nlargest(top_k, results, key=lambda x: x[2])
    
#     return results[:top_k], tokenized_query
    pass

def query_bm25_str(query: str, top_k: int, sparse_docs, fuzzy_weight: int = FUZZY_WEIGHT) -> list[dict]:
    """ Query the BM25 database for the top K articles matching the query. """ 
    
    # Tokenize the query
    tokenized_query = word_tokenize(query, format="text").lower().split()
    if len(tokenized_query) == 2:
        tokenized_query = tokenized_query + ["_".join(tokenized_query)] 
    else: 
        tokenized_query = smart_link(tokenized_query)
    print(tokenized_query)
    results = []
        
    if is_raw_vietnamese(query):
        for doc in sparse_docs:
            # BM25 score (fast dictionary lookup)
            score_bm25 = sum(doc["bm25_dict"].get(term, 0) for term in tokenized_query)
            
            # Fuzzy score (only if fuzzy_weight > 0)
            if fuzzy_weight > 0:
                doc_tokens = doc["title_token"] # precomputed once at index time
                if not doc_tokens:
                    fuzzy_scores = 0.0
                else:
                    token_scores = [
                        process.extractOne(q, doc_tokens, scorer=fuzz.ratio)[1]
                        for q in [query]
                    ]
                    if all(score == 100 for score in token_scores):
                        fuzzy_scores = sum(token_scores) / len(token_scores) / 100.0
                    else:
                        fuzzy_scores = 0.0
            else:
                fuzzy_scores = 0.0
                

            # Combine weighted score
            score = (1 - fuzzy_weight) * score_bm25 + fuzzy_weight * fuzzy_scores
            results.append((doc["id"], doc["text"], score, doc["keywords"], doc["figures"], doc["stats"], doc["date"]))
    else:
        for doc in sparse_docs:
            # BM25 score (fast dictionary lookup)
            score = sum(doc["bm25_dict"].get(term, 0) for term in tokenized_query)
            results.append((doc["id"], doc["text"], score, doc["keywords"], doc["figures"], doc["stats"], doc["date"]))

    # Sort
    results = heapq.nlargest(top_k, results, key=lambda x: x[2])
    
    return results, tokenized_query

def query_bgem3(query: str, top_k: int, list_of_text: list, sparse_docs: dict, fuzzy_weight = FUZZY_WEIGHT) -> list[dict]:
    """
    Query the BGEM3 API with cosine similarity and soft fuzzy scoring.

    fuzzy_weight: fraction of score contributed by fuzzy matching (0–1)
    """
    print("Query:", query)
    tokenized_query = word_tokenize(query, format="text").lower().split()
    fuzzy_2_gram = ["_".join(tokenized_query)] if len(tokenized_query) == 2 else smart_link(tokenized_query)

    # Embed query
    embedding_bgem3 = embed_bgem3(query)
    data = embedding_bgem3.get("data", None)
    if not data:
        logger.error("Failed to get BGEM3 embeddings.")
        return []

    query_vec = np.array(data[0], dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        query_vec = np.zeros_like(query_vec)
    else:
        query_vec /= norm

    # Select embedding field
    if is_raw_vietnamese(query):
        embeddings = np.array([a["raw_embedding"] for a in list_of_text], dtype=np.float32)
    else:
        embeddings = np.array([a["embedding"] for a in list_of_text], dtype=np.float32)
    embeddings = np.nan_to_num(
        embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    # Compute cosine similarities
    sims = embeddings @ query_vec

    if len(tokenized_query) <= 2:
        # Compute fuzzy scores
        with ThreadPoolExecutor() as ex:
            fuzzy_scores = list(ex.map(score_doc, sparse_docs, fuzzy_2_gram))

        fuzzy_scores = np.array(fuzzy_scores, dtype=np.float32)

        # Combine cosine similarity and fuzzy score
        final_scores = (1 - fuzzy_weight) * sims + fuzzy_weight * fuzzy_scores
    else:
        final_scores = sims

    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    # Top-k selection
    top_idx = np.argpartition(-final_scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-final_scores[top_idx])]
    
    results = [
        (list_of_text[i]["id"], list_of_text[i]["text"], float(final_scores[i]), list_of_text[i]["date"])
        for i in top_idx
    ]

    print(f"Found {len(results)} articles matching the query.")
    return results

def query_bgem3_str(query: str, top_k: int, list_of_text: list, sparse_docs: dict, fuzzy_weight=FUZZY_WEIGHT) -> list[dict]:
    """
    Query the BGEM3 API with cosine similarity and soft fuzzy scoring.

    fuzzy_weight: fraction of score contributed by fuzzy matching (0–1)
    """
    diacritics = False
    if not is_raw_vietnamese(query):
        diacritics = True
    
    print("Query:", query)
    tokenized_query = word_tokenize(query, format="text").lower().split()
    fuzzy_2_gram = ["_".join(tokenized_query)] if len(tokenized_query) == 2 else smart_link(tokenized_query)

    # Embed query
    embedding_bgem3 = embed_bgem3(query)
    data = embedding_bgem3.get("data", None)
    if not data:
        logger.error("Failed to get BGEM3 embeddings.")
        return []

    query_vec = np.array(data[0], dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        query_vec = np.zeros_like(query_vec)
    else:
        query_vec /= norm

    # Select embedding field
    if not diacritics:
        embeddings = np.array([a["raw_embedding"] for a in list_of_text], dtype=np.float32)
    else:
        embeddings = np.array([a["embedding"] for a in list_of_text], dtype=np.float32)
    embeddings = np.nan_to_num(
        embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    # Compute cosine similarities
    sims = embeddings @ query_vec

    if len(tokenized_query) <= 2 and not diacritics:
        # Compute fuzzy scores
        with ThreadPoolExecutor() as ex:
            fuzzy_scores = list(ex.map(score_doc, sparse_docs, fuzzy_2_gram))

        fuzzy_scores = np.array(fuzzy_scores, dtype=np.float32)

        # Combine cosine similarity and fuzzy score
        final_scores = (1 - fuzzy_weight) * sims + fuzzy_weight * fuzzy_scores
        final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        final_scores = sims


    # Top-k selection
    top_idx = np.argpartition(-final_scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-final_scores[top_idx])]
    
    results = [(
        list_of_text[i]["id"], 
        list_of_text[i]["text"], 
        float(final_scores[i]), 
        list_of_text[i]["keywords"], 
        list_of_text[i]["figures"], 
        list_of_text[i]["stats"], 
        list_of_text[i]["updated_at"]
        )
        for i in top_idx
    ]

    print(f"Found {len(results)} stories matching the query.")
    return results



def get_end_values(test_bm25, test_bgem3):
    # Initialize with None or appropriate extremes
    smallest_bm25 = None
    smallest_bgem3 = None
    largest_bm25 = None
    largest_bgem3 = None

    for t in test_bm25:
        val = t[2]
        if val != 0:
            if smallest_bm25 is None or val < smallest_bm25:
                smallest_bm25 = val
        if largest_bm25 is None or val > largest_bm25:
            largest_bm25 = val

    for t in test_bgem3:
        val = t[2]
        if smallest_bgem3 is None or val < smallest_bgem3:
            smallest_bgem3 = val
        if largest_bgem3 is None or val > largest_bgem3:
            largest_bgem3 = val

    # fallback defaults
    smallest_bm25 = smallest_bm25 or 0
    smallest_bgem3 = smallest_bgem3 or 0
    
    return smallest_bm25, smallest_bgem3, largest_bm25, largest_bgem3

def search_filter(query_tokens, test_bm25, test_bgem3, top_k = TOP_K, tau = TIME_TAU):
    # Convert lists to dict for fast lookup: {id: (title, score)}
    bm25_dict = {id_: (title, score, date) for id_, title, score, date in test_bm25}
    bgem3_dict = {id_: (title, score, date) for id_, title, score, date in test_bgem3}

    merged = {}
    smallest_bm25, smallest_bgem3, largest_bm25, largest_bgem3 = get_end_values(test_bm25, test_bgem3)

    EPS = 1e-16
    sparse_weight, dense_weight = compute_weights(query_tokens)
    
    # Merge BM25 first
    for id_, (title, score, date) in bm25_dict.items():
        if id_ in bgem3_dict:
            merged[id_] = (title, sparse_weight * ((score - smallest_bm25) / (largest_bm25 - smallest_bm25 + EPS)) + dense_weight * (bgem3_dict[id_][1] - smallest_bgem3) / (largest_bgem3 - smallest_bgem3 + EPS), date)   # Weighted average
            del bgem3_dict[id_]  # remove so only unique remain
        else:
            merged[id_] = (title, (score - smallest_bm25) / (largest_bm25 - smallest_bm25 + EPS), date)

    # Add remaining dense hits
    for id_, (title, score, date) in bgem3_dict.items():
        merged[id_] = (title, ((bgem3_dict[id_][1] - smallest_bgem3) / (largest_bgem3 - smallest_bgem3 + EPS)), date)

    
    
    now = datetime.now(timezone.utc)
    adjusted = []
    for id_, (title, score, date) in merged.items():
        time = time_score(date, now, tau)
        adjusted_score = score * RELEVANCE_W + time * TIME_W
        adjusted.append((id_, title, adjusted_score, date))

    # Convert back to list and sort
    search_results = heapq.nlargest(top_k, adjusted, key=lambda x: x[2])  # Limit to top k results
    
    # search_results = [(id_, title, score, date) for id_, (title, score, date) in merged.items()]
    # search_results = heapq.nlargest(top_k, search_results, key=lambda x: x[2])  # Limit to top k results
  
    return search_results  

def search_filter_str(query_tokens, test_bm25, test_bgem3, top_k: int = TOP_K, tau: int = TIME_TAU):
    # Convert lists to dict for fast lookup: {id: (title, score)}
    bm25_dict = {id_: (title, score, keywords, figures, stats, date) for id_, title, score, keywords, figures, stats, date in test_bm25}
    bgem3_dict = {id_: (title, score, keywords, figures, stats, date) for id_, title, score, keywords, figures, stats, date in test_bgem3}
    

    merged = {}
    smallest_bm25, smallest_bgem3, largest_bm25, largest_bgem3 = get_end_values(test_bm25, test_bgem3)

    EPS = 1e-16
    sparse_weight, dense_weight = compute_weights(query_tokens)
    
    
    # Merge BM25 first
    for id_, (title, score, keywords, figures, stats, date) in bm25_dict.items():
        if id_ in bgem3_dict:
            merged[id_] = (
                title, 
                sparse_weight * ((score - smallest_bm25) / (largest_bm25 - smallest_bm25 + EPS)) + dense_weight * (bgem3_dict[id_][1] - smallest_bgem3) / (largest_bgem3 - smallest_bgem3 + EPS),
                keywords, 
                figures, 
                stats, 
                date
                )   # Weighted average
            del bgem3_dict[id_]  # remove so only unique remain
        else:
            merged[id_] = (
                title, 
                (score - smallest_bm25) / (largest_bm25 - smallest_bm25 + EPS),
                keywords,
                figures, 
                stats, 
                date
                )

    # Add remaining dense hits
    for id_, (title, score, keywords, figures, stats, date) in bgem3_dict.items():
        merged[id_] = (
            title, 
            ((bgem3_dict[id_][1] - smallest_bgem3) / (largest_bgem3 - smallest_bgem3 + EPS)),
            keywords, 
            figures, 
            stats, 
            date
            )

    now = datetime.now(timezone.utc)
    adjusted = []
    for id_, (title, score, keywords, figures, stats, date) in merged.items():
        time = time_score(date, now, tau)
        adjusted_score = score * RELEVANCE_W + time * TIME_W
        adjusted.append({
            "ID": id_,
            "TITLE": title,
            "SCORE": adjusted_score,
            "KEYWORDS": keywords,
            "KEY_FIGURES": figures,
            "KEY_STATS": stats,
            "DATE": date
        })


    # Convert back to list and sort
    search_results = heapq.nlargest(top_k, adjusted, key=lambda x: x["SCORE"])

  
        
    return search_results  

def director(query_prompt, top_k):
    # Load articles
    with open(ARTICLES, "rb") as f:
        list_of_text = pickle.load(f)
        
    # Load BM25 model
    with open(BM25, "rb") as f:
        sparse_docs = pickle.load(f) 
    
    start_search = time.time()
    
    # query_prompt = restore(query_prompt)
        
    tokenized_query = word_tokenize(query_prompt, format="text").lower().split()
    # search_results = query_bgem3(query_prompt, top_k, list_of_text)
    # search_results, query_tokens = query_bm25(query_prompt, top_k, sparse_docs)
    
    if len(tokenized_query) < QUERY_LEN_MIN:
        test_bm25, query_tokens = query_bm25(query_prompt, top_k, sparse_docs)
        search_results = search_filter(tokenized_query, test_bm25, [], top_k)
    elif len(tokenized_query) >= QUERY_LEN_MAX:
        test_bgem3 = query_bgem3(query_prompt, top_k, list_of_text, sparse_docs)
        search_results = search_filter(tokenized_query, [], test_bgem3, top_k)
    else:
        test_bm25, query_tokens = query_bm25(query_prompt, top_k, sparse_docs)
        test_bgem3 = query_bgem3(query_prompt, top_k, list_of_text, sparse_docs)
        search_results = search_filter(tokenized_query, test_bm25, test_bgem3, top_k)
    
    end_search = time.time()
    runtime = end_search - start_search

      
    if len(tokenized_query) == 2:
        tokenized_query = ["_".join(tokenized_query)] 
    else: 
        tokenized_query = smart_link(tokenized_query)
        
    return search_results, runtime, tokenized_query
 
def director_str(query_prompt, top_k):
    # Load articles
    with open(STORIES, "rb") as f:
        list_of_text = pickle.load(f)
        
    # Load BM25 model
    with open(BM25_STR, "rb") as f:
        sparse_docs = pickle.load(f) 

    
    start_search = time.time()
    
    tokenized_query = word_tokenize(query_prompt, format="text").lower().split()
    
    if len(query_prompt) < QUERY_LEN_MIN:
        test_bm25, query_tokens = query_bm25_str(query_prompt, top_k, sparse_docs)
        search_results = search_filter_str(tokenized_query, test_bm25, [], top_k)
    elif len(query_prompt) >= QUERY_LEN_MAX:
        test_bgem3 = query_bgem3_str(query_prompt, top_k, list_of_text, sparse_docs)
        search_results = search_filter_str(tokenized_query, [], test_bgem3, top_k)
    else:
        test_bm25, query_tokens = query_bm25_str(query_prompt, top_k, sparse_docs)
        test_bgem3 = query_bgem3_str(query_prompt, top_k, list_of_text, sparse_docs)
        search_results = search_filter_str(tokenized_query, test_bm25, test_bgem3, top_k)
    
    end_search = time.time()
    runtime = end_search - start_search
    
    if len(tokenized_query) == 2:
        tokenized_query = ["_".join(tokenized_query)] 
    else: 
        tokenized_query = smart_link(tokenized_query)
    
    
    return search_results, runtime, tokenized_query



def refresh(crawl_time, source_data):
    if crawl_time == datetime.now():
        db = DatabaseService()
        if source_data == "articles":
            get_embedding_materials_art(db, datetime.now())
            embed_articles_bm25()
            embed_non_diacritics_bgem3(ARTICLES)
        elif source_data == "stories":
            get_embedding_materials_str(db, datetime.now())
            embed_stories_bm25()
            embed_non_diacritics_bgem3(STORIES)


# $env:PYTHONPATH = "D:\DH\Internship IceTea\digesty-stories\src"
# python -m pipeline.search.sparse_search

        
# if __name__ == "__main__":
    # Run the embedding process
    # db = DatabaseService()
    # get_embedding_materials_str(db, datetime.now())
    # embed_stories_bm25()
    # embed_non_diacritics_bgem3()
