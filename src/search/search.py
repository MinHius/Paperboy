import sys
sys.path.append("D:/DH/Senior/Paperboy") 

import json
import heapq
import logging
import psycopg2
import numpy as np
import unicodedata

from datetime import datetime, timedelta, timezone

from underthesea import word_tokenize
from src.api.BGEM3 import embed_bgem3
from src.databases.database import get_connection
from src.search.config_search import (
    QUERY_LEN_MAX, 
    QUERY_LEN_MIN, 
    TOP_K,
    TIME_W,
    INDEX_NAME,
    MAX_BM25_WEIGHT,
    MIN_BM25_WEIGHT
)

from src.search.command import (
    SEARCH_BM25,
    SEARCH_BGEM3,
    SEARCH_HYBRID,
)



def compute_weights(input_tokens: list, 
                    sparse_max: float = MAX_BM25_WEIGHT, sparse_min: float = MIN_BM25_WEIGHT, 
                    len_min: int = QUERY_LEN_MIN, len_max: int = QUERY_LEN_MAX) -> int:
    """
    Compute continuous sparse/dense weights based on query length.

    Args:
        query_tokens: list of tokens in the query
        sparse_max: max weight for sparse (shortest queries)
        sparse_min: min weight for sparse (longest queries)
        len_min: query length to start decreasing sparse weight
        len_max: query length where sparse weight hits sparse_min
    """
    
    query_length = len(input_tokens)
    
    # Linear interpolation
    slope = (sparse_min - sparse_max) / (len_max - len_min)
    sparse_weight = sparse_max + slope * (query_length - len_min)
    dense_weight = 1 - sparse_weight
    
    print(f"Computed weights for hybrid scoring.")
    
    return sparse_weight, dense_weight


def is_raw_vietnamese(text: str) -> bool:
    """
    Returns True if the text is 'raw' Vietnamese (no diacritics),
    False if it has Vietnamese diacritic vowels or tone marks.
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
   

def search_bm25(search_content: str) -> list:
    """ Search for stories using lexical search."""
    
    sql = SEARCH_BM25
    
    print(f"Searching for {search_content} lexically...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (search_content, TOP_K))
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()

            results = [dict(zip(columns, row)) for row in rows]
    
    return results


def search_bgem3(search_content):
    """Search for stories using semantic search."""
        
    embedding_bgem3 = embed_bgem3(search_content)
    data = embedding_bgem3.get("data")

    if data and len(data) > 0:
        embedded_search_content = data[0]
    else:
        embedded_search_content = [0.0] * 1024

    # Two commands for 2 different cases
    sql = SEARCH_BGEM3
    
    print(f"Searching for {search_content} semantically...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (embedded_search_content, TOP_K))
            results = cur.fetchall()
    
    return results


def search_hybrid(search_content):
    """Search stories using hybrid search."""
    
    # Embed query for semantic search
    embedding_bgem3 = embed_bgem3(search_content)
    data = embedding_bgem3.get("data", None)
    query_tokens = search_content.lower().split()
    sparse_weight, dense_weight = compute_weights(query_tokens)
    
    sql = SEARCH_HYBRID

    print(f"Searching for {search_content} semantically and lexically...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (search_content, 2 * TOP_K, data[0], data[0], 2 * TOP_K, 1 - TIME_W, dense_weight, sparse_weight, TIME_W, TOP_K))
            results = cur.fetchall()
    
    return results



def director_PDB(search_content):
    """Search orchestrator."""

    # Tokenize to check for query length.
    tokenized_query = word_tokenize(search_content, format="text").lower().split()
    print("Searching for stories related to: %s", search_content)
    
    # Conditional search
    if len(tokenized_query) <= QUERY_LEN_MIN:
        results = search_bm25(search_content)
    elif len(tokenized_query) > QUERY_LEN_MAX:
        results = search_bgem3(search_content)
    else:
        results = search_hybrid(search_content)
    
    print(f"Search completed!")
    
    return results











