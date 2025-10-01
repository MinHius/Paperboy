import asyncio
from fastapi import FastAPI, Query
from typing import List
from pipeline.search.sparse_search import director, director_str
from pipeline.search.config import TOP_K
import uvicorn
import os

from concurrent.futures import ThreadPoolExecutor


# $env:PYTHONPATH="D:\DH\Internship IceTea\digesty-stories\src"
# uvicorn pipeline.search.search_api:app --reload
# & "$env:LOCALAPPDATA\Microsoft\WindowsApps\ngrok.exe" http 8000




app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/search")
async def search(
    query: str = Query(...),
    page: int = Query(1, ge=1),
):
    # Run your search
    results, runtime, token = await asyncio.get_event_loop().run_in_executor(
        None, director_str, query, TOP_K * page
    )
    
    # Pagination logic
    start = TOP_K * (page - 1)
    end = start + 10

    return {
        "query": query,
        "page": page,
        "runtime": f"{runtime}s",
        "tokenized_query": token,
        "results": results[start:end],
    }
