import sys
sys.path.append("D:/DH/Senior/Paperboy") 

import asyncio
from fastapi import FastAPI, Query
from typing import List
from src.search.config_search import TOP_K
import uvicorn
import os

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor


from src.search.search import director_PDB


# $env:PYTHONPATH="D:/DH/Senior/Paperboy"
# uvicorn src.deploy.deploy:app --reload
# & "$env:LOCALAPPDATA\Microsoft\WindowsApps\ngrok.exe" http 8000



app = FastAPI()
templates = Jinja2Templates(directory="src/deploy/templates")

@app.get("/", response_class=HTMLResponse)
async def menu(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

executor = ThreadPoolExecutor(max_workers=4)


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str | None = Query(None),  # ✅ optional
    page: int = Query(1, ge=1),
):
    if not query:
        # no query yet → just show the search page
        return templates.TemplateResponse(
            "search.html",
            {"request": request, "query": "", "results": []},
        )

    # run actual search
    results = await asyncio.get_event_loop().run_in_executor(
        None, director_PDB, query
    )
    
    print(len(results))

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": query,
            "page": page,
            "results": results,
        },
    )


