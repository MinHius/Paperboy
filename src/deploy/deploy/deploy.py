import sys
sys.path.append("D:/DH/Senior/Paperboy/src") 

import asyncio
from src.search.config import TOP_K
import os

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query

from search.search import director_PDB
from api.chat.chat import generate_chat
from database.parade.database import load_stories, load_stories_display
from api.chat.utils import load_context, new_session_id, update_context
from api.chat.utils import json_to_paragraph

from search.config import (
    TOP_K
)

path="D:/DH/Senior/Paperboy/src/api/chat/chat_context.pkl"

# $env:PYTHONPATH="D:/DH/Senior/Paperboy/src"
# uvicorn deploy.deploy.deploy:app --reload
# & "$env:LOCALAPPDATA\Microsoft\WindowsApps\ngrok.exe" http 8000

app = FastAPI()
templates = Jinja2Templates(directory="src/deploy/templates")
app.mount("/static", StaticFiles(directory="src/deploy/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    stories = load_stories_display()
    stories = [story for story in stories if len(json_to_paragraph(story["story"])) >= 100]
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "stories": stories, "year": date.today().year}
    )


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
            {"request": request, "results": []},
        )

    # run actual search
    results = await asyncio.get_event_loop().run_in_executor(
        None, director_PDB, query, TOP_K
    )
    
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": query,
            "results": results
        },
    )
    
    
@app.get("/story/{id:path}", response_class=HTMLResponse)
async def story_page(request: Request, id: str):
    try:
        os.remove(path)
    except:
        pass
    stories = load_stories()
    story = next((s for s in stories if s["id"] == id), None)
    story_chat_id = new_session_id()
    if story:
        story['chat_id'] = story_chat_id
    
    if not story:
        return templates.TemplateResponse(
            "404.html", {"request": request}, status_code=404
        )

    return templates.TemplateResponse(
        "story.html",
        {"request": request, "story": story, "year": date.today().year},
    )


@app.post("/story/{id:path}/chat")
async def chat_endpoint(id: str, request: Request):
    payload = await request.json()
    message = payload.get("message")
    session_id = payload.get("chat_id")

    context = load_context(session_id, json_to_paragraph(payload.get("context"))) 
    reply = generate_chat(message, context)
    update_context(session_id, reply['context'])

    return {"reply": reply["response"]}