import sys
sys.path.append("D:/DH/Senior/Paperboy") 
import os
import json
import pickle
import redis
import uuid

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def new_session_id():
    sid = str(uuid.uuid4())
    return sid


def load_context(session_id, content):
    context = r.get(session_id)
    if not context:
        r.set(session_id, content)
        r.expire(session_id, 86400)
    
    return context

def update_context(session_id, content):
    context = r.get(session_id)
    new_context = context + "\n" + content
    print(new_context)

    r.set(session_id, new_context)
    r.expire(session_id, 86400)