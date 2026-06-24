"""Apertura gateway service.

Thin FastAPI proxy that routes:
  POST /ingest              → ingestion-service:8001/ingest
  POST /query               → query-service:8002/query
  GET  /suggest-questions   → query-service:8002/suggest-questions
  GET  /pages/*             → query-service:8002/pages/*
  GET  /health              → local

In production, replace with an nginx ingress or AWS ALB.
In local Docker Compose / kind, this runs on port 8000 and is the
only service the frontend talks to.
"""

import os
import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

INGESTION_URL = os.getenv("INGESTION_SERVICE_URL", "http://ingestion-service:8001")
QUERY_URL     = os.getenv("QUERY_SERVICE_URL",     "http://query-service:8002")

app = FastAPI(title="Apertura Gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            f"{INGESTION_URL}/ingest",
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


class QueryRequest(BaseModel):
    question: str
    doc_id: str = "default"


@app.post("/query")
async def query(req: QueryRequest):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{QUERY_URL}/query", json=req.model_dump())
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/suggest-questions/{doc_id}")
async def suggest_questions(doc_id: str):
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(f"{QUERY_URL}/suggest-questions/{doc_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/pages/{path:path}")
async def proxy_pages(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{QUERY_URL}/pages/{path}")
    return StreamingResponse(
        content=iter([resp.content]),
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )
