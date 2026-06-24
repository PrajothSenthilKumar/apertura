"""Mount static file serving for page images.
Import and call mount_static(app) from main.py.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def mount_static(app: FastAPI) -> None:
    pages_dir = Path("data/pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pages", StaticFiles(directory=str(pages_dir)), name="pages")
