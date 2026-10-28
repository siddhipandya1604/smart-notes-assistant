"""FastAPI application factory.

    uvicorn smartnotes.api.app:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from smartnotes.config import FRONTEND_DIST

from smartnotes.api.errors import install_error_handlers
from smartnotes.api.routes import classify, health, notes, search
from smartnotes.search.service import NoteSearch
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.notes import NoteStore


def create_app(classifier: NoteClassifier | None = None, store: NoteStore | None = None) -> FastAPI:
    """Build the app. Tests pass their own classifier/store; production loads the saved model once
    at startup and opens the SQLite database (data/notes.db, or $SMARTNOTES_DB)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or NoteClassifier.load()
        app.state.store = store or NoteStore()
        app.state.search = NoteSearch(app.state.classifier)
        yield

    app = FastAPI(title="Smart Notes Assistant", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(classify.router)
    app.include_router(health.router)
    app.include_router(notes.router)
    app.include_router(search.router)
    if FRONTEND_DIST.is_dir():  # last, so the API routes above win
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    return app


app = create_app()
