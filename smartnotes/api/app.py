"""FastAPI application factory.

    uvicorn smartnotes.api.app:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smartnotes.api.errors import install_error_handlers
from smartnotes.api.routes import classify, health
from smartnotes.services.classifier import NoteClassifier


def create_app(classifier: NoteClassifier | None = None) -> FastAPI:
    """Build the app. Tests pass their own classifier; production loads the saved model once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or NoteClassifier.load()
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
    return app


app = create_app()
