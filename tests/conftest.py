import dataclasses

import pytest
from fastapi.testclient import TestClient

from smartnotes.api.app import create_app
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.notes import NoteStore


@pytest.fixture(scope="session")
def classifier() -> NoteClassifier:
    """The real saved model, loaded once for the whole test run."""
    return NoteClassifier.load()


@pytest.fixture()
def db_path(tmp_path):
    """A throwaway database file, so tests never touch data/notes.db."""
    return tmp_path / "notes.db"


@pytest.fixture()
def store(db_path) -> NoteStore:
    return NoteStore(db_path)


@pytest.fixture()
def client(classifier, store):
    with TestClient(create_app(classifier=classifier, store=store)) as c:
        yield c


def with_threshold(classifier: NoteClassifier, threshold: float) -> NoteClassifier:
    """Same trained model, different uncertainty threshold."""
    return NoteClassifier(dataclasses.replace(classifier._bundle, threshold=threshold))
