import dataclasses

import pytest
from fastapi.testclient import TestClient

from smartnotes.api.app import create_app
from smartnotes.services.classifier import NoteClassifier


@pytest.fixture(scope="session")
def classifier() -> NoteClassifier:
    """The real saved model, loaded once for the whole test run."""
    return NoteClassifier.load()


@pytest.fixture()
def client(classifier):
    with TestClient(create_app(classifier=classifier)) as c:
        yield c


def with_threshold(classifier: NoteClassifier, threshold: float) -> NoteClassifier:
    """Same trained model, different uncertainty threshold."""
    return NoteClassifier(dataclasses.replace(classifier._bundle, threshold=threshold))
