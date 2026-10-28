from fastapi import APIRouter, HTTPException, Request, Response

from smartnotes.api.schemas import Category, LoadStarterResponse, NoteOut, NoteText
from smartnotes.store.starter import load_starter_notes

router = APIRouter(prefix="/api/notes", tags=["notes"])


@router.get("", response_model=list[NoteOut])
def list_notes(request: Request, category: Category | None = None, uncertain: bool | None = None,
               label: str | None = None):
    """All saved notes in ID order. `category` matches the four-label category (uncertain notes
    appear under their best guess), `label` the label shown in the app (e.g. Health), and
    `uncertain=true` lists only the flagged ones."""
    notes = request.app.state.store.list(category=category, uncertain=uncertain, label=label)
    return [n.to_dict() for n in notes]


@router.post("", response_model=NoteOut, status_code=201)
def add_note(payload: NoteText, request: Request):
    """Classify the note with the trained model, then save it under a new ID."""
    prediction = request.app.state.classifier.predict(payload.text)
    return request.app.state.store.add(payload.text, prediction).to_dict()


@router.post("/load-starter", response_model=LoadStarterResponse)
def load_starter(request: Request):
    """Load N001-N012 from data/starter_notes.json. Idempotent: existing notes are skipped."""
    result = load_starter_notes(request.app.state.store, request.app.state.classifier)
    return {"added": result.added, "skipped": result.skipped, "total": result.total}


@router.get("/{note_id}", response_model=NoteOut)
def get_note(note_id: str, request: Request):
    note = request.app.state.store.get(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"No note with ID {note_id}.")
    return note.to_dict()


@router.delete("/{note_id}", status_code=204)
def delete_note(note_id: str, request: Request) -> Response:
    if not request.app.state.store.delete(note_id):
        raise HTTPException(status_code=404, detail=f"No note with ID {note_id}.")
    return Response(status_code=204)
