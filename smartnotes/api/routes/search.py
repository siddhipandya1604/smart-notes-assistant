from fastapi import APIRouter, HTTPException, Query, Request

from smartnotes.api.schemas import SearchResponse

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(request: Request, q: str = Query(max_length=200), limit: int = Query(10, ge=1, le=50)):
    """Ranked saved notes for a keyword or short question. Returns an empty list (with a message)
    rather than unrelated notes when nothing matches well."""
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Type something to search for.")

    notes = request.app.state.store.list()
    if not notes:
        return SearchResponse(query=query, results=[], message="You have no saved notes yet.")

    by_id = {n.id: n for n in notes}
    hits = request.app.state.search.search(query, notes, limit)
    results = [{"note": by_id[h.doc_id].to_dict(), "score": round(h.score, 4), "match": h.match} for h in hits]
    message = None if results else f"No notes match “{query}” well."
    return SearchResponse(query=query, results=results, message=message)
