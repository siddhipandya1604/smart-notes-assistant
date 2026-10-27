from fastapi import APIRouter, Request

from smartnotes.api.schemas import ClassifyRequest, ClassifyResponse

router = APIRouter(tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
    """The assessment contract: {"text": "..."} -> {"category": "...", "uncertain": bool, ...}."""
    prediction = request.app.state.classifier.predict(payload.text)
    return ClassifyResponse(**prediction.__dict__)
