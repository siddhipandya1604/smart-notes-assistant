from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    classifier = request.app.state.classifier
    return {
        "status": "ok",
        "model": classifier.model_name,
        "labels": classifier.labels,
        "uncertainty_threshold": round(classifier.threshold, 4),
    }
