"""Turns validation failures into one friendly, predictable JSON shape."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from smartnotes.config import MAX_TEXT_CHARS


def _friendly(error: dict) -> str:
    kind = error.get("type", "")
    field = _field(error)  # "text" for note bodies, "q" for search, "body" for malformed JSON
    if kind == "missing":
        return f"The '{field}' field is required."
    if kind == "string_too_long":
        limit = (error.get("ctx") or {}).get("max_length", MAX_TEXT_CHARS)  # the limit that was actually broken
        subject = "A search query" if field == "q" else "A note"
        return f"{subject} can be at most {limit} characters."
    if kind in ("string_type", "string_strict"):
        return f"'{field}' must be a string."
    if kind == "json_invalid":
        return "The request body is not valid JSON."
    message = str(error.get("msg", "Invalid request."))
    return message.removeprefix("Value error, ")


def _field(error: dict) -> str:
    """Name of the offending field. For malformed JSON, `loc` ends in a character offset, not a field."""
    if error.get("type") == "json_invalid":
        return "body"
    return ".".join(str(part) for part in error["loc"][1:]) or "body"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": _field(e), "message": _friendly(e)} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"message": errors[0]["message"], "errors": errors})
