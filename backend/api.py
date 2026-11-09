"""FastAPI HTTP API for the DUM-E RAG engine."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import (
    AuthUser,
    authenticate_user,
    create_session,
    create_user,
    get_user_for_token,
    initialize_auth_database,
    validate_credentials,
)
from dume.config import Settings, get_settings
from dume.engine import DumeEngine
from dume.ingestion.indexer import index_documents
from dume.logging_setup import setup_logging
from dume.models import Answer, RetrievedChunk

logger = logging.getLogger(__name__)

settings = get_settings()
settings.ensure_directories()
setup_logging(level="INFO", log_file=settings.LOG_DIR / "dume.log")
engine = DumeEngine(settings=settings)
AUTH_DATABASE = settings.DATA_DIR / "users.sqlite"
initialize_auth_database(AUTH_DATABASE)
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(title="DUM-E API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://dum-e-dume5.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request payload for an engine chat turn."""

    session_id: str = Field(min_length=1)
    subject: str | None = Field(default=None, min_length=1)
    question: str = Field(min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True)


class LoginRequest(BaseModel):
    """Credentials submitted for login."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class RegisterRequest(LoginRequest):
    """Credentials and display name submitted for registration."""

    name: str = Field(min_length=1, max_length=120)


class AuthResponse(BaseModel):
    """Successful authentication response."""

    authenticated: bool = True
    token: str
    session_id: str
    email: str
    name: str


class ResetRequest(BaseModel):
    """Request payload for resetting a chat session."""

    session_id: str = Field(min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True)


class SourcePayload(BaseModel):
    """Serializable retrieval source information."""

    chunk_id: str
    source: str
    score: float
    rank: int
    metadata: dict[str, object]


class ChatResponse(BaseModel):
    """API representation of an engine answer."""

    text: str
    sources: list[SourcePayload]
    from_cache: bool
    timings_ms: float


class UploadResponse(BaseModel):
    """Result of saving and indexing an uploaded document."""

    filename: str
    subject: str
    kb_version: int


class ResetResponse(BaseModel):
    """Successful session reset response."""

    session_id: str
    reset: bool = True


class SubjectsResponse(BaseModel):
    """Available document subject folders."""

    subjects: list[str]


class DocumentPayload(BaseModel):
    """Document metadata shown in a subject workspace."""

    filename: str
    size_bytes: int
    uploaded_at: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """Documents belonging to a subject."""

    subject: str
    documents: list[DocumentPayload]


class ReindexResponse(BaseModel):
    """Result of a subject index synchronization."""

    subject: str
    kb_version: int
    chunks_indexed: int


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
) -> AuthUser:
    """Require a valid bearer session for protected API routes."""
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication is required.")
    try:
        user = get_user_for_token(AUTH_DATABASE, credentials.credentials)
    except (OSError, ValueError) as exc:
        logger.exception("Could not validate authentication session")
        raise HTTPException(status_code=500, detail="Unable to validate session.") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="Your session is invalid or expired.")
    return user


@app.post("/api/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest) -> AuthResponse:
    """Create a user account and start an authenticated session."""
    try:
        validate_credentials(request.email, request.password)
        user = create_user(
            AUTH_DATABASE, request.name, request.email, request.password
        )
        token = create_session(AUTH_DATABASE, user)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Could not register user")
        raise HTTPException(status_code=500, detail="Unable to create account.") from exc
    return AuthResponse(
        token=token, session_id=token, email=user.email, name=user.name
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login(request: LoginRequest) -> AuthResponse:
    """Validate credentials and start an authenticated session."""
    user = authenticate_user(AUTH_DATABASE, request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_session(AUTH_DATABASE, user)
    return AuthResponse(
        token=token, session_id=token, email=user.email, name=user.name
    )


def _safe_pdf_name(filename: str | None) -> str:
    """Validate and normalize an uploaded PDF filename."""
    if not filename:
        raise HTTPException(status_code=400, detail="A PDF filename is required.")
    safe_name = Path(filename).name
    if not safe_name or Path(safe_name).suffix.casefold() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    return safe_name


def _safe_subject(subject: str | None) -> str:
    """Validate a single subject folder name."""
    if not subject or not subject.strip():
        raise HTTPException(status_code=422, detail="A subject is required.")
    normalized = subject.strip()
    if normalized in {".", ".."} or Path(normalized).name != normalized:
        raise HTTPException(status_code=422, detail="Subject must be a folder name.")
    return normalized


def _user_raw_dir(user_id: int) -> Path:
    """Return the private raw-document root for a user."""
    return settings.RAW_DIR / str(user_id)


def _user_kb_version_path(user_id: int) -> Path:
    """Return the private indexing manifest path for a user."""
    return settings.DATA_DIR / "users" / str(user_id) / "kb_version.json"


def _user_collection_name(user_id: int) -> str:
    """Return the private Chroma collection name for a user."""
    return f"dume_user_{user_id}"


def _subject_path(user_id: int, subject: str, filename: str) -> Path:
    """Resolve a validated document path within a subject folder."""
    safe_subject = _safe_subject(subject)
    safe_filename = _safe_pdf_name(filename)
    return _user_raw_dir(user_id) / safe_subject / safe_filename


def _sync_index(user_id: int):
    """Synchronize only one user's private index after a file mutation."""
    return index_documents(
        raw_dir=_user_raw_dir(user_id),
        kb_version_path=_user_kb_version_path(user_id),
        collection_name=_user_collection_name(user_id),
        settings=settings,
        user_id=user_id,
    )


async def _save_upload(
    upload: UploadFile,
    subject: str,
    user_id: int,
    active_settings: Settings,
) -> Path:
    """Save a non-empty PDF upload below the configured raw directory."""
    filename = _safe_pdf_name(upload.filename)
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    raw_dir = active_settings.RAW_DIR / str(user_id) / subject
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    digest_path = destination.with_suffix(f"{destination.suffix}.sha256")
    digest = hashlib.sha256(content).hexdigest()

    try:
        if (
            not destination.exists()
            or not digest_path.exists()
            or digest_path.read_text(encoding="ascii") != digest
        ):
            destination.write_bytes(content)
            digest_path.write_text(digest, encoding="ascii")
    except OSError as exc:
        logger.exception("Could not save uploaded PDF %s", filename)
        raise HTTPException(
            status_code=500, detail="Unable to save the uploaded PDF."
        ) from exc
    return destination


def _source_payload(source: RetrievedChunk) -> SourcePayload:
    """Convert a retrieved dataclass into a JSON-safe response model."""
    return SourcePayload(
        chunk_id=source.chunk_id,
        source=source.source,
        score=source.score,
        rank=source.rank,
        metadata=source.metadata,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to index")],
    subject: Annotated[str, Form(description="Subject folder name")] = "general",
    user: Annotated[AuthUser, Depends(get_current_user)] = None,
) -> UploadResponse:
    """Save a PDF and synchronize the ChromaDB index."""
    safe_subject = _safe_subject(subject)
    path = await _save_upload(file, safe_subject, user.id, settings)
    try:
        result = _sync_index(user.id)
    except Exception as exc:
        logger.exception("Indexing failed for uploaded PDF %s", path.name)
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {exc}",
        ) from exc
    finally:
        await file.close()

    return UploadResponse(
        filename=path.name,
        subject=safe_subject,
        kb_version=result.kb_version,
    )


@app.get("/api/subjects/{subject}/files", response_model=DocumentsResponse)
def list_subject_files(
    subject: str,     user: Annotated[AuthUser, Depends(get_current_user)]
) -> DocumentsResponse:
    """List PDFs and indexed chunk counts for a subject."""
    safe_subject = _safe_subject(subject)
    subject_dir = _user_raw_dir(user.id) / safe_subject
    manifest_path = _user_kb_version_path(user.id)
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Could not read indexing manifest: %s", exc)

    files = manifest.get("files", {})
    documents: list[DocumentPayload] = []
    if subject_dir.exists():
        for path in sorted(subject_dir.glob("*.pdf")):
            relative_name = path.relative_to(_user_raw_dir(user.id)).as_posix()
            entry = files.get(relative_name, {})
            try:
                stat = path.stat()
            except OSError as exc:
                logger.warning("Could not stat document %s: %s", path, exc)
                continue
            documents.append(
                DocumentPayload(
                    filename=path.name,
                    size_bytes=stat.st_size,
                    uploaded_at=time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)
                    ),
                    chunk_count=int(entry.get("chunk_count", 0)),
                )
            )
    return DocumentsResponse(subject=safe_subject, documents=documents)


@app.delete("/api/subjects/{subject}/files/{filename}", response_model=ReindexResponse)
def delete_subject_file(
    subject: str,
    filename: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> ReindexResponse:
    """Delete a PDF and synchronize ChromaDB so its chunks are purged."""
    path = _subject_path(user.id, subject, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        path.unlink()
        sidecar = path.with_suffix(f"{path.suffix}.sha256")
        if sidecar.exists():
            sidecar.unlink()
        result = _sync_index(user.id)
    except OSError as exc:
        logger.exception("Could not delete document %s", path)
        raise HTTPException(status_code=500, detail="Unable to delete document.") from exc
    return ReindexResponse(
        subject=_safe_subject(subject),
        kb_version=result.kb_version,
        chunks_indexed=result.chunks_indexed,
    )


@app.post("/api/subjects/{subject}/files/{filename}/reindex", response_model=ReindexResponse)
def reindex_subject_file(
    subject: str,
    filename: str,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> ReindexResponse:
    """Re-run incremental indexing for one subject document."""
    path = _subject_path(user.id, subject, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        result = _sync_index(user.id)
    except Exception as exc:
        logger.exception("Could not re-index document %s", path)
        raise HTTPException(status_code=500, detail="Unable to re-index document.") from exc
    return ReindexResponse(
        subject=_safe_subject(subject),
        kb_version=result.kb_version,
        chunks_indexed=result.chunks_indexed,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, user: Annotated[AuthUser, Depends(get_current_user)]
) -> ChatResponse:
    """Answer a question for the requested conversation session."""
    started = time.perf_counter()
    try:
        answer: Answer = engine.ask(
            request.session_id,
            request.question,
            subject=request.subject.strip() if request.subject else None,
            user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed for session %s", request.session_id)
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from exc

    return ChatResponse(
        text=answer.text,
        sources=[_source_payload(source) for source in answer.sources],
        from_cache=answer.cached,
        timings_ms=(time.perf_counter() - started) * 1000,
    )


@app.post("/api/reset", response_model=ResetResponse)
def reset(
    request: ResetRequest, user: Annotated[AuthUser, Depends(get_current_user)]
) -> ResetResponse:
    """Clear all memory associated with a conversation session."""
    try:
        engine.reset_session(request.session_id, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reset failed for session %s", request.session_id)
        raise HTTPException(status_code=500, detail="Unable to reset the session.") from exc
    return ResetResponse(session_id=request.session_id)


@app.get("/api/subjects", response_model=SubjectsResponse)
def subjects(user: Annotated[AuthUser, Depends(get_current_user)]) -> SubjectsResponse:
    """Return subject folders containing documents."""
    try:
        user_raw_dir = _user_raw_dir(user.id)
        if not user_raw_dir.exists():
            return SubjectsResponse(subjects=[])
        names = sorted(
            path.name
            for path in user_raw_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
    except OSError as exc:
        logger.exception("Could not scan subject folders")
        raise HTTPException(
            status_code=500, detail="Unable to list document subjects."
        ) from exc
    return SubjectsResponse(subjects=names)
