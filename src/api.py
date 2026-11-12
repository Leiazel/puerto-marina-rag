"""
FastAPI REST API server for Club Atlético Puerto Marina RAG System (v2.2).
"""
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import BASE_DIR, RAW_DOCS_DIR, DB_PATH, CORS_ALLOW_ORIGINS
from src.database import get_db_connection, init_database
from src.rag_pipeline import ClubRAGPipeline
from src.ingestion import (
    ingest_document_file,
    run_full_ingestion,
    delete_document_by_id,
    seed_metadata,
)
from src.embeddings import get_embedding_engine

_DOC_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{3,80}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_reindex_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db_path = pipeline.db_path
    init_database(db_path)
    conn = get_db_connection(db_path)
    try:
        dept_count = conn.execute("SELECT count(*) FROM departments").fetchone()[0]
        if dept_count == 0:
            seed_metadata(conn)
    finally:
        conn.close()
    yield


app = FastAPI(
    title="Club Atlético Puerto Marina - RAG Knowledge Engine v2.2",
    description="Hybrid RAG system (SQLite FTS5 BM25 + FastEmbed ONNX + RRF) for football club operations.",
    version="2.2.0",
    lifespan=lifespan
)

allow_all_origins = "*" in CORS_ALLOW_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClubRAGPipeline(db_path=DB_PATH)
STATIC_DIR = BASE_DIR / "web" / "static"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Pregunta en lenguaje natural sobre el club o día a día")
    department_filter: Optional[str] = Field(None, description="Filtro opcional por departamento")
    top_k: int = Field(4, ge=1, le=15, description="Cantidad de fragmentos a recuperar")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Balance Híbrido: 1.0 = BM25, 0.0 = Vector, 0.5 = RRF")


class IngestDocRequest(BaseModel):
    doc_code: str = Field(..., min_length=3, max_length=80, description="Código único del documento (ej: MED-2026-11-08-01)")
    title: str = Field(..., min_length=1, max_length=250, description="Título oficial del documento")
    department_id: str = Field(..., min_length=1, max_length=64, description="ID del departamento responsable")
    author: str = Field("Staff Técnico", min_length=1, max_length=120, description="Autor o profesional emisor")
    date: str = Field("2026-11-08", description="Fecha en formato YYYY-MM-DD")
    tags: List[str] = Field(default=[], description="Etiquetas clave")
    content: str = Field(..., min_length=1, max_length=100000, description="Contenido en formato Markdown con secciones")


def _validate_department_exists(conn, department_id: str) -> None:
    row = conn.execute("SELECT id FROM departments WHERE id = ?", (department_id,)).fetchone()
    if not row:
        raise HTTPException(
            status_code=400,
            detail=f"Departamento '{department_id}' no válido."
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/status")
def get_status():
    stats = pipeline.get_stats()
    return {
        "status": "online",
        "club": "Club Atlético Puerto Marina",
        "stats": stats
    }


@app.get("/api/llm/health")
def get_llm_health():
    """Checks connectivity and available models on the configured LLM provider."""
    return pipeline.generator.check_llm_health()


@app.post("/api/query")
def run_query(req: QueryRequest):
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    dept_filter = req.department_filter.strip() if req.department_filter else None
    if dept_filter:
        conn = get_db_connection(pipeline.db_path)
        try:
            _validate_department_exists(conn, dept_filter)
        finally:
            conn.close()

    return pipeline.ask(
        query=clean_query,
        department_filter=dept_filter,
        top_k=req.top_k,
        alpha=req.alpha
    )


@app.post("/api/reindex")
def reindex_knowledge_base():
    """Re-runs full ingestion of Markdown files and SQL relational tables."""
    if not _reindex_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Ya hay una reindexación en progreso.")
    try:
        res = run_full_ingestion(db_path=pipeline.db_path, docs_dir=RAW_DOCS_DIR)
        return {"status": "reindexed", "details": res}
    finally:
        _reindex_lock.release()


@app.get("/api/documents")
def list_documents(department: Optional[str] = None):
    conn = get_db_connection(pipeline.db_path)
    try:
        cur = conn.cursor()
        if department:
            cur.execute("""
                SELECT d.*, dept.name as department_name, count(c.id) as chunk_count
                FROM documents d
                JOIN departments dept ON dept.id = d.department_id
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.department_id = ?
                GROUP BY d.id
                ORDER BY d.date DESC, d.id DESC;
            """, (department,))
        else:
            cur.execute("""
                SELECT d.*, dept.name as department_name, count(c.id) as chunk_count
                FROM documents d
                JOIN departments dept ON dept.id = d.department_id
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.date DESC, d.id DESC;
            """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    conn = get_db_connection(pipeline.db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        doc = cur.fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Documento no encontrado.")

        cur.execute(
            "SELECT id, chunk_index, title, content, metadata_json FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (doc_id,)
        )
        chunks = cur.fetchall()
        return {"document": dict(doc), "chunks": [dict(c) for c in chunks]}
    finally:
        conn.close()


@app.post("/api/documents")
def add_new_document(req: IngestDocRequest):
    doc_code = req.doc_code.strip()
    if not _DOC_CODE_RE.match(doc_code):
        raise HTTPException(
            status_code=400,
            detail="doc_code inválido: solo se permiten letras, números, '-' y '_' (3 a 80 caracteres)."
        )
    if not _DATE_RE.match(req.date.strip()):
        raise HTTPException(status_code=400, detail="Fecha inválida: use formato YYYY-MM-DD.")

    department_id = req.department_id.strip()
    title = req.title.strip()
    author = req.author.strip()
    content = req.content.strip()
    if not title or not author or not content:
        raise HTTPException(status_code=400, detail="Título, autor y contenido no pueden estar vacíos.")

    conn = get_db_connection(pipeline.db_path)
    try:
        _validate_department_exists(conn, department_id)

        raw_docs_root = RAW_DOCS_DIR.resolve()
        dept_folder = (raw_docs_root / department_id).resolve()
        if not dept_folder.is_relative_to(raw_docs_root):
            raise HTTPException(status_code=400, detail="Ruta de departamento inválida.")
        dept_folder.mkdir(parents=True, exist_ok=True)

        target_file = (dept_folder / f"{doc_code}.md").resolve()
        if not target_file.is_relative_to(dept_folder):
            raise HTTPException(status_code=400, detail="Ruta de archivo inválida.")

        # Clean previous file if same doc_code existed under another department folder
        prev = conn.execute("SELECT file_path FROM documents WHERE doc_code = ?", (doc_code,)).fetchone()
        if prev and prev["file_path"] and not str(prev["file_path"]).startswith("sqlite://"):
            prev_path = Path(prev["file_path"])
            if not prev_path.is_absolute():
                prev_path = (BASE_DIR / prev_path).resolve()
            else:
                prev_path = prev_path.resolve()
            if prev_path != target_file and prev_path.is_relative_to(raw_docs_root) and prev_path.exists():
                prev_path.unlink()

        clean_tags = [t.strip() for t in req.tags if t and t.strip()]
        frontmatter = {
            "id": doc_code,
            "title": title,
            "department": department_id,
            "author": author,
            "date": req.date.strip(),
            "tags": clean_tags,
        }
        fm_yaml = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
        raw_md = f"---\n{fm_yaml}\n---\n\n{content}\n"
        target_file.write_text(raw_md, encoding="utf-8")

        embedder = get_embedding_engine()
        doc_id, chunk_count = ingest_document_file(target_file, conn, embedder)

        return {
            "status": "success",
            "document_id": doc_id,
            "doc_code": doc_code,
            "chunks_created": chunk_count,
            "message": f"Documento '{title}' indexado correctamente en la base de datos."
        }
    finally:
        conn.close()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, delete_file: bool = Query(True)):
    conn = get_db_connection(pipeline.db_path)
    try:
        doc = conn.execute("SELECT id, doc_code, title, file_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Documento no encontrado.")
        if str(doc["file_path"] or "").startswith("sqlite://"):
            raise HTTPException(
                status_code=400,
                detail="No se pueden eliminar documentos virtuales sincronizados desde tablas SQL."
            )

        deleted = delete_document_by_id(doc_id, conn)
        if delete_file and deleted and deleted.get("file_path"):
            raw_docs_root = RAW_DOCS_DIR.resolve()
            fpath = Path(deleted["file_path"])
            if not fpath.is_absolute():
                fpath = (BASE_DIR / fpath).resolve()
            else:
                fpath = fpath.resolve()
            if fpath.is_relative_to(raw_docs_root) and fpath.exists():
                fpath.unlink()

        return {
            "status": "deleted",
            "document_id": doc_id,
            "doc_code": deleted["doc_code"] if deleted else None,
            "message": f"Documento '{deleted['title']}' eliminado del índice."
        }
    finally:
        conn.close()


@app.get("/api/players")
def list_players():
    conn = get_db_connection(pipeline.db_path)
    try:
        rows = conn.execute("SELECT * FROM players ORDER BY dorsal ASC;").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/staff")
def list_staff():
    conn = get_db_connection(pipeline.db_path)
    try:
        rows = conn.execute("""
            SELECT s.*, d.name as department_name 
            FROM staff s
            JOIN departments d ON d.id = s.department_id
            ORDER BY s.id ASC;
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/departments")
def list_departments():
    conn = get_db_connection(pipeline.db_path)
    try:
        rows = conn.execute("SELECT * FROM departments ORDER BY id ASC;").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/logs")
def list_logs(limit: int = Query(15, ge=1, le=200)):
    conn = get_db_connection(pipeline.db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM query_logs ORDER BY created_at DESC, id DESC LIMIT ?;",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Puerto Marina RAG Engine v2.2 is running."}

