"""
FastAPI REST API server for Club Atlético Puerto Marina RAG System (v3.0).
Features:
- Hybrid RAG queries (SQLite FTS5 BM25 + FastEmbed Multilingual ONNX + RRF).
- Hardened Path Traversal protection on file writes and department validation.
- Multi-format document upload (PDF, DOCX, DOC, MD, TXT) with SHA-256 deduplication and magic byte validation.
- Cascading document deletion and real-time in-memory vector cache invalidation.
- Telemetry, relational SQL entity inspection, and LLM diagnostics.
"""
import re
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import (
    BASE_DIR,
    DB_PATH,
    UPLOADS_DIR,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES
)
from src.database import get_db_connection
from src.rag_pipeline import ClubRAGPipeline
from src.ingestion import (
    ingest_document_file,
    ingest_uploaded_file,
    run_full_ingestion,
    INITIAL_DEPARTMENTS
)
from src.embeddings import get_embedding_engine
from src.extractors import ExtractionError

app = FastAPI(
    title="Club Atlético Puerto Marina - RAG Knowledge Engine v3.0",
    description="Hybrid RAG system (SQLite FTS5 BM25 + FastEmbed ONNX + RRF) with multi-format ingestion for football club operations.",
    version="3.0.0"
)

# Hardened CORS policy: avoid allow_credentials=True with wildcard allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClubRAGPipeline(db_path=DB_PATH)
STATIC_DIR = BASE_DIR / "web" / "static"
VALID_DEPT_IDS = {d["id"] for d in INITIAL_DEPARTMENTS}


def validate_department_id(dept_id: str) -> str:
    """Ensures department ID is whitelisted and prevents directory traversal."""
    clean = dept_id.strip().lower()
    if clean not in VALID_DEPT_IDS:
        conn = get_db_connection(DB_PATH)
        row = conn.execute("SELECT id FROM departments WHERE id = ?", (clean,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=400, detail=f"Departamento inválido: '{dept_id}'.")
    return clean


def sanitize_doc_code(code: str) -> str:
    """Sanitizes document identifier preventing directory traversal or path separators."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', code.strip())
    if not clean or len(clean) < 3:
        raise HTTPException(status_code=400, detail="Código de documento inválido o demasiado corto.")
    return clean


def sanitize_filename(filename: str) -> str:
    """Returns a safe basename without path separators or dangerous characters."""
    base = Path(filename).name
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    if not clean or clean.startswith("."):
        clean = f"documento_{clean}"
    return clean


class QueryRequest(BaseModel):
    query: str = Field(..., description="Pregunta en lenguaje natural sobre el club o día a día")
    department_filter: Optional[str] = Field(None, description="Filtro opcional por departamento")
    top_k: int = Field(4, ge=1, le=15, description="Cantidad de fragmentos a recuperar")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Balance Híbrido: 1.0 = BM25, 0.0 = Vector, 0.5 = RRF")


class IngestDocRequest(BaseModel):
    doc_code: str = Field(..., description="Código único del documento (ej: MED-2026-11-08-01)")
    title: str = Field(..., description="Título oficial del documento")
    department_id: str = Field(..., description="ID del departamento responsable")
    author: str = Field("Staff Técnico", description="Autor o profesional emisor")
    date: str = Field("2026-11-08", description="Fecha en formato YYYY-MM-DD")
    tags: List[str] = Field(default=[], description="Etiquetas clave")
    content: str = Field(..., description="Contenido en formato Markdown con secciones")


@app.get("/api/status")
def get_status():
    stats = pipeline.get_stats()
    return {
        "status": "online",
        "club": "Club Atlético Puerto Marina",
        "version": "3.0.0",
        "stats": stats
    }


@app.get("/api/llm/health")
def get_llm_health():
    """Checks connectivity and available models on the configured LLM provider."""
    return pipeline.generator.check_llm_health()


@app.post("/api/query")
def run_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")
    return pipeline.ask(
        query=req.query,
        department_filter=req.department_filter,
        top_k=req.top_k,
        alpha=req.alpha
    )


@app.post("/api/reindex")
def reindex_knowledge_base():
    """Re-runs full ingestion of Markdown files and SQL relational tables."""
    res = run_full_ingestion(DB_PATH)
    pipeline.invalidate_cache()
    return {"status": "reindexed", "details": res}


@app.get("/api/documents")
def list_documents(department: Optional[str] = None):
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    if department:
        cur.execute("""
            SELECT d.*, dept.name as department_name, count(c.id) as chunk_count
            FROM documents d
            JOIN departments dept ON dept.id = d.department_id
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.department_id = ?
            GROUP BY d.id
            ORDER BY d.created_at DESC;
        """, (department,))
    else:
        cur.execute("""
            SELECT d.*, dept.name as department_name, count(c.id) as chunk_count
            FROM documents d
            JOIN departments dept ON dept.id = d.department_id
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC;
        """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    cur.execute(
        "SELECT id, chunk_index, title, department_id, content, metadata_json FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
        (doc_id,)
    )
    chunks = cur.fetchall()
    conn.close()
    return {"document": dict(doc), "chunks": [dict(c) for c in chunks]}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: int):
    """Deletes a document and all associated chunks from SQLite and FTS5."""
    ok = pipeline.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    return {
        "status": "deleted",
        "document_id": doc_id,
        "message": f"Documento #{doc_id} y sus fragmentos fueron eliminados exitosamente."
    }


@app.post("/api/documents")
def add_new_document(req: IngestDocRequest):
    """Ingests a new Markdown document with strict path traversal prevention."""
    dept_id = validate_department_id(req.department_id)
    safe_code = sanitize_doc_code(req.doc_code)

    raw_md = (
        f"---\n"
        f"id: {safe_code}\n"
        f"title: {req.title.strip()}\n"
        f"department: {dept_id}\n"
        f"author: {req.author.strip()}\n"
        f"date: {req.date.strip()}\n"
        f"tags: [{', '.join(t.strip() for t in req.tags if t.strip())}]\n"
        f"---\n\n"
        f"{req.content.strip()}\n"
    )

    base_raw = (BASE_DIR / "data" / "raw_documents").resolve()
    dept_folder = (base_raw / dept_id).resolve()
    if not str(dept_folder).startswith(str(base_raw)):
        raise HTTPException(status_code=400, detail="Intento de navegación de ruta no autorizada.")

    dept_folder.mkdir(parents=True, exist_ok=True)
    target_file = (dept_folder / f"{safe_code}.md").resolve()
    if not str(target_file).startswith(str(dept_folder)):
        raise HTTPException(status_code=400, detail="Ruta de destino inválida.")

    target_file.write_text(raw_md, encoding="utf-8")

    conn = get_db_connection(DB_PATH)
    embedder = get_embedding_engine()
    doc_id, chunk_count = ingest_document_file(target_file, conn, embedder)
    conn.close()

    pipeline.invalidate_cache()

    return {
        "status": "success",
        "document_id": doc_id,
        "doc_code": safe_code,
        "chunks_created": chunk_count,
        "message": f"Documento '{req.title}' indexado correctamente en la base de datos."
    }


@app.post("/api/upload")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    department_id: str = Form(...),
    author: Optional[str] = Form("Staff Técnico"),
    tags: Optional[str] = Form("")
):
    """
    Multi-format document upload endpoint (v3.0).
    Accepts PDF, DOCX, DOC, MD, TXT files.
    Validates magic bytes, size limits, extracts text, performs SHA-256 deduplication,
    indexes into chunks and FTS5, and invalidates vector cache.
    """
    dept_id = validate_department_id(department_id)
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos para cargar.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = (UPLOADS_DIR / dept_id).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    results = []
    successful_count = 0
    failed_count = 0

    conn = get_db_connection(DB_PATH)
    embedder = get_embedding_engine()

    try:
        for file in files:
            original_name = file.filename or "archivo_subido"
            ext = Path(original_name).suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": f"Formato no permitido '{ext}'. Formatos soportados: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                })
                failed_count += 1
                continue

            file_bytes = await file.read()
            if len(file_bytes) == 0:
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": "El archivo está vacío (0 bytes)."
                })
                failed_count += 1
                continue

            if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
                limit_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": f"Tamaño ({len(file_bytes)} bytes) excede el límite máximo de {limit_mb} MB."
                })
                failed_count += 1
                continue

            # Verify binary magic headers
            if ext == ".pdf" and not file_bytes.startswith(b"%PDF-"):
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": "Firma binaria corrupta o inválida. No es un documento PDF estándar."
                })
                failed_count += 1
                continue
            elif ext == ".docx" and not file_bytes.startswith(b"PK\x03\x04"):
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": "Firma binaria inválida. No es un documento DOCX válido."
                })
                failed_count += 1
                continue

            safe_name = sanitize_filename(original_name)
            dest_path = (target_dir / safe_name).resolve()
            if not str(dest_path).startswith(str(UPLOADS_DIR.resolve())):
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": "Nombre de archivo sospechoso (intento de Path Traversal bloqueado)."
                })
                failed_count += 1
                continue

            dest_path.write_bytes(file_bytes)

            try:
                doc_id, chunk_count, doc_code = ingest_uploaded_file(
                    file_path=dest_path,
                    original_filename=original_name,
                    department_id=dept_id,
                    author=author or "Staff Técnico",
                    tags=tags or "",
                    conn=conn,
                    embedder=embedder,
                    file_bytes=file_bytes
                )
                results.append({
                    "filename": original_name,
                    "doc_code": doc_code,
                    "document_id": doc_id,
                    "chunks_created": chunk_count,
                    "size_bytes": len(file_bytes),
                    "status": "success",
                    "message": f"Indexado exitosamente ({chunk_count} fragmentos generados)."
                })
                successful_count += 1
            except ExtractionError as ee:
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": str(ee)
                })
                failed_count += 1
            except Exception as e:
                results.append({
                    "filename": original_name,
                    "status": "error",
                    "message": f"Error al procesar e indexar: {str(e)}"
                })
                failed_count += 1

    finally:
        conn.close()

    pipeline.invalidate_cache()

    return {
        "status": "completed",
        "total_files": len(files),
        "successful": successful_count,
        "failed": failed_count,
        "details": results
    }


@app.get("/api/players")
def list_players():
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM players ORDER BY dorsal ASC;")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/staff")
def list_staff():
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, d.name as department_name 
        FROM staff s
        JOIN departments d ON d.id = s.department_id
        ORDER BY s.id ASC;
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/departments")
def list_departments():
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM departments ORDER BY id ASC;")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/logs")
def list_logs(limit: int = 20):
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?;", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Puerto Marina RAG Engine v3.0 is running."}
