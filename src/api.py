"""
FastAPI REST API server for Club Atlético Puerto Marina RAG System (v2.0).
"""
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import BASE_DIR, DB_PATH
from src.database import get_db_connection
from src.rag_pipeline import ClubRAGPipeline
from src.ingestion import ingest_document_file, run_full_ingestion
from src.embeddings import get_embedding_engine

app = FastAPI(
    title="Club Atlético Puerto Marina - RAG Knowledge Engine v2.0",
    description="Hybrid RAG system (SQLite FTS5 BM25 + FastEmbed ONNX + RRF) for football club operations.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClubRAGPipeline(db_path=DB_PATH)
STATIC_DIR = BASE_DIR / "web" / "static"


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
    res = run_full_ingestion()
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
            ORDER BY d.date DESC;
        """, (department,))
    else:
        cur.execute("""
            SELECT d.*, dept.name as department_name, count(c.id) as chunk_count
            FROM documents d
            JOIN departments dept ON dept.id = d.department_id
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.date DESC;
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
        "SELECT id, chunk_index, title, content, metadata_json FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
        (doc_id,)
    )
    chunks = cur.fetchall()
    conn.close()
    return {"document": dict(doc), "chunks": [dict(c) for c in chunks]}


@app.post("/api/documents")
def add_new_document(req: IngestDocRequest):
    raw_md = (
        f"---\n"
        f"id: {req.doc_code}\n"
        f"title: {req.title}\n"
        f"department: {req.department_id}\n"
        f"author: {req.author}\n"
        f"date: {req.date}\n"
        f"tags: [{', '.join(req.tags)}]\n"
        f"---\n\n"
        f"{req.content}\n"
    )
    dept_folder = BASE_DIR / "data" / "raw_documents" / req.department_id
    dept_folder.mkdir(parents=True, exist_ok=True)
    target_file = dept_folder / f"{req.doc_code}.md"
    target_file.write_text(raw_md, encoding="utf-8")

    conn = get_db_connection(DB_PATH)
    embedder = get_embedding_engine()
    doc_id, chunk_count = ingest_document_file(target_file, conn, embedder)
    conn.close()

    return {
        "status": "success",
        "document_id": doc_id,
        "doc_code": req.doc_code,
        "chunks_created": chunk_count,
        "message": f"Documento '{req.title}' indexado correctamente en la base de datos."
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
def list_logs(limit: int = 15):
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
        return {"message": "Puerto Marina RAG Engine v2.0 is running."}
