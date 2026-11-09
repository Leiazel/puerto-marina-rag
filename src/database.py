"""
Database module for SQLite with FTS5 and Vector BLOB storage.
Optimized for ultra-fast hybrid queries (Lexical BM25 + Two-Phase Dense Vectors).
"""
import sqlite3
from pathlib import Path
from src.config import DB_PATH


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory and WAL high-concurrency pragma."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(db_path: Path = DB_PATH) -> None:
    """Creates relational tables, indexes, and FTS5 virtual table."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Departments table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        manager TEXT
    );
    """)

    # 2. Players table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dorsal INTEGER NOT NULL UNIQUE,
        name TEXT NOT NULL,
        nickname TEXT,
        position TEXT NOT NULL,
        age INTEGER,
        height_m REAL,
        weight_kg REAL,
        notes TEXT
    );
    """)

    # 3. Staff members table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS staff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        department_id TEXT NOT NULL REFERENCES departments(id),
        email TEXT,
        phone TEXT
    );
    """)

    # 4. Documents table (enhanced in v3.0 with source_type, original_filename, content_hash, file_size_bytes)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_code TEXT UNIQUE,
        title TEXT NOT NULL,
        department_id TEXT NOT NULL REFERENCES departments(id),
        author TEXT,
        date TEXT,
        file_path TEXT,
        confidentiality TEXT DEFAULT 'interno',
        tags TEXT,
        raw_content TEXT NOT NULL,
        source_type TEXT DEFAULT 'markdown',
        original_filename TEXT,
        content_hash TEXT,
        file_size_bytes INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Schema migration: Add columns to existing documents table if not present
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(documents);").fetchall()}
    if "source_type" not in existing_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN source_type TEXT DEFAULT 'markdown';")
    if "original_filename" not in existing_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN original_filename TEXT;")
    if "content_hash" not in existing_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT;")
    if "file_size_bytes" not in existing_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN file_size_bytes INTEGER DEFAULT 0;")

    # 5. Chunks table (Stores text + float32 vector embedding as BLOB)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        title TEXT NOT NULL,
        department_id TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 6. FTS5 Virtual Table for full-text search with BM25 ranking
    # We use rowid directly aligned with chunks.id for fast integer JOINs
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id UNINDEXED,
        title,
        department_id,
        content,
        tags,
        tokenize = 'unicode61 remove_diacritics 2'
    );
    """)

    # 7. Query logs for monitoring and audit
    cur.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT NOT NULL,
        department_filter TEXT,
        retrieved_chunks_count INTEGER,
        top_similarity REAL,
        response_text TEXT,
        retrieval_ms REAL,
        total_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Performance Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_dept ON chunks(department_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_dept ON documents(department_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON documents(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_code ON documents(doc_code);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);")

    conn.commit()
    conn.close()


def delete_document(doc_id: int, conn: sqlite3.Connection) -> bool:
    """
    Cascades deletion of a document:
    1. Removes associated FTS5 rows.
    2. Removes chunk records.
    3. Deletes document record.
    Returns True if a document was deleted, False otherwise.
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
    if not cur.fetchone():
        return False

    # Get chunk IDs to delete from FTS5
    cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
    chunk_ids = [r["id"] for r in cur.fetchall()]

    for cid in chunk_ids:
        cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))

    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    return True
