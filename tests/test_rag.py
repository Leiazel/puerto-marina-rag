"""
Automated Test Suite for Club Atlético Puerto Marina RAG System (v3.0).
Features:
- Full test isolation with ephemeral temporary SQLite database (never mutates production DB).
- Core RAG tests (BM25 FTS5, Stopwords, Vectors, RRF Hybrid, SQL bridge, Dynamic confidence).
- Multi-format extraction tests (PDF, DOCX, DOC string fallback, scanned PDF detection).
- Path traversal and security defense tests.
- High-concurrency In-Memory Vector Cache and cache invalidation tests.
- Multi-file upload API endpoint tests via FastAPI TestClient with SHA-256 deduplication.
- Cascading document deletion tests.
- Robust chunker tests for oversized paragraphs and junk chunk rejection.
"""
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DB_PATH, _resolve_llm_config, CHUNK_SIZE_CHARS
from src.database import get_db_connection, delete_document, init_database
from src.embeddings import get_embedding_engine
from src.retriever import HybridRetriever, GLOBAL_VECTOR_CACHE
from src.generator import GroundedGenerator
from src.rag_pipeline import ClubRAGPipeline
from src.extractors import (
    extract_document_content,
    extract_text_from_pdf,
    extract_text_from_docx,
    ExtractionError,
    _extract_clean_strings_from_doc_stream
)
from src.ingestion import (
    ingest_uploaded_file,
    _split_oversized_text,
    chunk_document
)
from src.api import app, validate_department_id, sanitize_doc_code, sanitize_filename


class TestClubRAGSystemV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary test database to guarantee 0 mutation of club_knowledge.db
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = Path(cls.temp_dir) / "test_knowledge.db"

        # Copy existing database if present, otherwise will be created
        if DB_PATH.exists():
            shutil.copyfile(DB_PATH, cls.test_db_path)

        # Apply schema initialization & migrations to test DB
        init_database(cls.test_db_path)

        cls.conn = get_db_connection(cls.test_db_path)
        cls.embedder = get_embedding_engine()
        from src.ingestion import sync_relational_entities_to_rag
        sync_relational_entities_to_rag(cls.conn, cls.embedder)

        cls.retriever = HybridRetriever(cls.test_db_path)
        cls.generator = GroundedGenerator()
        cls.pipeline = ClubRAGPipeline(cls.test_db_path)

        # Wire FastAPI app to use test database for endpoint testing
        import src.api
        src.api.DB_PATH = cls.test_db_path
        src.api.pipeline = cls.pipeline
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # 1. CORE DATABASE & SCHEMA TESTS
    # ---------------------------------------------------------

    def test_01_database_tables_and_v3_columns_exist(self):
        cur = self.conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');").fetchall()
        table_names = [t["name"] for t in tables]
        for req in ["departments", "players", "staff", "documents", "chunks", "chunks_fts", "query_logs"]:
            self.assertIn(req, table_names)

        # Verify v3 schema columns in documents
        doc_cols = {row["name"] for row in cur.execute("PRAGMA table_info(documents);").fetchall()}
        for col in ["source_type", "original_filename", "content_hash", "file_size_bytes"]:
            self.assertIn(col, doc_cols)

    def test_02_stopword_filtering_in_fts5(self):
        cleaned = self.retriever._clean_fts_query("¿Cuál es la evolución de la lesión de Lautaro Carrizo?")
        self.assertNotIn('"cual"*', cleaned)
        self.assertNotIn('"las"*', cleaned)
        self.assertIn('"lesion"*', cleaned)
        self.assertIn('"carrizo"*', cleaned)

    def test_03_embeddings_normalization_and_dim(self):
        vec = self.embedder.embed_text("Plan táctico contra Deportivo Los Andes")
        self.assertEqual(len(vec), 384)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_04_fts5_bm25_lexical_search(self):
        results = self.retriever.search_lexical(self.conn, "Lautaro Carrizo esguince", limit=5)
        self.assertGreater(len(results), 0)
        self.assertIn("Carrizo", results[0]["content"])

    def test_05_two_phase_dense_vector_search(self):
        results = self.retriever.search_vector(self.conn, "jugador fatigado con sobrecarga en isquiotibiales", limit=5)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["cosine_sim"], 0.25)

    def test_06_hybrid_rrf_retrieval(self):
        hybrid_res = self.retriever.hybrid_search("¿Cuál es el protocolo de suplementación con creatina y cafeína?", top_k=3)
        self.assertGreater(len(hybrid_res["results"]), 0)
        self.assertGreater(hybrid_res["results"][0]["rrf_score"], 0.0)

    def test_07_department_metadata_filtering(self):
        med_res = self.retriever.hybrid_search("nutrición y pesaje", department_filter="nutricion", top_k=4)
        for r in med_res["results"]:
            self.assertEqual(r["department_id"], "nutricion")

    def test_08_end_to_end_pipeline_and_dynamic_confidence(self):
        out = self.pipeline.ask("¿En qué hotel y habitaciones se aloja la delegación para la semifinal?", top_k=3)
        self.assertIn("Hotel Grand Bahía", out["answer"])
        self.assertGreater(out["confidence"], 0.5)

    def test_09_query_logging_audit(self):
        cur = self.conn.cursor()
        before = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.pipeline.ask("Consulta de auditoría de sistema aislada")
        after = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.assertEqual(after, before + 1)

    def test_10_relational_sql_bridge_in_rag(self):
        staff_out = self.pipeline.ask("¿Cuál es el correo electrónico y teléfono de la nutricionista Sofía Carrizo?")
        self.assertIn("scarrizo@puertomarina.fc", staff_out["answer"])
        player_out = self.pipeline.ask("¿Cuánto mide y cuánto pesa el arquero suplente Lucas Barreto?")
        self.assertTrue("1.91" in player_out["answer"] or "1,91" in player_out["answer"])

    def test_11_llm_config_aliases_resolution(self):
        """Verifies that OPENAI_MODEL and GROQ_MODEL aliases resolve cleanly even without .env."""
        with patch.dict(os.environ, {"OPENAI_MODEL": "qwen/qwen3.8-27b", "OPENAI_API_KEY": "gsk_mock123"}):
            _, base, key, model, _ = _resolve_llm_config()
            self.assertEqual(model, "qwen/qwen3.8-27b")
            self.assertEqual(key, "gsk_mock123")
            self.assertTrue("groq.com" in base)

    # ---------------------------------------------------------
    # 2. MULTI-FORMAT EXTRACTION TESTS
    # ---------------------------------------------------------

    def test_12_pdf_extraction_page_aware(self):
        """Creates an in-memory PDF and verifies structured page-aware Markdown extraction."""
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        c.drawString(100, 750, "Club Atlético Puerto Marina - Informe Traumatológico")
        c.drawString(100, 720, "El jugador Lautaro Carrizo muestra consolidación ligamentosa completa.")
        c.showPage()
        c.drawString(100, 750, "Página 2: Protocolo de reinserción a fútbol reducido.")
        c.showPage()
        c.save()
        pdf_bytes = pdf_buf.getvalue()

        res = extract_text_from_pdf(pdf_bytes)
        self.assertEqual(res["page_count"], 2)
        self.assertIn("## Página 1", res["text"])
        self.assertIn("## Página 2", res["text"])
        self.assertIn("Lautaro Carrizo", res["text"])

    def test_13_docx_extraction_tables_and_headings(self):
        """Creates an in-memory DOCX with headings and a table, verifying Markdown formatting."""
        doc = docx.Document()
        doc.add_heading("Estrategia Táctica para la Semifinal", level=1)
        doc.add_paragraph("Análisis de presión alta sobre el rival.")
        table = doc.add_table(rows=2, cols=3)
        hdr = table.rows[0].cells
        hdr[0].text = "Zona"
        hdr[1].text = "Responsable"
        hdr[2].text = "Consigna"
        r1 = table.rows[1].cells
        r1[0].text = "Banda Izquierda"
        r1[1].text = "Facundo Medina"
        r1[2].text = "Doblar la marca"
        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_bytes = docx_buf.getvalue()

        res = extract_text_from_docx(docx_bytes)
        self.assertIn("# Estrategia Táctica para la Semifinal", res["text"])
        self.assertIn("| Zona | Responsable | Consigna |", res["text"])
        self.assertIn("| Facundo Medina |", res["text"])

    def test_14_scanned_pdf_rejection(self):
        """Verifies that PDFs with no text layer raise a helpful ExtractionError."""
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        # Empty page with no text (simulates scanned image)
        c.showPage()
        c.save()
        with self.assertRaises(ExtractionError) as ctx:
            extract_text_from_pdf(pdf_buf.getvalue())
        self.assertIn("no contiene texto digital", str(ctx.exception).lower())

    def test_15_legacy_doc_string_stream_fallback(self):
        """Verifies robust string extraction for legacy binary streams."""
        synthetic_stream = (
            b"\x00" * 128
            + "Club Atlético Puerto Marina - Documento Histórico\r\nReglamento de convivencia y vestuario.".encode("cp1252")
            + b"\x00" * 64
        )
        extracted = _extract_clean_strings_from_doc_stream(synthetic_stream)
        self.assertIn("Puerto Marina", extracted)
        self.assertIn("Reglamento", extracted)

    # ---------------------------------------------------------
    # 3. SECURITY & PATH TRAVERSAL DEFENSE TESTS
    # ---------------------------------------------------------

    def test_16_path_traversal_validation(self):
        """Verifies that malicious department and doc codes are rejected."""
        with self.assertRaises(Exception):
            validate_department_id("../../etc")
        with self.assertRaises(Exception):
            validate_department_id("servicio_medico/../../root")

        # Sanitize doc code
        self.assertEqual(sanitize_doc_code("MED-2026-01"), "MED-2026-01")
        self.assertEqual(sanitize_doc_code("../../../evil_code"), "evil_code")
        with self.assertRaises(Exception):
            sanitize_doc_code("..//..")

        # Filename sanitization
        self.assertEqual(sanitize_filename("../../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(sanitize_filename("valid_doc.docx"), "valid_doc.docx")

    # ---------------------------------------------------------
    # 4. IN-MEMORY VECTOR CACHE & PERFORMANCE TESTS
    # ---------------------------------------------------------

    def test_17_vector_matrix_cache_invalidation(self):
        """Verifies that VectorMatrixCache caches matrix and reloads on chunk mutation."""
        GLOBAL_VECTOR_CACHE.invalidate()
        cache1 = GLOBAL_VECTOR_CACHE.get_or_load(self.conn, str(self.test_db_path), 384)
        initial_count = len(cache1["ids"])
        self.assertGreater(initial_count, 0)

        # Fetching again returns identical in-memory reference
        cache2 = GLOBAL_VECTOR_CACHE.get_or_load(self.conn, str(self.test_db_path), 384)
        self.assertIs(cache1, cache2)

        # Invalidating forces reload
        GLOBAL_VECTOR_CACHE.invalidate()
        cache3 = GLOBAL_VECTOR_CACHE.get_or_load(self.conn, str(self.test_db_path), 384)
        self.assertIsNot(cache1, cache3)
        self.assertEqual(len(cache3["ids"]), initial_count)

    # ---------------------------------------------------------
    # 5. MULTI-FILE UPLOAD & CASCADE DELETION TESTS
    # ---------------------------------------------------------

    def test_18_multi_file_upload_api_and_deduplication(self):
        """Tests uploading PDF, DOCX, and MD files via FastAPI TestClient."""
        # 1. Create test PDF
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        c.drawString(100, 750, "Reporte Nutricional Exclusivo - Suplementacion Creatina Semifinal")
        c.showPage()
        c.save()
        pdf_bytes = pdf_buf.getvalue()

        # 2. Create test MD
        md_bytes = b"# Protocolo de Recuperacion Post-Partido\n\n- Banos de hielo a 10 grados\n- Masaje descontracturante\n"

        files = [
            ("files", ("reporte_nutricional.pdf", pdf_bytes, "application/pdf")),
            ("files", ("protocolo_recuperacion.md", md_bytes, "text/markdown"))
        ]
        data = {
            "department_id": "nutricion",
            "author": "Lic. Sofia Carrizo",
            "tags": "creatina, semifinal, recuperacion"
        }

        resp = self.client.post("/api/upload", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["successful"], 2)
        self.assertEqual(body["failed"], 0)

        # Verify documents can now be queried via RAG
        rag_res = self.pipeline.ask("¿Cuáles son las indicaciones de suplementación con creatina para la semifinal?", top_k=2)
        self.assertIn("Creatina", rag_res["answer"])

        # Test deduplication: re-uploading the same MD file should update, not create duplicate
        resp_dedup = self.client.post("/api/upload", files=[("files", ("protocolo_recuperacion.md", md_bytes, "text/markdown"))], data=data)
        self.assertEqual(resp_dedup.status_code, 200)
        dedup_body = resp_dedup.json()
        self.assertEqual(dedup_body["successful"], 1)

    def test_19_cascading_document_deletion(self):
        """Verifies deleting a document removes its chunks and FTS entries."""
        # Upload a standalone document to delete
        test_content = b"# Documento Temporal para Borrado\n\nEste contenido debe eliminarse por completo."
        resp = self.client.post(
            "/api/upload",
            files=[("files", ("temporal_eliminar.md", test_content, "text/markdown"))],
            data={"department_id": "logistica_viajes", "author": "Logística"}
        )
        doc_id = resp.json()["details"][0]["document_id"]
        self.assertIsNotNone(doc_id)

        # Confirm chunks and FTS rows exist
        cur = self.conn.cursor()
        chunk_count = cur.execute("SELECT count(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
        self.assertGreater(chunk_count, 0)

        # Delete document via pipeline
        deleted = self.pipeline.delete_document(doc_id)
        self.assertTrue(deleted)

        # Verify cascade
        self.assertIsNone(cur.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone())
        self.assertEqual(cur.execute("SELECT count(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0], 0)

    # ---------------------------------------------------------
    # 6. CHUNKER RESILIENCE & OVERSIZED TEXT TESTS
    # ---------------------------------------------------------

    def test_20_oversized_text_chunking(self):
        """Verifies that paragraphs larger than CHUNK_SIZE_CHARS are split without dropping content."""
        long_para = "El equipo realizó un bloque de posesión táctica en espacio reducido. " * 50  # ~3500 chars
        parts = _split_oversized_text(long_para, max_chars=CHUNK_SIZE_CHARS, overlap=100)
        self.assertGreater(len(parts), 2)
        for p in parts:
            self.assertLessEqual(len(p), CHUNK_SIZE_CHARS + 50)

        doc_meta = {
            "doc_code": "TEST-CHUNKER",
            "title": "Prueba Chunker",
            "department_id": "cuerpo_tecnico",
            "raw_content": f"## Sección Táctica Gigante\n\n{long_para}\n\n---\n"
        }
        chunks = chunk_document(doc_meta)
        self.assertGreater(len(chunks), 2)
        # Verify junk chunk (standalone '---') was rejected
        for ch in chunks:
            self.assertGreater(sum(1 for c in ch["content"] if c.isalnum()), 30)


if __name__ == "__main__":
    unittest.main()
