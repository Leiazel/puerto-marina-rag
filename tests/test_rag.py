"""
Automated Test Suite for Club Atlético Puerto Marina RAG System (v3.0).
All tests run against an isolated temporary SQLite database, document copy,
and uploads directory so production data/ is never mutated by test runs.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import List
from unittest.mock import patch

import docx
import numpy as np
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import src.api as api_module
from src.config import BASE_DIR, RAW_DOCS_DIR, _resolve_llm_config
from src.database import get_db_connection, init_database
from src.embeddings import get_embedding_engine
from src.extractors import (
    ExtractionError,
    _extract_clean_strings_from_doc_stream,
    extract_document_content,
    extract_text_from_docx,
    extract_text_from_pdf,
    validate_file_magic_bytes,
)
from src.generator import GroundedGenerator, _clean_llm_output
from src.ingestion import _portable_file_path, _split_long_line, _split_text_lines, run_full_ingestion
from src.rag_pipeline import ClubRAGPipeline
from src.retriever import HybridRetriever, VectorMatrixCache


def _build_minimal_pdf(pages_text: List[str]) -> bytes:
    """Constructs a valid multi-page PDF 1.4 binary in pure Python without external generators."""
    objects: List[bytes] = []
    # Object 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Object 2 placeholder: Pages
    page_obj_ids = [4 + i * 2 for i in range(len(pages_text))]
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages_text)} >>".encode("latin-1"))
    # Object 3: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(pages_text):
        page_id = 4 + i * 2
        content_id = page_id + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_str = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n"
        stream_bytes = stream_str.encode("latin-1", errors="replace")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("latin-1")
        )
        objects.append(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream"
        )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj_body in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(f"{idx} 0 obj\n".encode("latin-1"))
        buf.write(obj_body)
        buf.write(b"\nendobj\n")

    xref_offset = buf.tell()
    buf.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("latin-1"))

    buf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )
    return buf.getvalue()


def _build_minimal_docx() -> bytes:
    """Creates an in-memory DOCX document with headings, prose, and a table."""
    document = docx.Document()
    document.add_heading("Informe Táctico Semifinal", level=1)
    document.add_heading("Análisis del Rival", level=2)
    document.add_paragraph("El equipo rival presiona alto con bloque 4-4-2 en campo contrario y sufre a espaldas de sus laterales.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Zona | Sector"
    table.cell(0, 1).text = "Prioridad"
    table.cell(1, 0).text = "Banda Derecha"
    table.cell(1, 1).text = "Alta"
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestClubRAGSystemV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory(prefix="puerto_marina_test_")
        cls.tmp_root = Path(cls.tmp_dir.name)
        cls.tmp_docs_dir = cls.tmp_root / "raw_documents"
        cls.tmp_uploads_dir = cls.tmp_root / "uploads"
        cls.tmp_uploads_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(RAW_DOCS_DIR, cls.tmp_docs_dir)
        cls.tmp_db_path = cls.tmp_root / "test_club_knowledge.db"

        cls.ingest_summary = run_full_ingestion(
            db_path=cls.tmp_db_path,
            docs_dir=cls.tmp_docs_dir,
            uploads_dir=cls.tmp_uploads_dir,
        )
        cls.conn = get_db_connection(cls.tmp_db_path)
        cls.embedder = get_embedding_engine()
        cls.retriever = HybridRetriever(cls.tmp_db_path)
        cls.generator = GroundedGenerator()
        cls.pipeline = ClubRAGPipeline(cls.tmp_db_path)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        cls.tmp_dir.cleanup()

    def test_01_database_tables_and_indexes_exist(self):
        cur = self.conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');").fetchall()
        table_names = [t["name"] for t in tables]
        for req in ["departments", "players", "staff", "documents", "chunks", "chunks_fts", "query_logs"]:
            self.assertIn(req, table_names)

        # Ensure _portable_file_path produces relative paths for workspace documents
        sample_md = next(RAW_DOCS_DIR.rglob("*.md"))
        rel = _portable_file_path(sample_md)
        self.assertFalse(rel.startswith("/"))
        self.assertTrue(rel.startswith("data/raw_documents/"))
        self.assertEqual(len(list(BASE_DIR.glob("data/raw_documents/**/*.md"))), 17)

    def test_02_stopword_filtering_and_short_domain_terms(self):
        cleaned = self.retriever._clean_fts_query("¿Cuál es la evolución de la lesión de Lautaro Carrizo?")
        self.assertNotIn('"cual"*', cleaned)
        self.assertNotIn('"las"*', cleaned)
        self.assertIn('"lesion"*', cleaned)
        self.assertIn('"carrizo"*', cleaned)

        # 2-char football domain terms (xG, xA, DT) must be preserved
        cleaned_xg = self.retriever._clean_fts_query("¿Qué xG y xA tiene el DT?")
        self.assertIn('"xg"*', cleaned_xg)
        self.assertIn('"xa"*', cleaned_xg)
        self.assertIn('"dt"*', cleaned_xg)

        # Pure stopword query should return empty FTS query (not fall back to OR of stopwords)
        self.assertEqual(self.retriever._clean_fts_query("¿Cuál es de la para el?"), "")

    def test_03_embeddings_normalization_dim_and_empty_batch(self):
        vec = self.embedder.embed_text("Plan táctico contra Deportivo Los Andes")
        self.assertEqual(len(vec), 384)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

        empty_vecs = self.embedder.embed_batch([])
        self.assertEqual(empty_vecs, [])
        batch_vecs = self.embedder.embed_batch(["Puerto Marina", "Semifinal"])
        self.assertEqual(len(batch_vecs), 2)
        self.assertEqual(len(batch_vecs[0]), 384)

    def test_04_fts5_bm25_lexical_search(self):
        results = self.retriever.search_lexical(self.conn, "Lautaro Carrizo esguince", limit=5)
        self.assertGreater(len(results), 0)
        self.assertIn("Carrizo", results[0]["content"])

        xg_results = self.retriever.search_lexical(self.conn, "xG xA Cáceres", limit=5)
        self.assertGreater(len(xg_results), 0)

    def test_05_two_phase_dense_vector_search(self):
        results = self.retriever.search_vector(self.conn, "jugador fatigado con sobrecarga en isquiotibiales", limit=5)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["cosine_sim"], 0.25)

    def test_06_hybrid_rrf_and_pure_alpha_modes(self):
        hybrid_res = self.retriever.hybrid_search("¿Cuál es el protocolo de suplementación con creatina y cafeína?", top_k=3)
        self.assertGreater(len(hybrid_res["results"]), 0)
        self.assertGreater(hybrid_res["results"][0]["rrf_score"], 0.0)

        # Pure lexical mode (alpha=1.0) must skip vector search
        lex_only = self.retriever.hybrid_search("Lautaro Carrizo", top_k=3, alpha=1.0)
        self.assertGreater(len(lex_only["results"]), 0)
        for r in lex_only["results"]:
            self.assertIsNone(r["vector_rank"])
            self.assertIn("lexical", r["modalities"])

        # Pure vector mode (alpha=0.0) must skip lexical search
        vec_only = self.retriever.hybrid_search("sobrecarga muscular y fatiga", top_k=3, alpha=0.0)
        self.assertGreater(len(vec_only["results"]), 0)
        for r in vec_only["results"]:
            self.assertIsNone(r["lexical_rank"])
            self.assertIn("dense_vector", r["modalities"])

    def test_07_department_metadata_filtering(self):
        med_res = self.retriever.hybrid_search("nutrición y pesaje", department_filter="nutricion", top_k=4)
        self.assertGreater(len(med_res["results"]), 0)
        for r in med_res["results"]:
            self.assertEqual(r["department_id"], "nutricion")

    def test_08_chunk_quality_table_headers_and_orphan_cleanup(self):
        rows = self.conn.execute("SELECT content FROM chunks").fetchall()
        self.assertGreater(len(rows), 80)
        for r in rows:
            content = r["content"]
            self.assertGreater(len(content), 100)
            pipe_lines = [l.strip() for l in content.splitlines() if l.strip().startswith("|")]
            if pipe_lines:
                # Every chunk containing Markdown table rows must preserve the header separator
                self.assertTrue(any("---" in l for l in pipe_lines), f"Missing table header separator in chunk: {pipe_lines[:2]}")

        # Orphan cleanup test on separate temporary dir
        with tempfile.TemporaryDirectory(prefix="orphan_test_") as od:
            od_path = Path(od)
            docs_copy = od_path / "docs"
            shutil.copytree(RAW_DOCS_DIR, docs_copy)
            temp_db = od_path / "orphan.db"
            extra_doc = docs_copy / "nutricion" / "TEMP-ORPHAN-01.md"
            extra_doc.write_text(
                "---\nid: TEMP-ORPHAN-01\ntitle: Doc Temporal\ndepartment: nutricion\nauthor: Test\ndate: 2026-11-08\n---\n\n## Sección\nContenido temporal.\n",
                encoding="utf-8",
            )
            run_full_ingestion(db_path=temp_db, docs_dir=docs_copy)
            c = get_db_connection(temp_db)
            self.assertIsNotNone(c.execute("SELECT id FROM documents WHERE doc_code = 'TEMP-ORPHAN-01'").fetchone())
            c.close()

            extra_doc.unlink()
            run_full_ingestion(db_path=temp_db, docs_dir=docs_copy)
            c2 = get_db_connection(temp_db)
            self.assertIsNone(c2.execute("SELECT id FROM documents WHERE doc_code = 'TEMP-ORPHAN-01'").fetchone())
            c2.close()

    def test_09_end_to_end_pipeline_and_out_of_domain_guard(self):
        out = self.pipeline.ask("¿En qué hotel y habitaciones se aloja la delegación para la semifinal?", top_k=3)
        self.assertIn("Hotel Grand Bahía", out["answer"])
        self.assertGreater(out["confidence"], 0.45)

        # Out-of-domain query must return low-confidence insufficient_evidence guard
        ood = self.pipeline.ask("¿Cuál es el precio de Bitcoin hoy en Nueva York?", top_k=3)
        self.assertEqual(ood["model_used"], "insufficient_evidence")
        self.assertLess(ood["confidence"], 0.25)

    def test_10_query_logging_audit(self):
        before = self.conn.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.pipeline.ask("Consulta de auditoría de sistema sobre Agustín Ríos")
        after = self.conn.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.assertEqual(after, before + 1)

    def test_11_relational_sql_bridge_in_rag(self):
        """Verifies that SQL table rows (staff & players) are indexed and retrievable via RAG."""
        staff_out = self.pipeline.ask("¿Cuál es el correo electrónico y teléfono de la nutricionista Sofía Carrizo?")
        self.assertIn("scarrizo@puertomarina.fc", staff_out["answer"])
        player_out = self.pipeline.ask("¿Cuánto mide y cuánto pesa el arquero suplente Lucas Barreto?")
        self.assertTrue("1.91" in player_out["answer"] or "1,91" in player_out["answer"])

    def test_12_llm_config_precedence_and_mocked_resilience(self):
        # 1) Explicit USE_EXTERNAL_LLM=false must override presence of API key
        with patch.dict(os.environ, {"USE_EXTERNAL_LLM": "false", "LLM_API_KEY": "sk-dummy"}, clear=False):
            use_llm, _, _, _, _ = _resolve_llm_config()
            self.assertFalse(use_llm)

        # 2) Local Ollama endpoint does not require an API key to be active
        gen = GroundedGenerator()
        gen.use_llm = True
        gen.api_base = "http://localhost:11434/v1"
        gen.api_key = ""
        self.assertTrue(gen.is_llm_active())

        # 3) Output cleaning handles both closed and unclosed <think> tags
        self.assertEqual(_clean_llm_output("<think>razonamiento</think>\nRespuesta final."), "Respuesta final.")
        self.assertEqual(_clean_llm_output("<think>razonamiento truncado sin cierre"), "")

        # 4) Mocked 404 model_not_found auto-recovery + <think> tag cleaning
        gen.api_base = "https://api.openai.com/v1"
        gen.api_key = "sk-test-mock"
        gen.model = "nonexistent-model"
        chunks = self.retriever.hybrid_search("Lesión de Lautaro Carrizo", top_k=2)["results"]

        call_state = {"calls": 0}

        def fake_urlopen(req, timeout=20):
            url = req.full_url
            if url.endswith("/models"):
                return _FakeHTTPResponse({"data": [{"id": "gpt-4.1-mini"}, {"id": "dall-e-3"}]})
            call_state["calls"] += 1
            if call_state["calls"] == 1:
                raise urllib.error.HTTPError(
                    url, 404, "Not Found", {}, io.BytesIO(b'{"error":{"code":"model_not_found"}}')
                )
            return _FakeHTTPResponse({
                "choices": [{"message": {"content": "<think>interno</think>\nEsguince grado II en tobillo derecho [1]."}}]
            })

        with patch.object(gen, "_refresh_config_if_needed", return_value=None):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                res = gen.generate_grounded_answer("¿Qué lesión tiene Lautaro Carrizo?", chunks)
                self.assertEqual(gen.model, "gpt-4.1-mini")
                self.assertTrue(res["model_used"].startswith("llm:"))
                self.assertIn("Esguince grado II", res["answer"])
                self.assertNotIn("<think>", res["answer"])
                self.assertFalse(res["fallback_used"])

            # 5) Mocked 429 retry then content:null fallback telemetry
            attempt_counter = {"n": 0}

            def fake_429_then_null(req, timeout=20):
                attempt_counter["n"] += 1
                if attempt_counter["n"] == 1:
                    raise urllib.error.HTTPError(url=req.full_url, code=429, msg="Too Many Requests", hdrs={"Retry-After": "0"}, fp=io.BytesIO(b"rate limit"))
                return _FakeHTTPResponse({"choices": [{"message": {"content": None}}]})

            with patch("time.sleep", return_value=None):
                with patch("urllib.request.urlopen", side_effect=fake_429_then_null):
                    fallback_res = gen.generate_grounded_answer("¿Qué lesión tiene Lautaro Carrizo?", chunks)
                    self.assertTrue(fallback_res["fallback_used"])
                    self.assertIsNotNone(fallback_res["fallback_reason"])
                    self.assertEqual(fallback_res["model_used"], "grounded_extractive_v2")

    def test_13_fastapi_endpoints_and_security(self):
        orig_pipeline = api_module.pipeline
        orig_docs_dir = api_module.RAW_DOCS_DIR
        orig_uploads_dir = api_module.UPLOADS_DIR
        test_pipeline = ClubRAGPipeline(db_path=self.tmp_db_path)

        with (
            patch.object(api_module, "pipeline", test_pipeline),
            patch.object(api_module, "RAW_DOCS_DIR", self.tmp_docs_dir),
            patch.object(api_module, "UPLOADS_DIR", self.tmp_uploads_dir),
        ):
            with TestClient(api_module.app) as client:
                # Health & Status & Frontend
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/").status_code, 200)
                st = client.get("/api/status")
                self.assertEqual(st.status_code, 200)
                self.assertGreaterEqual(st.json()["stats"]["documents_count"], 19)

                llm_h = client.get("/api/llm/health")
                self.assertEqual(llm_h.status_code, 200)
                self.assertIn("status", llm_h.json())

                # Query validation & execution
                bad_q = client.post("/api/query", json={"query": "   "})
                self.assertEqual(bad_q.status_code, 400)
                bad_dept = client.post("/api/query", json={"query": "lesión", "department_filter": "inventado"})
                self.assertEqual(bad_dept.status_code, 400)
                good_q = client.post("/api/query", json={"query": "¿Qué lesión tiene Lautaro Carrizo?", "top_k": 3})
                self.assertEqual(good_q.status_code, 200)
                self.assertIn("Carrizo", good_q.json()["answer"])

                # Path traversal & invalid doc_code rejected with 400
                trav = client.post("/api/documents", json={
                    "doc_code": "../../etc/passwd",
                    "title": "Malicioso",
                    "department_id": "nutricion",
                    "content": "Prueba"
                })
                self.assertEqual(trav.status_code, 400)

                # YAML special characters (colons, quotes) safe ingestion + detail + delete
                created = client.post("/api/documents", json={
                    "doc_code": "TEST-YAML-2026-01",
                    "title": 'Reporte: "Estado de Cáceres" #1',
                    "department_id": "scouting",
                    "author": "Lucas: Benítez",
                    "date": "2026-11-08",
                    "tags": ["test", "scouting:2026"],
                    "content": "## Evaluación\nInforme de prueba con caracteres especiales."
                })
                self.assertEqual(created.status_code, 200)
                doc_id = created.json()["document_id"]
                created_file = self.tmp_docs_dir / "scouting" / "TEST-YAML-2026-01.md"
                self.assertTrue(created_file.exists())

                detail = client.get(f"/api/documents/{doc_id}")
                self.assertEqual(detail.status_code, 200)
                self.assertEqual(detail.json()["document"]["title"], 'Reporte: "Estado de Cáceres" #1')

                # Cannot delete virtual sqlite:// document
                docs_list = client.get("/api/documents").json()
                sql_doc = next(d for d in docs_list if str(d["file_path"]).startswith("sqlite://"))
                del_sql = client.delete(f"/api/documents/{sql_doc['id']}")
                self.assertEqual(del_sql.status_code, 400)

                # Delete custom document cleanly
                del_res = client.delete(f"/api/documents/{doc_id}")
                self.assertEqual(del_res.status_code, 200)
                self.assertFalse(created_file.exists())

                # Reindex, players, staff, departments, logs bounds
                self.assertEqual(client.post("/api/reindex").status_code, 200)
                self.assertGreater(len(client.get("/api/players").json()), 0)
                self.assertGreater(len(client.get("/api/staff").json()), 0)
                self.assertEqual(len(client.get("/api/departments").json()), 7)
                self.assertEqual(client.get("/api/logs?limit=5").status_code, 200)
                self.assertEqual(client.get("/api/logs?limit=999").status_code, 422)

        self.assertIs(api_module.pipeline, orig_pipeline)
        self.assertEqual(api_module.RAW_DOCS_DIR, orig_docs_dir)
        self.assertEqual(api_module.UPLOADS_DIR, orig_uploads_dir)

    def test_14_pdf_extraction_magic_bytes_and_scanned_rejection(self):
        pdf_bytes = _build_minimal_pdf([
            "Parte Medico Oficial de Lautaro Carrizo: evolucion favorable en fase 3 RTP.",
            "Segundo bloque con carga GPS Catapult Vector S7 sin sobrecarga muscular.",
        ])
        validate_file_magic_bytes("parte_medico.pdf", pdf_bytes)
        res = extract_text_from_pdf(pdf_bytes)
        self.assertEqual(res["format"], "pdf")
        self.assertEqual(res["page_count"], 2)
        self.assertIn("## Página 1", res["text"])
        self.assertIn("## Página 2", res["text"])
        self.assertIn("Lautaro Carrizo", res["text"])

        # Corrupt magic bytes rejected
        with self.assertRaises(ExtractionError):
            validate_file_magic_bytes("falso.pdf", b"NO_ES_UN_PDF_REAL_123456")

        # Empty/scanned PDF with no text layer rejected
        scanned_pdf = _build_minimal_pdf(["   "])
        with self.assertRaises(ExtractionError):
            extract_text_from_pdf(scanned_pdf)

    def test_15_docx_doc_and_text_extraction_validation(self):
        docx_bytes = _build_minimal_docx()
        validate_file_magic_bytes("informe_tactico.docx", docx_bytes)
        res = extract_text_from_docx(docx_bytes)
        self.assertEqual(res["format"], "docx")
        self.assertIn("# Informe Táctico Semifinal", res["text"])
        self.assertIn("## Análisis del Rival", res["text"])
        self.assertIn("Zona \\| Sector", res["text"])
        self.assertIn("| Banda Derecha | Alta |", res["text"])

        # Corrupt DOCX magic bytes rejected
        with self.assertRaises(ExtractionError):
            validate_file_magic_bytes("corrupto.docx", b"NOT_A_ZIP_CONTAINER")

        # Corrupt legacy .doc OLE2 magic bytes rejected
        with self.assertRaises(ExtractionError):
            validate_file_magic_bytes("legado.doc", b"NOT_AN_OLE2_HEADER")

        # Valid OLE2 magic bytes pass header validation, and UTF-16LE/ASCII fallback scanner works
        ole_payload = "Informe tactico legado de Puerto Marina para semifinal.".encode("utf-16le")
        scanned = _extract_clean_strings_from_doc_stream(b"\x00\x00" + ole_payload + b"\x00\x00")
        self.assertIn("Informe tactico legado", scanned)

        # Text / Markdown extraction with UTF-8 BOM and null-byte guard
        txt_bytes = "\ufeffPlan nutricional pre-competitivo de hidratación.".encode("utf-8")
        validate_file_magic_bytes("notas.txt", txt_bytes)
        txt_res = extract_document_content(txt_bytes, "notas.txt")
        self.assertEqual(txt_res["format"], "txt")
        self.assertTrue(txt_res["text"].startswith("Plan nutricional"))
        with self.assertRaises(ExtractionError):
            validate_file_magic_bytes("binario.txt", b"texto\x00binario")

    def test_16_oversized_single_line_chunk_splitting(self):
        long_line = " ".join(f"Oración técnica número {i} sobre carga GPS Catapult." for i in range(60))
        self.assertGreater(len(long_line), 2000)

        parts = _split_long_line(long_line, max_chars=500, overlap_chars=80)
        self.assertGreater(len(parts), 3)
        for p in parts:
            self.assertLessEqual(len(p), 500)

        chunks = _split_text_lines(long_line, max_body_chars=500, overlap_chars=80)
        self.assertGreater(len(chunks), 3)
        self.assertIn("Oración técnica número 0", chunks[0])
        self.assertIn("Oración técnica número 59", chunks[-1])

    def test_17_vector_matrix_cache_lifecycle_and_blob_resilience(self):
        with tempfile.TemporaryDirectory(prefix="vcache_test_") as vdir:
            db_p = Path(vdir) / "vcache.db"
            init_database(db_p)
            conn = get_db_connection(db_p)
            try:
                conn.execute("INSERT INTO departments (id, name) VALUES ('scouting', 'Scouting')")
                conn.execute(
                    "INSERT INTO documents (doc_code, title, department_id, author, date, file_path, raw_content) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("VC-1", "Test Cache", "scouting", "Staff", "2026-11-10", "test.md", "Contenido"),
                )
                doc_id = conn.execute("SELECT id FROM documents WHERE doc_code = 'VC-1'").fetchone()[0]
                valid_vec = np.ones(384, dtype=np.float32)
                valid_vec /= np.linalg.norm(valid_vec)
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, title, department_id, content, embedding, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, 0, "C0", "scouting", "Chunk valido uno", valid_vec.tobytes(), "{}"),
                )
                # Inject corrupt-length BLOB to verify resilience
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, title, department_id, content, embedding, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, 1, "C1", "scouting", "Chunk corrupto", b"short_blob", "{}"),
                )
                conn.commit()

                cache = VectorMatrixCache()
                d1 = cache.get_or_load(conn, str(db_p), dim=384)
                self.assertEqual(len(d1["ids"]), 1)
                self.assertEqual(d1["departments"], ["scouting"])
                self.assertEqual(d1["matrix"].shape, (1, 384))

                # Subsequent call with unchanged DB signature must reuse cached matrix dict
                d2 = cache.get_or_load(conn, str(db_p), dim=384)
                self.assertIs(d1, d2)

                # Invalidating cache forces fresh rebuild
                cache.invalidate(str(db_p))
                d3 = cache.get_or_load(conn, str(db_p), dim=384)
                self.assertIsNot(d1, d3)
                self.assertEqual(len(d3["ids"]), 1)
            finally:
                conn.close()

    def test_18_fastapi_multiformat_upload_dedup_reindex_and_delete(self):
        with tempfile.TemporaryDirectory(prefix="upload_api_test_") as udir:
            u_root = Path(udir)
            u_docs = u_root / "raw_documents"
            u_uploads = u_root / "uploads"
            u_uploads.mkdir(parents=True, exist_ok=True)
            shutil.copytree(RAW_DOCS_DIR, u_docs)
            u_db = u_root / "upload_test.db"

            run_full_ingestion(db_path=u_db, docs_dir=u_docs, uploads_dir=u_uploads)
            test_pipeline = ClubRAGPipeline(db_path=u_db)

            pdf_bytes = _build_minimal_pdf([
                "Protocolo exclusivo Puerto Marina Operacion Albatros para concentracion hotelera."
            ])
            docx_bytes = _build_minimal_docx()
            txt_bytes = "Resumen nutricional especial sin azucares para concentracion previa.".encode("utf-8")

            with (
                patch.object(api_module, "pipeline", test_pipeline),
                patch.object(api_module, "RAW_DOCS_DIR", u_docs),
                patch.object(api_module, "UPLOADS_DIR", u_uploads),
            ):
                with TestClient(api_module.app) as client:
                    # 1. Valid multi-format batch upload (PDF + DOCX + TXT)
                    resp = client.post(
                        "/api/upload",
                        data={"department_id": "logistica_viajes", "author": "Charly Pereyra", "tags": "albatros, hotel"},
                        files=[
                            ("files", ("operacion_albatros.pdf", pdf_bytes, "application/pdf")),
                            ("files", ("informe_semifinal.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
                            ("files", ("menu_concentracion.txt", txt_bytes, "text/plain")),
                        ],
                    )
                    self.assertEqual(resp.status_code, 200)
                    payload = resp.json()
                    self.assertEqual(payload["successful"], 3)
                    self.assertEqual(payload["failed"], 0)
                    pdf_detail = next(d for d in payload["details"] if d["filename"] == "operacion_albatros.pdf")
                    pdf_doc_id = pdf_detail["document_id"]
                    pdf_doc_code = pdf_detail["doc_code"]

                    # Verify uploaded file exists on disk in temp uploads dir
                    saved_pdf = u_uploads / "logistica_viajes" / "operacion_albatros.pdf"
                    self.assertTrue(saved_pdf.exists())

                    # 2. Invalid extension & corrupt PDF magic bytes leave zero residue on disk
                    bad_resp = client.post(
                        "/api/upload",
                        data={"department_id": "scouting"},
                        files=[
                            ("files", ("malware.exe", b"MZ\x90\x00", "application/octet-stream")),
                            ("files", ("falso.pdf", b"NOT_A_PDF_PAYLOAD_000", "application/pdf")),
                        ],
                    )
                    self.assertEqual(bad_resp.status_code, 200)
                    bad_payload = bad_resp.json()
                    self.assertEqual(bad_payload["successful"], 0)
                    self.assertEqual(bad_payload["failed"], 2)
                    self.assertFalse((u_uploads / "scouting" / "malware.exe").exists())
                    self.assertFalse((u_uploads / "scouting" / "falso.pdf").exists())

                    # 3. Department-scoped SHA-256 deduplication reuses same doc_code and doc_id
                    dup_resp = client.post(
                        "/api/upload",
                        data={"department_id": "logistica_viajes"},
                        files=[("files", ("operacion_albatros.pdf", pdf_bytes, "application/pdf"))],
                    )
                    self.assertEqual(dup_resp.status_code, 200)
                    dup_item = dup_resp.json()["details"][0]
                    self.assertEqual(dup_item["doc_code"], pdf_doc_code)
                    self.assertEqual(dup_item["document_id"], pdf_doc_id)

                    # 4. Uploaded content is immediately queryable and exposes metadata
                    q_resp = client.post("/api/query", json={"query": "Operacion Albatros hotelera", "top_k": 3})
                    self.assertEqual(q_resp.status_code, 200)
                    self.assertIn("Albatros", q_resp.json()["answer"])

                    doc_meta_resp = client.get(f"/api/documents/{pdf_doc_id}")
                    self.assertEqual(doc_meta_resp.status_code, 200)
                    doc_info = doc_meta_resp.json()["document"]
                    self.assertEqual(doc_info["source_type"], "pdf")
                    self.assertEqual(doc_info["original_filename"], "operacion_albatros.pdf")
                    self.assertTrue(doc_info["content_hash"])

                    # 5. Reindex must preserve uploaded documents
                    reindex_resp = client.post("/api/reindex")
                    self.assertEqual(reindex_resp.status_code, 200)
                    self.assertGreaterEqual(reindex_resp.json()["details"]["documents_count"], 22)

                    # 6. Deleting uploaded document removes both DB record and disk file
                    del_resp = client.delete(f"/api/documents/{pdf_doc_id}")
                    self.assertEqual(del_resp.status_code, 200)
                    self.assertFalse(saved_pdf.exists())


if __name__ == "__main__":
    unittest.main()

