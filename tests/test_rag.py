"""
Automated Test Suite for Club Atlético Puerto Marina RAG System (v2.2).
All tests run against an isolated temporary SQLite database and document copy
so production data/database/club_knowledge.db is never mutated by test runs.
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
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import src.api as api_module
from src.config import BASE_DIR, RAW_DOCS_DIR, _resolve_llm_config
from src.database import get_db_connection
from src.embeddings import get_embedding_engine
from src.generator import GroundedGenerator, _clean_llm_output
from src.ingestion import _portable_file_path, run_full_ingestion
from src.rag_pipeline import ClubRAGPipeline
from src.retriever import HybridRetriever


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
        shutil.copytree(RAW_DOCS_DIR, cls.tmp_docs_dir)
        cls.tmp_db_path = cls.tmp_root / "test_club_knowledge.db"

        cls.ingest_summary = run_full_ingestion(db_path=cls.tmp_db_path, docs_dir=cls.tmp_docs_dir)
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
        test_pipeline = ClubRAGPipeline(db_path=self.tmp_db_path)

        with patch.object(api_module, "pipeline", test_pipeline), patch.object(api_module, "RAW_DOCS_DIR", self.tmp_docs_dir):
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


if __name__ == "__main__":
    unittest.main()
