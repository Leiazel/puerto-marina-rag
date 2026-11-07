"""
Automated Test Suite for Club Atlético Puerto Marina RAG System.
Tests SQLite schema, FTS5 BM25, dense vector search, hybrid RRF, metadata filtering,
and grounded generation.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DB_PATH
from src.database import get_db_connection, init_database
from src.embeddings import get_embedding_engine
from src.retriever import HybridRetriever
from src.generator import GroundedGenerator
from src.rag_pipeline import ClubRAGPipeline


class TestClubRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = get_db_connection(DB_PATH)
        cls.embedder = get_embedding_engine()
        cls.retriever = HybridRetriever(DB_PATH)
        cls.generator = GroundedGenerator()
        cls.pipeline = ClubRAGPipeline(DB_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_01_database_tables_exist(self):
        """Verifies that all required relational and FTS5 tables exist."""
        cur = self.conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');").fetchall()
        table_names = [t["name"] for t in tables]
        
        required = ["departments", "players", "staff", "documents", "chunks", "chunks_fts", "query_logs"]
        for req in required:
            self.assertIn(req, table_names, f"La tabla {req} debería existir en la base de datos.")

    def test_02_seeded_data_integrity(self):
        """Checks that departments, players, staff, and documents are populated."""
        cur = self.conn.cursor()
        dept_count = cur.execute("SELECT count(*) FROM departments").fetchone()[0]
        player_count = cur.execute("SELECT count(*) FROM players").fetchone()[0]
        staff_count = cur.execute("SELECT count(*) FROM staff").fetchone()[0]
        doc_count = cur.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunk_count = cur.execute("SELECT count(*) FROM chunks").fetchone()[0]

        self.assertGreaterEqual(dept_count, 7)
        self.assertGreaterEqual(player_count, 14)
        self.assertGreaterEqual(staff_count, 10)
        self.assertGreaterEqual(doc_count, 15)
        self.assertGreaterEqual(chunk_count, 70)

    def test_03_embeddings_normalization_and_dim(self):
        """Checks embedding dimension and unit norm."""
        vec = self.embedder.embed_text("Plan táctico contra Deportivo Los Andes")
        self.assertEqual(len(vec), 384)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_04_fts5_bm25_lexical_search(self):
        """Checks that FTS5 BM25 retrieves exact medical / tactical keywords."""
        results = self.retriever.search_lexical(self.conn, "Lautaro Carrizo esguince", limit=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("Carrizo", top["content"])
        self.assertGreater(top["bm25_score"], 0)

    def test_05_dense_vector_semantic_search(self):
        """Checks semantic search on synonym concepts (e.g., cansancio muscular)."""
        results = self.retriever.search_vector(self.conn, "jugador fatigado con riesgo físico en isquiotibiales", limit=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertGreater(top["cosine_sim"], 0.25)

    def test_06_hybrid_rrf_retrieval(self):
        """Checks that hybrid search produces combined RRF scores and identifies modalities."""
        hybrid_res = self.retriever.hybrid_search("¿Cuál es el protocolo de suplementación con creatina y cafeína?", top_k=3)
        self.assertGreater(len(hybrid_res["results"]), 0)
        top_chunk = hybrid_res["results"][0]
        self.assertIn("rrf_score", top_chunk)
        self.assertGreater(top_chunk["rrf_score"], 0.0)

    def test_07_department_metadata_filtering(self):
        """Checks that filtering by department restricts results to only that area."""
        med_res = self.retriever.hybrid_search("nutrición y pesaje", department_filter="nutricion", top_k=4)
        for r in med_res["results"]:
            self.assertEqual(r["department_id"], "nutricion", "El resultado debería pertenecer exclusivamente al departamento filtrado.")

    def test_08_end_to_end_pipeline_and_citations(self):
        """Tests the full RAG pipeline and confirms citations and timings."""
        query = "¿En qué hotel y habitaciones se aloja la delegación para el partido de semifinal?"
        out = self.pipeline.ask(query, top_k=3)
        self.assertIn("Hotel Grand Bahía", out["answer"])
        self.assertGreater(len(out["citations"]), 0)
        self.assertIn("timings", out)
        self.assertGreater(out["timings"]["total_ms"], 0)

    def test_09_query_logging_audit(self):
        """Verifies that queries are stored in the SQLite query_logs table."""
        cur = self.conn.cursor()
        log_count_before = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.pipeline.ask("Consulta de prueba de auditoría")
        log_count_after = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.assertEqual(log_count_after, log_count_before + 1)


if __name__ == "__main__":
    unittest.main()
