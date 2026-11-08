"""
Automated Test Suite for Club Atlético Puerto Marina RAG System (v2.0).
"""
import unittest
import numpy as np
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DB_PATH
from src.database import get_db_connection
from src.embeddings import get_embedding_engine
from src.retriever import HybridRetriever
from src.generator import GroundedGenerator
from src.rag_pipeline import ClubRAGPipeline


class TestClubRAGSystemV2(unittest.TestCase):
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
        cur = self.conn.cursor()
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');").fetchall()
        table_names = [t["name"] for t in tables]
        for req in ["departments", "players", "staff", "documents", "chunks", "chunks_fts", "query_logs"]:
            self.assertIn(req, table_names)

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
        self.pipeline.ask("Consulta de auditoría de sistema")
        after = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]
        self.assertEqual(after, before + 1)

    def test_10_relational_sql_bridge_in_rag(self):
        """Verifies that SQL table rows (staff & players) are indexed and retrievable via RAG."""
        staff_out = self.pipeline.ask("¿Cuál es el correo electrónico y teléfono de la nutricionista Sofía Carrizo?")
        self.assertIn("scarrizo@puertomarina.fc", staff_out["answer"])
        player_out = self.pipeline.ask("¿Cuánto mide y cuánto pesa el arquero suplente Lucas Barreto?")
        self.assertIn("1.91", player_out["answer"])


if __name__ == "__main__":
    unittest.main()
