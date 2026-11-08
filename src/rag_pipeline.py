"""
End-to-end RAG Pipeline orchestrator for Club Atlético Puerto Marina (v2.0).
Coordinates hybrid retrieval, grounded generation, and SQLite audit logging.
"""
import time
from typing import Dict, Any, Optional

from src.config import DB_PATH, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA, USE_EXTERNAL_LLM, LLM_MODEL, LLM_API_KEY
from src.database import get_db_connection
from src.retriever import HybridRetriever
from src.generator import GroundedGenerator


class ClubRAGPipeline:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.retriever = HybridRetriever(db_path=db_path)
        self.generator = GroundedGenerator()

    def ask(
        self,
        query: str,
        department_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_HYBRID_ALPHA
    ) -> Dict[str, Any]:
        """Executes end-to-end Hybrid RAG + Grounded Synthesis + Audit Logging."""
        pipeline_start = time.perf_counter()

        retrieval_output = self.retriever.hybrid_search(
            query=query,
            department_filter=department_filter,
            top_k=top_k,
            alpha=alpha
        )

        retrieval_time_ms = retrieval_output["retrieval_ms"]
        results = retrieval_output["results"]

        gen_start = time.perf_counter()
        gen_output = self.generator.generate_grounded_answer(query, results)
        gen_time_ms = round((time.perf_counter() - gen_start) * 1000.0, 2)

        total_time_ms = round((time.perf_counter() - pipeline_start) * 1000.0, 2)

        top_sim = results[0]["cosine_sim"] if results else 0.0
        self._log_query(
            query=query,
            dept_filter=department_filter,
            chunk_count=len(results),
            top_sim=top_sim,
            response=gen_output["answer"],
            retrieval_ms=retrieval_time_ms,
            total_ms=total_time_ms
        )

        return {
            "query": query,
            "department_filter": department_filter,
            "answer": gen_output["answer"],
            "citations": gen_output["citations"],
            "confidence": gen_output["confidence"],
            "model_used": gen_output["model_used"],
            "retrieved_chunks": results,
            "timings": {
                "retrieval_ms": retrieval_time_ms,
                "generation_ms": gen_time_ms,
                "total_ms": total_time_ms
            }
        }

    def _log_query(
        self,
        query: str,
        dept_filter: Optional[str],
        chunk_count: int,
        top_sim: float,
        response: str,
        retrieval_ms: float,
        total_ms: float
    ) -> None:
        try:
            conn = get_db_connection(self.db_path)
            conn.execute("""
                INSERT INTO query_logs (
                    query_text, department_filter, retrieved_chunks_count,
                    top_similarity, response_text, retrieval_ms, total_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (query, dept_filter, chunk_count, top_sim, response[:1000], retrieval_ms, total_ms))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: could not write to query_logs ({e})")

    def get_stats(self) -> Dict[str, Any]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()

        docs_count = cur.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks_count = cur.execute("SELECT count(*) FROM chunks").fetchone()[0]
        players_count = cur.execute("SELECT count(*) FROM players").fetchone()[0]
        staff_count = cur.execute("SELECT count(*) FROM staff").fetchone()[0]
        dept_count = cur.execute("SELECT count(*) FROM departments").fetchone()[0]
        queries_count = cur.execute("SELECT count(*) FROM query_logs").fetchone()[0]

        avg_latency_row = cur.execute("SELECT avg(total_ms) FROM query_logs").fetchone()[0]
        avg_latency = round(avg_latency_row, 2) if avg_latency_row else 0.0

        dept_docs = cur.execute("""
            SELECT d.id, d.name, count(doc.id) as doc_count
            FROM departments d
            LEFT JOIN documents doc ON doc.department_id = d.id
            GROUP BY d.id
        """).fetchall()

        conn.close()

        self.generator._refresh_config_if_needed()
        llm_active = self.generator.use_llm and bool(self.generator.api_key)
        return {
            "documents_count": docs_count,
            "chunks_count": chunks_count,
            "players_count": players_count,
            "staff_count": staff_count,
            "departments_count": dept_count,
            "total_queries_logged": queries_count,
            "average_query_latency_ms": avg_latency,
            "synthesis_mode": f"LLM ({self.generator.model})" if llm_active else "Local Grounded Synthesizer v2",
            "llm_api_base": self.generator.api_base if llm_active else None,
            "departments": [dict(r) for r in dept_docs]
        }
