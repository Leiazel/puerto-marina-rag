"""
Hybrid Retriever module combining SQLite FTS5 (BM25) and Dense Vector Cosine Similarity
with Reciprocal Rank Fusion (RRF) and metadata filtering.
"""
import time
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

from src.config import DB_PATH, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA, RRF_K_CONSTANT
from src.database import get_db_connection
from src.embeddings import get_embedding_engine


class HybridRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.embedder = get_embedding_engine()

    def _clean_fts_query(self, query: str) -> str:
        """
        Cleans and formats a query string for SQLite FTS5.
        Removes special symbols and adds prefix matching for Spanish words.
        """
        # Remove characters that cause FTS5 syntax errors: * " ( ) : ^
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in query)
        words = cleaned.split()
        if not words:
            return ""
        # Build an OR-based query with prefix wildcards for flexible matching
        # e.g. "lesion Carrizo" -> "lesion* OR Carrizo*"
        fts_terms = [f'"{w}"*' for w in words if len(w) > 2]
        if not fts_terms:
            fts_terms = [f'"{w}"' for w in words]
        return " OR ".join(fts_terms)

    def search_lexical(
        self,
        conn: sqlite3.Connection,
        query: str,
        department_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Performs full-text search with BM25 ranking on chunks_fts.
        """
        fts_q = self._clean_fts_query(query)
        if not fts_q:
            return []

        cur = conn.cursor()
        try:
            if department_filter:
                sql = """
                    SELECT 
                        c.id as chunk_id,
                        c.document_id,
                        c.title,
                        c.department_id,
                        c.content,
                        c.metadata_json,
                        bm25(chunks_fts) as bm25_rank,
                        snippet(chunks_fts, 3, '«', '»', '...', 20) as match_snippet
                    FROM chunks_fts
                    JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
                    WHERE chunks_fts MATCH ? AND c.department_id = ?
                    ORDER BY bm25_rank ASC
                    LIMIT ?;
                """
                cur.execute(sql, (fts_q, department_filter, limit))
            else:
                sql = """
                    SELECT 
                        c.id as chunk_id,
                        c.document_id,
                        c.title,
                        c.department_id,
                        c.content,
                        c.metadata_json,
                        bm25(chunks_fts) as bm25_rank,
                        snippet(chunks_fts, 3, '«', '»', '...', 20) as match_snippet
                    FROM chunks_fts
                    JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25_rank ASC
                    LIMIT ?;
                """
                cur.execute(sql, (fts_q, limit))
            
            rows = cur.fetchall()
            results = []
            for idx, r in enumerate(rows):
                # SQLite bm25 returns lower numbers for better matches (typically negative)
                raw_bm25 = float(r["bm25_rank"])
                # Normalize BM25 into a positive score
                score = abs(raw_bm25) if raw_bm25 < 0 else (1.0 / (1.0 + raw_bm25))
                results.append({
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "title": r["title"],
                    "department_id": r["department_id"],
                    "content": r["content"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "match_snippet": r["match_snippet"],
                    "bm25_score": round(score, 4),
                    "lexical_rank": idx + 1
                })
            return results
        except sqlite3.OperationalError as e:
            # Fallback if FTS syntax edge case
            return []

    def search_vector(
        self,
        conn: sqlite3.Connection,
        query: str,
        department_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Performs dense semantic vector search via BLAS dot product over stored BLOBs.
        """
        query_vec = self.embedder.embed_text(query)
        cur = conn.cursor()

        if department_filter:
            cur.execute("""
                SELECT id, document_id, title, department_id, content, embedding, metadata_json
                FROM chunks
                WHERE department_id = ?;
            """, (department_filter,))
        else:
            cur.execute("""
                SELECT id, document_id, title, department_id, content, embedding, metadata_json
                FROM chunks;
            """)

        rows = cur.fetchall()
        if not rows:
            return []

        chunk_ids = []
        doc_ids = []
        titles = []
        depts = []
        contents = []
        metas = []
        blobs = []

        for r in rows:
            chunk_ids.append(r["id"])
            doc_ids.append(r["document_id"])
            titles.append(r["title"])
            depts.append(r["department_id"])
            contents.append(r["content"])
            metas.append(r["metadata_json"])
            blobs.append(r["embedding"])

        # Pack into numpy matrix
        dim = len(query_vec)
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, dim)

        # Vectorized cosine similarity
        similarities = self.embedder.compute_cosine_similarity(query_vec, matrix)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for rank, idx in enumerate(top_indices):
            sim = float(similarities[idx])
            results.append({
                "chunk_id": chunk_ids[idx],
                "document_id": doc_ids[idx],
                "title": titles[idx],
                "department_id": depts[idx],
                "content": contents[idx],
                "metadata": json.loads(metas[idx] or "{}"),
                "cosine_sim": round(sim, 4),
                "vector_rank": rank + 1
            })
        return results

    def hybrid_search(
        self,
        query: str,
        department_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_HYBRID_ALPHA
    ) -> Dict[str, Any]:
        """
        Performs hybrid retrieval using Reciprocal Rank Fusion (RRF):
        RRF(d) = alpha * (1 / (K + rank_lexical)) + (1 - alpha) * (1 / (K + rank_vector))
        """
        start_time = time.perf_counter()
        conn = get_db_connection(self.db_path)

        # 1. Retrieve candidates from both sources
        lexical_candidates = self.search_lexical(conn, query, department_filter, limit=30)
        vector_candidates = self.search_vector(conn, query, department_filter, limit=30)

        # 2. Map and fuse with Reciprocal Rank Fusion
        fused_scores: Dict[int, Dict[str, Any]] = {}
        K = RRF_K_CONSTANT

        # Index lexical results
        for item in lexical_candidates:
            cid = item["chunk_id"]
            rank = item["lexical_rank"]
            rrf_val = alpha * (1.0 / (K + rank))
            fused_scores[cid] = {
                "chunk_id": cid,
                "document_id": item["document_id"],
                "title": item["title"],
                "department_id": item["department_id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "match_snippet": item.get("match_snippet", ""),
                "bm25_score": item["bm25_score"],
                "lexical_rank": rank,
                "vector_rank": None,
                "cosine_sim": 0.0,
                "rrf_score": rrf_val,
                "modalities": ["lexical"]
            }

        # Combine with vector results
        for item in vector_candidates:
            cid = item["chunk_id"]
            rank = item["vector_rank"]
            vector_rrf = (1.0 - alpha) * (1.0 / (K + rank))

            if cid in fused_scores:
                fused_scores[cid]["rrf_score"] += vector_rrf
                fused_scores[cid]["vector_rank"] = rank
                fused_scores[cid]["cosine_sim"] = item["cosine_sim"]
                fused_scores[cid]["modalities"].append("dense_vector")
            else:
                fused_scores[cid] = {
                    "chunk_id": cid,
                    "document_id": item["document_id"],
                    "title": item["title"],
                    "department_id": item["department_id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "match_snippet": "",
                    "bm25_score": 0.0,
                    "lexical_rank": None,
                    "vector_rank": rank,
                    "cosine_sim": item["cosine_sim"],
                    "rrf_score": vector_rrf,
                    "modalities": ["dense_vector"]
                }

        conn.close()

        # 3. Sort by combined RRF score descending
        sorted_chunks = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_results = sorted_chunks[:top_k]

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "department_filter": department_filter,
            "total_candidates": len(fused_scores),
            "top_k": top_k,
            "alpha": alpha,
            "retrieval_ms": round(duration_ms, 2),
            "results": top_results
        }
