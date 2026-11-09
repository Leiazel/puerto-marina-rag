"""
Hybrid Retriever module (v2.0) combining:
- SQLite FTS5 (BM25) with Spanish Stopword filtering and rowid integer joins.
- Two-Phase Dense Vector Cosine Similarity (loads only IDs & BLOBs first, then fetches text for Top-K).
- Reciprocal Rank Fusion (RRF) and metadata filtering.
"""
import time
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional

from src.config import (
    DB_PATH,
    DEFAULT_TOP_K,
    DEFAULT_HYBRID_ALPHA,
    RRF_K_CONSTANT,
    SPANISH_STOPWORDS
)
from src.database import get_db_connection
from src.embeddings import get_embedding_engine


import threading
import unicodedata

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


class VectorMatrixCache:
    """
    In-memory dense vector matrix cache (v3.0).
    Eliminates redundant SQLite BLOB reads and numpy buffer deserialization on every query.
    Auto-invalidates when chunk count or max ID changes.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_or_load(self, conn: sqlite3.Connection, db_key: str, dim: int) -> Dict[str, Any]:
        cur = conn.cursor()
        sig_row = cur.execute("SELECT count(id), coalesce(max(id), 0) FROM chunks;").fetchone()
        sig = (sig_row[0], sig_row[1]) if sig_row else (0, 0)

        with self._lock:
            cached = self._cache.get(db_key)
            if cached and cached["sig"] == sig:
                return cached

            # Rebuild cache
            cur.execute("SELECT id, department_id, embedding FROM chunks ORDER BY id ASC;")
            rows = cur.fetchall()
            if not rows:
                empty = {
                    "sig": sig,
                    "ids": np.array([], dtype=np.int64),
                    "departments": [],
                    "matrix": np.empty((0, dim), dtype=np.float32)
                }
                self._cache[db_key] = empty
                return empty

            chunk_ids = np.array([r["id"] for r in rows], dtype=np.int64)
            dept_ids = [r["department_id"] for r in rows]
            blobs = [r["embedding"] for r in rows]
            matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, dim)

            data = {
                "sig": sig,
                "ids": chunk_ids,
                "departments": dept_ids,
                "matrix": matrix
            }
            self._cache[db_key] = data
            return data

    def invalidate(self, db_key: Optional[str] = None):
        with self._lock:
            if db_key:
                self._cache.pop(db_key, None)
            else:
                self._cache.clear()


GLOBAL_VECTOR_CACHE = VectorMatrixCache()


class HybridRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.embedder = get_embedding_engine()

    def invalidate_cache(self):
        GLOBAL_VECTOR_CACHE.invalidate(str(self.db_path))

    def _clean_fts_query(self, query: str) -> str:
        """
        Cleans a natural language Spanish query for SQLite FTS5.
        Strips accents, punctuation, removes Spanish stopwords, and applies prefix wildcards.
        """
        no_accents = _strip_accents(query.lower())
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in no_accents)
        raw_words = [w for w in cleaned.split() if len(w) >= 2 or w.isdigit()]
        if not raw_words:
            return ""

        # Filter out Spanish stopwords so BM25 focuses strictly on domain terms and numbers
        meaningful_words = [
            w for w in raw_words
            if (len(w) > 2 or w.isdigit()) and w not in SPANISH_STOPWORDS
        ]
        if not meaningful_words:
            meaningful_words = raw_words

        fts_terms = [f'"{w}"*' for w in meaningful_words]
        return " OR ".join(fts_terms)

    def search_lexical(
        self,
        conn: sqlite3.Connection,
        query: str,
        department_filter: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Performs full-text search with BM25 ranking on chunks_fts using direct rowid join.
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
                        snippet(chunks_fts, 3, '«', '»', '...', 22) as match_snippet
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
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
                        snippet(chunks_fts, 3, '«', '»', '...', 22) as match_snippet
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25_rank ASC
                    LIMIT ?;
                """
                cur.execute(sql, (fts_q, limit))

            rows = cur.fetchall()
            results = []
            for idx, r in enumerate(rows):
                raw_bm25 = float(r["bm25_rank"])
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
        except sqlite3.OperationalError:
            return []

    def search_vector(
        self,
        conn: sqlite3.Connection,
        query: str,
        department_filter: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Two-Phase Dense Vector Search with In-Memory Matrix Cache (v3.0):
        Phase 1: Zero-disk BLAS dot product against cached normalized float32 matrix.
        Phase 2: Fetch text & metadata ONLY for the winning Top-K IDs from SQLite.
        """
        query_vec = self.embedder.embed_text(query)
        dim = len(query_vec)
        db_key = str(self.db_path)
        cache_data = GLOBAL_VECTOR_CACHE.get_or_load(conn, db_key, dim)

        if len(cache_data["ids"]) == 0:
            return []

        if department_filter:
            mask = [d == department_filter for d in cache_data["departments"]]
            if not any(mask):
                return []
            sub_indices = np.where(mask)[0]
            sub_matrix = cache_data["matrix"][sub_indices]
            sub_ids = cache_data["ids"][sub_indices]
            similarities = self.embedder.compute_cosine_similarity(query_vec, sub_matrix)
            top_sub_idx = np.argsort(similarities)[::-1][:limit]
            winning_ids = [int(sub_ids[i]) for i in top_sub_idx]
            sim_by_id = {int(sub_ids[i]): float(similarities[i]) for i in top_sub_idx}
        else:
            similarities = self.embedder.compute_cosine_similarity(query_vec, cache_data["matrix"])
            top_indices = np.argsort(similarities)[::-1][:limit]
            winning_ids = [int(cache_data["ids"][i]) for i in top_indices]
            sim_by_id = {int(cache_data["ids"][i]): float(similarities[i]) for i in top_indices}

        if not winning_ids:
            return []

        # Phase 2: Fetch text & metadata only for the winning IDs
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in winning_ids)
        cur.execute(f"""
            SELECT id, document_id, title, department_id, content, metadata_json
            FROM chunks
            WHERE id IN ({placeholders});
        """, winning_ids)

        detail_map = {r["id"]: r for r in cur.fetchall()}

        results = []
        for rank, cid in enumerate(winning_ids, 1):
            r = detail_map.get(cid)
            if not r:
                continue
            results.append({
                "chunk_id": cid,
                "document_id": r["document_id"],
                "title": r["title"],
                "department_id": r["department_id"],
                "content": r["content"],
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "cosine_sim": round(sim_by_id[cid], 4),
                "vector_rank": rank
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
        try:
            lexical_candidates = self.search_lexical(conn, query, department_filter, limit=25)
            vector_candidates = self.search_vector(conn, query, department_filter, limit=25)
        finally:
            conn.close()

        fused_scores: Dict[int, Dict[str, Any]] = {}
        K = RRF_K_CONSTANT

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
