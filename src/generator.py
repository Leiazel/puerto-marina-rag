"""
Generator module for Grounded Response Generation and Citations (v2.0).
Features:
1. Upgraded Local Extractive Synthesizer (preserves table headers, scores lines by relevance,
   and computes dynamic confidence based on cosine similarity and modality overlap).
2. Universal OpenAI-compatible LLM Adapter (ready for Ollama, Groq, Gemini, or OpenAI).
"""
import json
import re
import unicodedata
import urllib.request
from typing import List, Dict, Any

from src.config import (
    USE_EXTERNAL_LLM,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SEC,
    SPANISH_STOPWORDS
)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


class GroundedGenerator:
    def __init__(self):
        self.use_llm = USE_EXTERNAL_LLM or bool(LLM_API_KEY)
        self.api_base = LLM_API_BASE
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.timeout = LLM_TIMEOUT_SEC

    def build_prompt_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Constructs formatted context block with clear citation IDs."""
        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            doc_code = chunk.get("metadata", {}).get("doc_code", f"DOC-{idx}")
            title = chunk.get("title", "Documento")
            dept = chunk.get("department_id", "general")
            date = chunk.get("metadata", {}).get("date", "2026-11")
            content = chunk.get("content", "").strip()

            block = (
                f"--- [FUENTE #{idx} | REF: {doc_code} | ÁREA: {dept} | FECHA: {date}] ---\n"
                f"Título: {title}\n"
                f"{content}\n"
            )
            context_blocks.append(block)
        return "\n".join(context_blocks)

    def _compute_dynamic_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Computes a realistic confidence score in [0.0, 0.99] based on:
        - Top chunk cosine similarity
        - Whether top chunks matched BOTH lexical (BM25) and dense_vector modalities
        """
        if not chunks:
            return 0.0
        top = chunks[0]
        cos_sim = max(0.0, float(top.get("cosine_sim", 0.0)))
        both_modalities = len(top.get("modalities", [])) >= 2

        # Base confidence scaled from cosine similarity (typical MiniLM range 0.35 - 0.85)
        base = min(0.92, max(0.45, cos_sim * 1.15))
        if both_modalities:
            base = min(0.98, base + 0.08)
        return round(base, 2)

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates a structured, evidence-grounded answer."""
        if not retrieved_chunks:
            return {
                "answer": "No se encontraron documentos o registros relevantes en la base de datos del club para responder a esta consulta.",
                "citations": [],
                "confidence": 0.0,
                "model_used": "none"
            }

        if self.use_llm and self.api_key:
            try:
                return self._call_external_llm(query, retrieved_chunks)
            except Exception as e:
                print(f"Warning: external LLM call failed ({e}), falling back to local grounded synthesis.")

        return self._local_grounded_synthesis(query, retrieved_chunks)

    def _local_grounded_synthesis(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Upgraded Local Grounded Synthesizer (v2.0):
        - Scores lines by normalized keyword overlap with the query.
        - Automatically preserves Markdown table headers when extracting matching table rows.
        """
        citations = []
        for idx, ch in enumerate(chunks, 1):
            doc_code = ch.get("metadata", {}).get("doc_code", f"DOC-{idx}")
            citations.append({
                "source_num": idx,
                "ref_code": doc_code,
                "title": ch.get("title", ""),
                "department": ch.get("department_id", ""),
                "rrf_score": round(ch.get("rrf_score", 0.0), 5),
                "modalities": ch.get("modalities", [])
            })

        top_chunk = chunks[0]
        top_dept = top_chunk.get("department_id", "General")
        top_ref = top_chunk.get("metadata", {}).get("doc_code", "DOC-01")

        # Normalize query terms (without accents and without stopwords)
        norm_query = _strip_accents(query.lower())
        query_terms = [
            w for w in re.findall(r'\b\w{3,}\b', norm_query)
            if w not in SPANISH_STOPWORDS
        ]

        paragraphs = []
        bullet_items = []
        extracted_tables: List[List[str]] = []
        seen_lines = set()

        for chunk_rank, chunk in enumerate(chunks):
            raw_lines = chunk["content"].split("\n")
            current_table_header = []
            current_table_matches = []

            for line in raw_lines:
                clean_l = line.strip()
                if not clean_l or clean_l.startswith("DOCUMENTO:") or clean_l.startswith("TAGS:"):
                    continue

                norm_l = _strip_accents(clean_l.lower())
                term_hits = sum(1 for w in query_terms if w in norm_l)

                # Handle Markdown Table rows while preserving headers
                if clean_l.startswith("|"):
                    if "---" in clean_l:
                        if len(current_table_header) == 1:
                            current_table_header.append(clean_l)
                        continue
                    if not current_table_header:
                        current_table_header = [clean_l]
                        continue
                    # It is a data row inside a table
                    if term_hits > 0 or (chunk_rank == 0 and len(current_table_matches) < 5):
                        if clean_l not in seen_lines:
                            seen_lines.add(clean_l)
                            current_table_matches.append((term_hits, clean_l))
                    continue

                # Non-table lines
                if clean_l.startswith("###") or clean_l.startswith("##"):
                    if chunk_rank == 0 and len(paragraphs) < 2 and clean_l not in seen_lines:
                        seen_lines.add(clean_l)
                        paragraphs.append(f"**{clean_l.replace('#', '').strip()}**")
                elif clean_l.startswith("- ") or clean_l.startswith("* "):
                    if term_hits > 0 or chunk_rank == 0:
                        if clean_l not in seen_lines:
                            seen_lines.add(clean_l)
                            bullet_items.append((term_hits + (2 if chunk_rank == 0 else 0), clean_l))
                else:
                    if term_hits > 0 or (chunk_rank == 0 and len(paragraphs) < 3):
                        if clean_l not in seen_lines:
                            seen_lines.add(clean_l)
                            paragraphs.append(clean_l)

            if current_table_header and current_table_matches:
                # Sort matched table rows by relevance hits
                current_table_matches.sort(key=lambda x: x[0], reverse=True)
                best_rows = [r[1] for r in current_table_matches[:6]]
                if len(current_table_header) == 1:
                    cols = current_table_header[0].count("|") - 1
                    current_table_header.append("|" + "---|" * max(1, cols))
                extracted_tables.append(current_table_header + best_rows)

        # Sort bullets by relevance score descending
        bullet_items.sort(key=lambda x: x[0], reverse=True)
        top_bullets = [b[1] for b in bullet_items[:8]]

        dept_labels = {
            "servicio_medico": "Departamento Médico y Fisioterapia",
            "preparacion_fisica": "Área de Preparación Física y Monitoreo GPS",
            "cuerpo_tecnico": "Cuerpo Técnico y Análisis Táctico",
            "nutricion": "Departamento de Nutrición y Rendimiento",
            "scouting": "Dirección de Scouting e Inteligencia Deportiva",
            "logistica_viajes": "Gerencia de Operaciones y Logística",
            "disciplina_prensa": "Prensa, Comunicación y Reglamento Interno"
        }
        dept_name = dept_labels.get(top_dept, top_dept.replace("_", " ").title())

        lines_output = [
            "### Informe Oficial del Club Atlético Puerto Marina",
            f"**Área Responsable:** {dept_name}  \n**Referencia Principal:** `{top_ref}`\n",
            "#### Resumen Ejecutivo y Evidencia Documental:"
        ]

        if paragraphs:
            for p in paragraphs[:3]:
                lines_output.append(p)

        if top_bullets:
            lines_output.append("\n**Puntos clave verificados:**")
            lines_output.extend(top_bullets)

        if extracted_tables:
            lines_output.append("\n**Registros y mediciones en tablas oficiales:**")
            for tbl in extracted_tables[:2]:
                lines_output.append("\n" + "\n".join(tbl))

        lines_output.append("\n---")
        lines_output.append(f"**Fuentes documentales consultadas ({len(citations)}):**")
        for c in citations:
            lines_output.append(f"- **[Fuente #{c['source_num']}]** `{c['ref_code']}` - {c['title']} (*{c['department']}*)")

        confidence = self._compute_dynamic_confidence(chunks)

        return {
            "answer": "\n".join(lines_output),
            "citations": citations,
            "confidence": confidence,
            "model_used": "grounded_extractive_v2"
        }

    def _call_external_llm(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Universal OpenAI-compatible Chat Completions adapter.
        Works seamlessly with OpenAI, Groq, Ollama (/v1), and Google Gemini (/v1beta/openai).
        """
        context = self.build_prompt_context(chunks)
        system_prompt = (
            "Eres el Asistente de Inteligencia Operativa y RAG del Club Atlético Puerto Marina.\n"
            "Responde a la consulta de manera ejecutiva, rigurosa, estructurada en Markdown y "
            "basada ESTRICTAMENTE en las fuentes provistas.\n"
            "Cita siempre las fuentes entre corchetes usando su código [REF: CODIGO]. "
            "No inventes ni extrapoles datos que no figuren explícitamente en el contexto."
        )
        user_prompt = (
            f"CONTEXTO DOCUMENTAL RECUPERADO:\n{context}\n\n"
            f"CONSULTA DEL STAFF: {query}\n\n"
            "RESPUESTA FUNDAMENTADA:"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            data=json.dumps(payload).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer_text = data["choices"][0]["message"]["content"]

        citations = [
            {
                "source_num": idx,
                "ref_code": ch.get("metadata", {}).get("doc_code", f"DOC-{idx}"),
                "title": ch.get("title", ""),
                "department": ch.get("department_id", ""),
                "rrf_score": round(ch.get("rrf_score", 0.0), 5),
                "modalities": ch.get("modalities", [])
            }
            for idx, ch in enumerate(chunks, 1)
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": self._compute_dynamic_confidence(chunks),
            "model_used": f"llm:{self.model}"
        }
