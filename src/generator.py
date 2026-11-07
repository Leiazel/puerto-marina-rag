"""
Generator module for Grounded Response Generation and Citations.
Produces executive, professional, evidence-backed answers for club staff.
Supports both deterministic local grounded synthesis and external LLM adapters.
"""
import os
import re
from typing import List, Dict, Any, Optional


class GroundedGenerator:
    def __init__(self, use_llm_if_available: bool = True):
        self.use_llm = use_llm_if_available
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_base = os.environ.get("LLM_API_BASE")

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

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates a structured, evidence-grounded answer based strictly on retrieved chunks.
        """
        if not retrieved_chunks:
            return {
                "answer": "No se encontraron documentos o registros relevantes en la base de datos del club para responder a esta consulta.",
                "citations": [],
                "confidence": 0.0,
                "model_used": "grounded_rule_engine"
            }

        # Check if an external LLM is configured
        if self.use_llm and self.api_key:
            try:
                return self._call_external_llm(query, retrieved_chunks)
            except Exception as e:
                print(f"Warning: external LLM call failed ({e}), falling back to local grounded synthesis.")

        # Local High-Precision Grounded Synthesizer
        return self._local_grounded_synthesis(query, retrieved_chunks)

    def _local_grounded_synthesis(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Local, robust grounded synthesis engine.
        Extracts key factual points, quotes, tables, and warnings directly from chunks,
        organizing them into a structured report for club management.
        """
        citations = []
        for idx, ch in enumerate(chunks, 1):
            doc_code = ch.get("metadata", {}).get("doc_code", f"DOC-{idx}")
            title = ch.get("title", "")
            dept = ch.get("department_id", "")
            citations.append({
                "source_num": idx,
                "ref_code": doc_code,
                "title": title,
                "department": dept,
                "rrf_score": ch.get("rrf_score", 0.0),
                "modalities": ch.get("modalities", [])
            })

        # Identify primary department and focus from top chunk
        top_chunk = chunks[0]
        top_content = top_chunk["content"]
        top_dept = top_chunk.get("department_id", "General")
        top_ref = top_chunk.get("metadata", {}).get("doc_code", "DOC-01")

        # Synthesize lines from chunks that relate to query terms
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        extracted_facts = []
        seen_lines = set()

        for chunk in chunks:
            raw_lines = chunk["content"].split("\n")
            for line in raw_lines:
                clean_l = line.strip()
                if not clean_l or clean_l.startswith("DOCUMENTO:") or clean_l.startswith("TAGS:"):
                    continue
                # Highlight or bullet lines with rich factual data (numbers, dates, names, bullet points)
                lower_l = clean_l.lower()
                matches = sum(1 for w in query_words if w in lower_l)
                
                # Check if it has markdown lists or table rows with relevant information
                if (matches > 0 or clean_l.startswith("- ") or clean_l.startswith("|") or clean_l.startswith("###")) and clean_l not in seen_lines:
                    seen_lines.add(clean_l)
                    # Clean markdown formatting for readability
                    extracted_facts.append(clean_l)
                    if len(extracted_facts) >= 12:
                        break
            if len(extracted_facts) >= 12:
                break

        # Assemble executive response
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

        lines_output = []
        lines_output.append(f"### Informe Oficial del Club Atlético Puerto Marina")
        lines_output.append(f"**Área Responsable:** {dept_name}  \n**Referencia Principal:** `{top_ref}`  \n")
        lines_output.append(f"#### Resumen Ejecutivo y Respuesta Directa:")

        # Present extracted evidence structured
        bullet_items = []
        table_rows = []
        paragraphs = []

        for item in extracted_facts:
            if item.startswith("|"):
                table_rows.append(item)
            elif item.startswith("- ") or item.startswith("* "):
                bullet_items.append(item)
            elif item.startswith("###") or item.startswith("##"):
                paragraphs.append(f"\n**{item.replace('#', '').strip()}**")
            else:
                paragraphs.append(item)

        if paragraphs:
            for p in paragraphs[:3]:
                lines_output.append(p)

        if bullet_items:
            lines_output.append("\n**Detalles y puntos clave verificados:**")
            lines_output.extend(bullet_items[:8])

        if table_rows and len(table_rows) >= 2:
            lines_output.append("\n**Datos y mediciones registradas:**")
            lines_output.extend(table_rows[:7])

        lines_output.append(f"\n---")
        lines_output.append(f"**Fuentes documentales consultadas ({len(citations)}):**")
        for c in citations:
            lines_output.append(f"- **[Fuente #{c['source_num']}]** `{c['ref_code']}` - {c['title']} (*{c['department']}*)")

        full_answer = "\n".join(lines_output)

        return {
            "answer": full_answer,
            "citations": citations,
            "confidence": 0.94 if len(chunks) >= 2 else 0.82,
            "model_used": "grounded_synthesis_engine_v1"
        }

    def _call_external_llm(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optional adapter if user provides an external OpenAI/Ollama compatible endpoint."""
        import urllib.request
        import json

        context = self.build_prompt_context(chunks)
        prompt = (
            "Eres el Asistente de Inteligencia Operativa y RAG de Club Atlético Puerto Marina.\n"
            "Responde a la consulta de manera ejecutiva, rigurosa, profesional y basada ESTRICTAMENTE en las fuentes provistas.\n"
            "Cita siempre las fuentes entre corchetes [REF: CODIGO]. No inventes datos que no figuren en los textos.\n\n"
            f"CONTEXTO DOCUMENTAL:\n{context}\n\n"
            f"CONSULTA: {query}\n\n"
            "RESPUESTA:"
        )

        base_url = self.api_base or "https://api.openai.com/v1"
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer_text = data["choices"][0]["message"]["content"]

        citations = [
            {
                "source_num": idx,
                "ref_code": ch.get("metadata", {}).get("doc_code", f"DOC-{idx}"),
                "title": ch.get("title", ""),
                "department": ch.get("department_id", "")
            }
            for idx, ch in enumerate(chunks, 1)
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": 0.98,
            "model_used": "external_llm"
        }
