"""
Ingestion pipeline module for Club Atlético Puerto Marina (v2.0).
- Parses markdown frontmatter and chunks documents with contextual prefixes.
- Preserves Markdown table headers across chunks.
- Automatically indexes relational SQL tables (players, staff, departments) into RAG chunks.
"""
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

from src.config import RAW_DOCS_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, DB_PATH, UPLOADS_DIR
from src.database import get_db_connection, init_database
from src.embeddings import get_embedding_engine
from src.extractors import extract_document_content, ExtractionError
from src.retriever import GLOBAL_VECTOR_CACHE

INITIAL_DEPARTMENTS = [
    {
        "id": "servicio_medico",
        "name": "Departamento Médico y Fisioterapia",
        "description": "Traumatología deportiva, kinesiología, rehabilitación y readaptación funcional (RTP).",
        "manager": "Dr. Federico Arismendi"
    },
    {
        "id": "preparacion_fisica",
        "name": "Preparación Física y Monitoreo GPS",
        "description": "Cargas de entrenamiento, métricas Catapult/Wimu, ratio ACWR, fatiga neuromuscular y test CMJ.",
        "manager": "Prof. Germán Silveira"
    },
    {
        "id": "cuerpo_tecnico",
        "name": "Cuerpo Técnico y Análisis Táctico",
        "description": "Estrategia, modelo de juego, microciclos, scouting de rivales y balón parado.",
        "manager": "Javier Méndez (DT)"
    },
    {
        "id": "nutricion",
        "name": "Nutrición y Rendimiento Metabólico",
        "description": "Composición corporal ISAK, carga de glucógeno, hidratación, suplementación y dietas especiales.",
        "manager": "Lic. Sofía Carrizo"
    },
    {
        "id": "scouting",
        "name": "Scouting e Inteligencia Deportiva",
        "description": "Seguimiento de futbolistas, fichajes, análisis de métricas avanzadas (xG, xA, pases progresivos).",
        "manager": "Lucas Benítez"
    },
    {
        "id": "logistica_viajes",
        "name": "Operaciones, Seguridad y Logística",
        "description": "Vuelos chárter, concentración en hoteles, asignación de habitaciones y utilería.",
        "manager": "Carlos 'Charly' Pereyra"
    },
    {
        "id": "disciplina_prensa",
        "name": "Prensa, Comunicación y Reglamento Interno",
        "description": "Ruedas de prensa, medios oficiales, convivencia, pesaje obligatorio y régimen de multas.",
        "manager": "Mariana Gómez Solís"
    }
]

INITIAL_PLAYERS = [
    {"dorsal": 1, "name": "Mateo Rossi", "nickname": "Araña", "position": "Arquero Titular", "age": 29, "height_m": 1.89, "weight_kg": 84.2, "notes": "Especialista en mano a mano, excelente juego aéreo. Subcapitán."},
    {"dorsal": 2, "name": "Bruno Benítez", "nickname": "Mariscal", "position": "Defensor Central Derecho", "age": 31, "height_m": 1.86, "weight_kg": 82.1, "notes": "Capitán del equipo, líder del vestuario y tesorero del fondo común, gran juego aéreo defensivo."},
    {"dorsal": 3, "name": "Facundo Medina", "nickname": "Zurdo", "position": "Lateral Izquierdo", "age": 24, "height_m": 1.77, "weight_kg": 73.0, "notes": "Recorrido de banda, precisión en centros, ejecutor de córners zurdos."},
    {"dorsal": 4, "name": "Rodrigo Aguirre", "nickname": "Vasco", "position": "Lateral Derecho", "age": 27, "height_m": 1.80, "weight_kg": 76.2, "notes": "Marcaje agresivo, potente físicamente, subida constante."},
    {"dorsal": 5, "name": "Thiago Almada", "nickname": "Pulpo", "position": "Mediocampista Central / Tapón", "age": 25, "height_m": 1.78, "weight_kg": 75.4, "notes": "Equilibrio defensivo, recuperador de pelotas, primer pase vertical. Subcapitán."},
    {"dorsal": 6, "name": "Lautaro Carrizo", "nickname": "Toro", "position": "Defensor Central Izquierdo", "age": 23, "height_m": 1.88, "weight_kg": 83.0, "notes": "Lesionado: Esguince de tobillo izquierdo grado II (LTFA). En RTP Fase 3."},
    {"dorsal": 7, "name": "Franco Valenzuela", "nickname": "Rayo", "position": "Extremo Derecho", "age": 22, "height_m": 1.74, "weight_kg": 69.1, "notes": "Velocidad pura (34 km/h), desborde. Carga controlada por sobrecarga en bíceps femoral derecho (ACWR 1.48)."},
    {"dorsal": 8, "name": "Gastón Pereyra", "nickname": "El Duende", "position": "Volante Mixto / Box-to-Box", "age": 26, "height_m": 1.82, "weight_kg": 77.8, "notes": "Despliegue aeróbico (7.840 m en MD-4), llegada al área rival y remate."},
    {"dorsal": 9, "name": "Tomás Cáceres", "nickname": "Flecha", "position": "Segundo Delantero / Extremo", "age": 20, "height_m": 1.79, "weight_kg": 72.4, "notes": "Canterano revelación, alta tasa de presión en campo contrario (5.4/90min)."},
    {"dorsal": 10, "name": "Julián Morales", "nickname": "La Joya", "position": "Mediapunta / Enganche", "age": 26, "height_m": 1.72, "weight_kg": 68.3, "notes": "Creador de juego, ejecutor N°1 de penales y tiros libres. Alta médica definitiva de pubalgia."},
    {"dorsal": 11, "name": "Emiliano Santos", "nickname": "Tanque", "position": "Centrodelantero Goleador", "age": 28, "height_m": 1.87, "weight_kg": 86.4, "notes": "Goleador del torneo (11 goles), juego de espaldas y fijación de centrales."},
    {"dorsal": 12, "name": "Lucas Barreto", "nickname": "Flaco", "position": "Arquero Suplente", "age": 22, "height_m": 1.91, "weight_kg": 85.0, "notes": "Gran envergadura (1.91 m, 85 kg), arquero de proyección."},
    {"dorsal": 14, "name": "Agustín Ríos", "nickname": "Motor", "position": "Volante Defensivo / Central Ocasional", "age": 23, "height_m": 1.81, "weight_kg": 77.2, "notes": "Diagnóstico de celiaquía: requiere protocolo nutricional estricto Sin TACC."},
    {"dorsal": 18, "name": "Marco Tagliafico", "nickname": "Pibe", "position": "Extremo / Lateral Izquierdo Juvenil", "age": 19, "height_m": 1.76, "weight_kg": 71.0, "notes": "Promovido de reserva, dinamismo y desfachatez en el 1v1."}
]

INITIAL_STAFF = [
    {"name": "Javier Méndez", "role": "Director Técnico Principal", "department_id": "cuerpo_tecnico", "email": "jmendez@puertomarina.fc", "phone": "+54 9 223 555-0101"},
    {"name": "Pablo D'Angelo", "role": "Asistente Técnico y Videoanalista", "department_id": "cuerpo_tecnico", "email": "pdangelo@puertomarina.fc", "phone": "+54 9 223 555-0102"},
    {"name": "Prof. Germán Silveira", "role": "Preparador Físico Jefe", "department_id": "preparacion_fisica", "email": "gsilveira@puertomarina.fc", "phone": "+54 9 223 555-0103"},
    {"name": "Dr. Federico Arismendi", "role": "Jefe de Departamento Médico", "department_id": "servicio_medico", "email": "farismendi@puertomarina.fc", "phone": "+54 9 223 555-0104"},
    {"name": "Lic. Valentina Russo", "role": "Kinesióloga y Fisioterapeuta", "department_id": "servicio_medico", "email": "vrusso@puertomarina.fc", "phone": "+54 9 223 555-0105"},
    {"name": "Lic. Sofía Carrizo", "role": "Nutricionista Deportiva ISAK 1", "department_id": "nutricion", "email": "scarrizo@puertomarina.fc", "phone": "+54 9 223 555-0106"},
    {"name": "Lucas Benítez", "role": "Director de Scouting y Datos", "department_id": "scouting", "email": "lbenitez@puertomarina.fc", "phone": "+54 9 223 555-0107"},
    {"name": "Carlos 'Charly' Pereyra", "role": "Gerente de Operaciones y Logística", "department_id": "logistica_viajes", "email": "cpereyra@puertomarina.fc", "phone": "+54 9 223 555-0108"},
    {"name": "Mariana Gómez Solís", "role": "Directora de Prensa y Medios", "department_id": "disciplina_prensa", "email": "mgomez@puertomarina.fc", "phone": "+54 9 223 555-0109"},
    {"name": "Lic. Matías Echeverría", "role": "Psicólogo Deportivo", "department_id": "servicio_medico", "email": "mecheverria@puertomarina.fc", "phone": "+54 9 223 555-0110"}
]


def seed_metadata(conn) -> None:
    """Populates departments, players, and staff tables."""
    cur = conn.cursor()

    for dept in INITIAL_DEPARTMENTS:
        cur.execute("""
            INSERT INTO departments (id, name, description, manager)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                manager=excluded.manager;
        """, (dept["id"], dept["name"], dept["description"], dept["manager"]))

    for p in INITIAL_PLAYERS:
        cur.execute("""
            INSERT INTO players (dorsal, name, nickname, position, age, height_m, weight_kg, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dorsal) DO UPDATE SET
                name=excluded.name,
                nickname=excluded.nickname,
                position=excluded.position,
                age=excluded.age,
                height_m=excluded.height_m,
                weight_kg=excluded.weight_kg,
                notes=excluded.notes;
        """, (p["dorsal"], p["name"], p["nickname"], p["position"], p["age"], p["height_m"], p["weight_kg"], p["notes"]))

    for s in INITIAL_STAFF:
        cur.execute("SELECT id FROM staff WHERE name = ?", (s["name"],))
        existing = cur.fetchone()
        if existing:
            cur.execute("""
                UPDATE staff SET role=?, department_id=?, email=?, phone=? WHERE id=?
            """, (s["role"], s["department_id"], s["email"], s["phone"], existing["id"]))
        else:
            cur.execute("""
                INSERT INTO staff (name, role, department_id, email, phone)
                VALUES (?, ?, ?, ?, ?)
            """, (s["name"], s["role"], s["department_id"], s["email"], s["phone"]))

    conn.commit()


def parse_markdown_file(file_path: Path) -> Dict[str, Any]:
    """Reads a markdown file with optional YAML frontmatter."""
    text = file_path.read_text(encoding="utf-8")
    frontmatter = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_fm = parts[1].strip()
            content = parts[2].strip()
            try:
                frontmatter = yaml.safe_load(raw_fm) or {}
            except Exception as e:
                print(f"Warning: could not parse frontmatter in {file_path}: {e}")

    doc_code = frontmatter.get("id") or file_path.stem
    title = frontmatter.get("title") or file_path.stem.replace("-", " ").title()
    department_id = frontmatter.get("department") or file_path.parent.name
    author = frontmatter.get("author") or "Club Atlético Puerto Marina"
    date_str = str(frontmatter.get("date") or "2026-11-01")
    confidentiality = frontmatter.get("confidentiality") or "interno"
    tags = frontmatter.get("tags")
    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags or "")

    return {
        "doc_code": doc_code,
        "title": title,
        "department_id": department_id,
        "author": author,
        "date": date_str,
        "confidentiality": confidentiality,
        "tags": tags_str,
        "raw_content": content,
        "file_path": str(file_path)
    }


def _split_oversized_text(text: str, max_chars: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """
    Splits text that exceeds max_chars: first tries sentences, then falls back to sliding window.
    """
    if len(text) <= max_chars:
        return [text]

    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_len = len(s_clean)

        if s_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Hard window split for long single sentence / table row
            start = 0
            while start < s_len:
                end = min(start + max_chars, s_len)
                chunks.append(s_clean[start:end])
                start += max_chars - overlap
            continue

        if current_len + s_len + 1 > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            if len(current_chunk[-1]) <= overlap:
                current_chunk = [current_chunk[-1], s_clean]
                current_len = len(current_chunk[0]) + s_len + 1
            else:
                current_chunk = [s_clean]
                current_len = s_len
        else:
            current_chunk.append(s_clean)
            current_len += s_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunks document content with:
    - Section header preservation & hierarchical title enrichment.
    - Sentence-aware and sliding-window splitting for oversized text.
    - Markdown table header retention across split chunks.
    - Automatic rejection of empty or junk chunks (e.g. standalone dividers or empty headings).
    - Explicit contextual prefix injection for dense vector and lexical alignment.
    """
    content = doc_meta["raw_content"]
    title = doc_meta["title"]
    dept = doc_meta["department_id"]
    author = doc_meta.get("author") or "Club Atlético Puerto Marina"
    date_str = str(doc_meta.get("date") or "2026-11-01")
    tags = doc_meta.get("tags") or ""

    section_pattern = re.compile(r'(?=^#{2,3}\s+)', re.MULTILINE)
    raw_sections = [s.strip() for s in section_pattern.split(content) if s.strip()]

    if not raw_sections:
        raw_sections = [content]

    final_chunks = []
    chunk_idx = 0

    for sec in raw_sections:
        # Determine section title if a heading exists
        header_match = re.search(r'^(#{2,3})\s+(.+)$', sec, re.MULTILINE)
        section_title = header_match.group(2).strip() if header_match else f"Sección {chunk_idx + 1}"
        chunk_title = f"{title} - {section_title}" if section_title != title else f"{title} (Parte {chunk_idx + 1})"

        # Check for junk / trivial section (< 25 alphanumeric chars)
        alnum_count = sum(1 for ch in sec if ch.isalnum())
        if alnum_count < 25:
            continue

        if len(sec) <= CHUNK_SIZE_CHARS:
            augmented_content = (
                f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                f"TAGS: {tags}\n\n"
                f"{sec}"
            )
            final_chunks.append({
                "chunk_index": chunk_idx,
                "title": chunk_title,
                "department_id": dept,
                "content": augmented_content,
                "metadata": {
                    "doc_code": doc_meta.get("doc_code", f"DOC-{chunk_idx+1}"),
                    "date": date_str,
                    "author": author,
                    "tags": tags,
                    "source_type": doc_meta.get("source_type", "markdown")
                }
            })
            chunk_idx += 1
        else:
            paragraphs = sec.split("\n\n")
            current_buffer = []
            current_len = 0
            last_table_header = None

            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue

                # Detect and track table headers to preserve them if table is split
                if p_clean.startswith("|") and "\n|" in p_clean and "---" in p_clean:
                    lines = p_clean.split("\n")
                    if len(lines) >= 2 and "---" in lines[1]:
                        last_table_header = lines[0] + "\n" + lines[1]

                # If paragraph itself is oversized, split it using sentence/window splitter
                sub_parts = _split_oversized_text(p_clean, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)

                for part in sub_parts:
                    part_len = len(part)
                    if current_len + part_len > CHUNK_SIZE_CHARS and current_buffer:
                        chunk_text = "\n\n".join(current_buffer)
                        augmented_content = (
                            f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                            f"TAGS: {tags}\n\n"
                            f"{chunk_text}"
                        )
                        final_chunks.append({
                            "chunk_index": chunk_idx,
                            "title": f"{chunk_title} (Parte {chunk_idx + 1})",
                            "department_id": dept,
                            "content": augmented_content,
                            "metadata": {
                                "doc_code": doc_meta.get("doc_code", f"DOC-{chunk_idx+1}"),
                                "date": date_str,
                                "author": author,
                                "tags": tags,
                                "source_type": doc_meta.get("source_type", "markdown")
                            }
                        })
                        chunk_idx += 1

                        # Carry forward table header if next part is table rows without header
                        if last_table_header and part.startswith("|") and "---" not in part:
                            current_buffer = [last_table_header, part]
                            current_len = len(last_table_header) + part_len + 2
                        elif len(current_buffer[-1]) < CHUNK_OVERLAP_CHARS:
                            current_buffer = [current_buffer[-1], part]
                            current_len = len(current_buffer[0]) + part_len + 2
                        else:
                            current_buffer = [part]
                            current_len = part_len
                    else:
                        current_buffer.append(part)
                        current_len += part_len + 2

            if current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                if sum(1 for ch in chunk_text if ch.isalnum()) >= 25:
                    augmented_content = (
                        f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                        f"TAGS: {tags}\n\n"
                        f"{chunk_text}"
                    )
                    final_chunks.append({
                        "chunk_index": chunk_idx,
                        "title": f"{chunk_title} (Parte {chunk_idx + 1})",
                        "department_id": dept,
                        "content": augmented_content,
                        "metadata": {
                            "doc_code": doc_meta.get("doc_code", f"DOC-{chunk_idx+1}"),
                            "date": date_str,
                            "author": author,
                            "tags": tags,
                            "source_type": doc_meta.get("source_type", "markdown")
                        }
                    })
                    chunk_idx += 1

    return final_chunks


def ingest_doc_metadata_dict(doc_meta: Dict[str, Any], conn, embedder) -> Tuple[int, int]:
    """Inserts or updates a parsed document dict and indexes its chunks into SQLite & FTS5."""
    cur = conn.cursor()

    cur.execute("SELECT id FROM departments WHERE id = ?", (doc_meta["department_id"],))
    if not cur.fetchone():
        doc_meta["department_id"] = "cuerpo_tecnico"

    source_type = doc_meta.get("source_type", "markdown")
    original_filename = doc_meta.get("original_filename") or (Path(doc_meta["file_path"]).name if doc_meta.get("file_path") else "")
    content_hash = doc_meta.get("content_hash") or hashlib.sha256(doc_meta["raw_content"].encode("utf-8")).hexdigest()
    file_size_bytes = int(doc_meta.get("file_size_bytes") or len(doc_meta["raw_content"].encode("utf-8")))

    cur.execute("""
        INSERT INTO documents (
            doc_code, title, department_id, author, date, file_path,
            confidentiality, tags, raw_content, source_type,
            original_filename, content_hash, file_size_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_code) DO UPDATE SET
            title=excluded.title,
            department_id=excluded.department_id,
            author=excluded.author,
            date=excluded.date,
            file_path=excluded.file_path,
            confidentiality=excluded.confidentiality,
            tags=excluded.tags,
            raw_content=excluded.raw_content,
            source_type=excluded.source_type,
            original_filename=excluded.original_filename,
            content_hash=excluded.content_hash,
            file_size_bytes=excluded.file_size_bytes
        RETURNING id;
    """, (
        doc_meta["doc_code"],
        doc_meta["title"],
        doc_meta["department_id"],
        doc_meta["author"],
        doc_meta["date"],
        doc_meta["file_path"],
        doc_meta.get("confidentiality", "interno"),
        doc_meta.get("tags", ""),
        doc_meta["raw_content"],
        source_type,
        original_filename,
        content_hash,
        file_size_bytes
    ))
    row = cur.fetchone()
    doc_id = row[0] if row else None

    if not doc_id:
        cur.execute("SELECT id FROM documents WHERE doc_code = ?", (doc_meta["doc_code"],))
        doc_id = cur.fetchone()["id"]

    # Delete previous chunks and FTS5 entries
    cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
    old_chunk_ids = [r["id"] for r in cur.fetchall()]
    for cid in old_chunk_ids:
        cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    chunks = chunk_document(doc_meta)
    if not chunks:
        GLOBAL_VECTOR_CACHE.invalidate()
        conn.commit()
        return doc_id, 0

    chunk_texts = [c["content"] for c in chunks]
    vectors = embedder.embed_batch(chunk_texts)

    for c, vec in zip(chunks, vectors):
        blob = embedder.vector_to_blob(vec)
        meta_json = json.dumps(c["metadata"], ensure_ascii=False)

        cur.execute("""
            INSERT INTO chunks (document_id, chunk_index, title, department_id, content, embedding, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id;
        """, (doc_id, c["chunk_index"], c["title"], c["department_id"], c["content"], blob, meta_json))
        chunk_id = cur.fetchone()[0]

        # Insert into FTS5 with explicit rowid = chunk_id for zero-cast JOINs
        cur.execute("""
            INSERT INTO chunks_fts (rowid, chunk_id, title, department_id, content, tags)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (chunk_id, str(chunk_id), c["title"], c["department_id"], c["content"], doc_meta["tags"]))

    conn.commit()
    GLOBAL_VECTOR_CACHE.invalidate()
    return doc_id, len(chunks)


def ingest_uploaded_file(
    file_path: Path,
    original_filename: str,
    department_id: str,
    author: str,
    tags: str,
    conn,
    embedder,
    file_bytes: Optional[bytes] = None
) -> Tuple[int, int, str]:
    """
    Ingests an uploaded document (PDF, DOCX, DOC, MD, TXT):
    1. Extracts structured Markdown using src.extractors.
    2. Computes SHA-256 hash.
    3. Handles deduplication.
    4. Indexes chunks and FTS5.
    5. Returns (doc_id, chunk_count, doc_code).
    """
    if file_bytes is None:
        file_bytes = file_path.read_bytes()

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    file_size_bytes = len(file_bytes)

    extraction = extract_document_content(file_bytes, original_filename)
    raw_markdown = extraction["text"]
    inferred_title = extraction.get("inferred_title", Path(original_filename).stem.replace("-", " ").title())
    source_format = extraction.get("format", Path(original_filename).suffix.lstrip("."))

    cur = conn.cursor()
    cur.execute("SELECT id, doc_code FROM documents WHERE content_hash = ?", (content_hash,))
    existing = cur.fetchone()
    if existing:
        doc_code = existing["doc_code"]
    else:
        dept_prefix = department_id[:3].upper() if department_id else "GEN"
        slug = re.sub(r'[^a-zA-Z0-9]', '', Path(original_filename).stem)[:12].upper()
        doc_code = f"UPL-{dept_prefix}-{slug}-{content_hash[:6].upper()}"

    doc_meta = {
        "doc_code": doc_code,
        "title": inferred_title,
        "department_id": department_id,
        "author": author or "Staff Técnico",
        "date": "2026-11-09",
        "confidentiality": "interno",
        "tags": tags,
        "raw_content": raw_markdown,
        "file_path": str(file_path),
        "source_type": source_format,
        "original_filename": original_filename,
        "content_hash": content_hash,
        "file_size_bytes": file_size_bytes
    }

    doc_id, chunk_count = ingest_doc_metadata_dict(doc_meta, conn, embedder)
    return doc_id, chunk_count, doc_code


def ingest_document_file(file_path: Path, conn, embedder) -> Tuple[int, int]:
    """Ingests a markdown file from disk."""
    doc_meta = parse_markdown_file(file_path)
    return ingest_doc_metadata_dict(doc_meta, conn, embedder)


def sync_relational_entities_to_rag(conn, embedder) -> Tuple[int, int]:
    """
    NEW IN V2.0: Synchronizes relational SQL tables (players, staff, departments)
    into virtual RAG documents so any natural language query about player biometrics,
    dorsals, roles, emails, or phone numbers is retrieved by both BM25 and Vector search.
    """
    cur = conn.cursor()

    # 1. Build Virtual Roster Document from SQL `players`
    players = cur.execute("SELECT * FROM players ORDER BY dorsal ASC;").fetchall()
    roster_lines = [
        "## Plantel Profesional Completo - Fichas Antropométricas, Dorsales y Estado",
        "| Dorsal | Nombre | Apodo | Posición | Edad | Altura (m) | Peso (kg) | Observaciones Clínicas y Tácticas |",
        "|---|---|---|---|---|---|---|---|"
    ]
    for p in players:
        roster_lines.append(
            f"| #{p['dorsal']} | {p['name']} | {p['nickname']} | {p['position']} | {p['age']} años | {p['height_m']} m | {p['weight_kg']} kg | {p['notes']} |"
        )

    player_sections = ["\n## Fichas Individuales de Futbolistas del Plantel"]
    for p in players:
        player_sections.append(
            f"### Futbolista: #{p['dorsal']} {p['name']} (\"{p['nickname']}\")\n"
            f"- **Dorsal**: #{p['dorsal']}\n"
            f"- **Nombre Completo**: {p['name']}\n"
            f"- **Posición**: {p['position']}\n"
            f"- **Edad**: {p['age']} años\n"
            f"- **Estatura / Altura**: {p['height_m']} m\n"
            f"- **Peso Corporal**: {p['weight_kg']} kg\n"
            f"- **Observaciones Clínicas y Tácticas**: {p['notes']}\n"
        )

    roster_doc = {
        "doc_code": "SQL-PLANTEL-2026",
        "title": "Padrón Oficial del Plantel Profesional - Peso, Altura, Edad y Dorsales",
        "department_id": "cuerpo_tecnico",
        "author": "Secretaría Técnica y Base de Datos SQL",
        "date": "2026-11-06",
        "confidentiality": "interno",
        "tags": "plantel, jugadores, dorsales, altura, peso, edad, posicion, arquero, delantero, defensor, mediocampista",
        "raw_content": "\n".join(roster_lines) + "\n\n" + "\n\n".join(player_sections),
        "file_path": "sqlite://players",
        "source_type": "relational_sql"
    }

    # 2. Build Virtual Staff & Departments Document from SQL `staff` & `departments`
    staff_rows = cur.execute("""
        SELECT s.*, d.name as dept_name, d.description as dept_desc
        FROM staff s
        JOIN departments d ON d.id = s.department_id
        ORDER BY s.id ASC;
    """).fetchall()

    staff_lines = [
        "## Directorio Oficial de Staff Técnico, Médico, Nutrición y Operaciones (Contactos, Teléfonos y Correos)",
        "| Profesional | Cargo / Rol | Departamento | Correo Electrónico (Email) | Teléfono Directo |",
        "|---|---|---|---|---|"
    ]
    for s in staff_rows:
        staff_lines.append(
            f"| {s['name']} | {s['role']} | {s['dept_name']} | {s['email']} | {s['phone']} |"
        )

    staff_sections = ["\n## Contacto Directo y Responsabilidades Individuales del Staff"]
    for s in staff_rows:
        staff_sections.append(
            f"### Integrante del Staff: {s['name']} ({s['role']})\n"
            f"- **Nombre Completo**: {s['name']}\n"
            f"- **Cargo / Función**: {s['role']}\n"
            f"- **Departamento**: {s['dept_name']}\n"
            f"- **Correo Electrónico Oficial (Email)**: {s['email']}\n"
            f"- **Teléfono Directo de Contacto**: {s['phone']}\n"
            f"- **Misión y Competencia del Área**: {s['dept_desc']}\n"
        )

    staff_doc = {
        "doc_code": "SQL-STAFF-2026",
        "title": "Directorio Oficial del Staff Profesional, Correos, Teléfonos y Jefaturas de Área",
        "department_id": "disciplina_prensa",
        "author": "Gerencia de Operaciones y Recursos Humanos",
        "date": "2026-11-06",
        "confidentiality": "interno",
        "tags": "staff, directorio, telefono, email, correo, contacto, medico, nutricionista, kinesiologa, entrenador, preparador_fisico",
        "raw_content": "\n".join(staff_lines) + "\n\n" + "\n\n".join(staff_sections),
        "file_path": "sqlite://staff",
        "source_type": "relational_sql"
    }

    _, c1 = ingest_doc_metadata_dict(roster_doc, conn, embedder)
    _, c2 = ingest_doc_metadata_dict(staff_doc, conn, embedder)
    return 2, (c1 + c2)


def run_full_ingestion(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Runs end-to-end ingestion over all markdown files AND relational SQL tables."""
    print(f"1. Inicializando esquema SQLite + FTS5 en {db_path}...")
    init_database(db_path)
    conn = get_db_connection(db_path)
    seed_metadata(conn)

    print("2. Cargando motor de embeddings multilingüe (FastEmbed ONNX)...")
    embedder = get_embedding_engine()

    total_docs = 0
    total_chunks = 0
    all_files = sorted(list(RAW_DOCS_DIR.rglob("*.md")))

    print(f"3. Indexando {len(all_files)} documentos Markdown desde {RAW_DOCS_DIR}...")
    for f in all_files:
        doc_id, count = ingest_document_file(f, conn, embedder)
        total_docs += 1
        total_chunks += count
        print(f"   -> [MD] {f.name} (Doc ID {doc_id}: {count} chunks)")

    print("4. Sincronizando tablas relacionales SQL (players, staff, departments) al índice RAG...")
    sql_docs, sql_chunks = sync_relational_entities_to_rag(conn, embedder)
    total_docs += sql_docs
    total_chunks += sql_chunks
    print(f"   -> [SQL] {sql_docs} documentos virtuales generados ({sql_chunks} chunks)")

    conn.close()
    GLOBAL_VECTOR_CACHE.invalidate(str(db_path))
    print(f"Ingesta completada con éxito: {total_docs} documentos y {total_chunks} fragmentos activos.")
    return {
        "documents_count": total_docs,
        "chunks_count": total_chunks,
        "db_path": str(db_path)
    }


if __name__ == "__main__":
    run_full_ingestion()
