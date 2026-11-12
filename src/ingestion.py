"""
Ingestion pipeline module for Club Atlético Puerto Marina (v2.2).
- Parses markdown frontmatter safely and chunks documents with contextual prefixes.
- Preserves Markdown table headers across split chunks.
- Eliminates empty/separator-only/heading-only junk chunks.
- Automatically indexes relational SQL tables (players, staff, departments) into RAG chunks.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

from src.config import BASE_DIR, RAW_DOCS_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, DB_PATH
from src.database import get_db_connection, init_database
from src.embeddings import get_embedding_engine

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


def _portable_file_path(file_path: Path) -> str:
    """Stores path relative to BASE_DIR when inside the repo for portability across environments."""
    try:
        return file_path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(file_path)


def parse_markdown_file(file_path: Path) -> Dict[str, Any]:
    """Reads a markdown file with optional YAML frontmatter."""
    text = file_path.read_text(encoding="utf-8-sig")
    frontmatter: Dict[str, Any] = {}
    content = text

    fm_match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)', text, flags=re.DOTALL)
    if fm_match:
        raw_fm = fm_match.group(1).strip()
        content = text[fm_match.end():].strip()
        try:
            loaded = yaml.safe_load(raw_fm)
            if isinstance(loaded, dict):
                frontmatter = loaded
        except Exception as e:
            print(f"Warning: could not parse frontmatter in {file_path}: {e}")

    doc_code = str(frontmatter.get("id") or file_path.stem)
    title = str(frontmatter.get("title") or file_path.stem.replace("-", " ").title())
    department_id = str(frontmatter.get("department") or file_path.parent.name)
    author = str(frontmatter.get("author") or "Club Atlético Puerto Marina")
    date_str = str(frontmatter.get("date") or "2026-11-01")
    confidentiality = str(frontmatter.get("confidentiality") or "interno")
    tags = frontmatter.get("tags")
    if isinstance(tags, list):
        tags_str = ", ".join(str(t).strip() for t in tags if str(t).strip())
    else:
        tags_str = str(tags or "")

    return {
        "doc_code": doc_code,
        "title": title,
        "department_id": department_id,
        "author": author,
        "date": date_str,
        "confidentiality": confidentiality,
        "tags": tags_str,
        "raw_content": content,
        "file_path": _portable_file_path(file_path)
    }


def _is_heading_only(text: str) -> bool:
    """Returns True if every non-empty line in text is a Markdown heading or horizontal rule."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    return all(bool(re.match(r'^(#{1,6}\s+|---+$)', l)) for l in lines)


def _split_markdown_blocks(section_text: str) -> List[Tuple[str, str]]:
    """
    Parses a section into typed blocks: ('table', text) or ('text', text).
    Strips standalone horizontal rules (---) so they never become standalone chunks.
    """
    lines = section_text.splitlines()
    blocks: List[Tuple[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()
        if not stripped or re.match(r'^---+$', stripped):
            i += 1
            continue

        # Markdown table block
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
            continue

        # Prose / heading / list block
        text_lines = []
        while i < n:
            cur_s = lines[i].strip()
            if cur_s.startswith("|"):
                break
            if re.match(r'^---+$', cur_s):
                i += 1
                continue
            if not cur_s:
                if text_lines and text_lines[-1] != "":
                    text_lines.append("")
                i += 1
                continue
            text_lines.append(lines[i].rstrip())
            i += 1

        joined = "\n".join(text_lines).strip()
        if joined:
            blocks.append(("text", joined))

    return blocks


def _split_table_preserving_header(table_text: str, heading_prefix: str, max_body_chars: int) -> List[str]:
    """Splits a Markdown table into multiple body chunks while preserving the header + separator row on every chunk."""
    raw_lines = [l.strip() for l in table_text.splitlines() if l.strip()]
    if len(raw_lines) <= 2:
        full = f"{heading_prefix}\n{table_text}".strip() if heading_prefix else table_text
        return [full]

    header_rows: List[str] = []
    data_start = 0
    if len(raw_lines) >= 2 and "---" in raw_lines[1]:
        header_rows = raw_lines[:2]
        data_start = 2
    else:
        header_rows = [raw_lines[0]]
        cols = max(1, raw_lines[0].count("|") - 1)
        header_rows.append("|" + "---|" * cols)
        data_start = 1

    data_rows = raw_lines[data_start:]
    if not data_rows:
        full = f"{heading_prefix}\n" + "\n".join(header_rows)
        return [full.strip()]

    table_header_str = "\n".join(header_rows)
    base_prefix = f"{heading_prefix}\n{table_header_str}".strip() if heading_prefix else table_header_str

    chunks: List[str] = []
    current_rows: List[str] = []
    current_len = len(base_prefix)

    for row in data_rows:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > max_body_chars:
            chunks.append(base_prefix + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len = len(base_prefix) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunks.append(base_prefix + "\n" + "\n".join(current_rows))

    return chunks


def _split_text_lines(text: str, max_body_chars: int, overlap_chars: int) -> List[str]:
    """Splits long prose or list blocks by line boundaries with clean overlap."""
    if len(text) <= max_body_chars:
        return [text]

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_lines and current_len + line_len > max_body_chars:
            chunks.append("\n".join(current_lines))
            # Retain trailing lines up to overlap_chars
            overlap_buf: List[str] = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                if overlap_len + len(prev_line) + 1 <= overlap_chars:
                    overlap_buf.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1
                else:
                    break
            current_lines = overlap_buf + [line]
            current_len = sum(len(x) + 1 for x in current_lines)
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def chunk_document(doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunks markdown content by headers (##, ###), preserves table headers across splits,
    eliminates heading-only/separator-only junk chunks, and enriches each chunk with a
    contextual prefix.
    """
    content = doc_meta["raw_content"]
    title = doc_meta["title"]
    dept = doc_meta["department_id"]
    author = doc_meta["author"]
    date_str = doc_meta["date"]
    tags = doc_meta["tags"]

    prefix = (
        f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
        f"TAGS: {tags}\n\n"
    )
    max_body_chars = max(550, CHUNK_SIZE_CHARS - len(prefix))

    section_pattern = re.compile(r'(?=^#{1,3}\s+)', re.MULTILINE)
    raw_sections = [s.strip() for s in section_pattern.split(content) if s.strip()]
    if not raw_sections:
        raw_sections = [content.strip()] if content.strip() else []

    # Merge heading-only sections forward so headers never stand alone as junk chunks
    merged_sections: List[str] = []
    pending_header = ""
    for sec in raw_sections:
        cleaned_lines = [l for l in sec.splitlines() if not re.match(r'^\s*---+\s*$', l)]
        cleaned_sec = "\n".join(cleaned_lines).strip()
        if not cleaned_sec:
            continue
        if _is_heading_only(cleaned_sec):
            pending_header = f"{pending_header}\n{cleaned_sec}".strip() if pending_header else cleaned_sec
        else:
            if pending_header:
                cleaned_sec = f"{pending_header}\n{cleaned_sec}"
                pending_header = ""
            merged_sections.append(cleaned_sec)

    if pending_header:
        if merged_sections:
            merged_sections[-1] = f"{merged_sections[-1]}\n\n{pending_header}"
        else:
            merged_sections.append(pending_header)

    body_chunks: List[str] = []
    for sec in merged_sections:
        if len(sec) <= max_body_chars:
            body_chunks.append(sec)
            continue

        blocks = _split_markdown_blocks(sec)
        current_buf: List[str] = []
        current_len = 0
        pending_sec_heading = ""

        for b_type, b_text in blocks:
            if b_type == "text" and _is_heading_only(b_text):
                pending_sec_heading = f"{pending_sec_heading}\n{b_text}".strip() if pending_sec_heading else b_text
                continue

            if b_type == "table":
                if current_buf:
                    body_chunks.append("\n\n".join(current_buf))
                    current_buf = []
                    current_len = 0

                table_chunks = _split_table_preserving_header(b_text, pending_sec_heading, max_body_chars)
                pending_sec_heading = ""
                body_chunks.extend(table_chunks)
                continue

            # Text block
            full_text = f"{pending_sec_heading}\n{b_text}".strip() if pending_sec_heading else b_text
            pending_sec_heading = ""

            if len(full_text) > max_body_chars:
                if current_buf:
                    body_chunks.append("\n\n".join(current_buf))
                    current_buf = []
                    current_len = 0
                body_chunks.extend(_split_text_lines(full_text, max_body_chars, CHUNK_OVERLAP_CHARS))
            elif current_buf and current_len + len(full_text) + 2 > max_body_chars:
                body_chunks.append("\n\n".join(current_buf))
                current_buf = [full_text]
                current_len = len(full_text)
            else:
                current_buf.append(full_text)
                current_len += len(full_text) + 2

        if pending_sec_heading:
            if current_buf:
                current_buf.append(pending_sec_heading)
            elif body_chunks:
                body_chunks[-1] = f"{body_chunks[-1]}\n\n{pending_sec_heading}"

        if current_buf:
            body_chunks.append("\n\n".join(current_buf))

    final_chunks = []
    for chunk_idx, body in enumerate(body_chunks):
        if _is_heading_only(body):
            continue
        label = "Sección" if len(body_chunks) == 1 else "Parte"
        final_chunks.append({
            "chunk_index": len(final_chunks),
            "title": f"{title} ({label} {len(final_chunks) + 1})",
            "department_id": dept,
            "content": f"{prefix}{body}",
            "metadata": {
                "doc_code": doc_meta["doc_code"],
                "date": date_str,
                "author": author,
                "tags": tags
            }
        })

    return final_chunks


def ingest_doc_metadata_dict(doc_meta: Dict[str, Any], conn, embedder) -> Tuple[int, int]:
    """Inserts or updates a parsed document dict and indexes its chunks into SQLite & FTS5 atomically."""
    cur = conn.cursor()

    cur.execute("SELECT id FROM departments WHERE id = ?", (doc_meta["department_id"],))
    if not cur.fetchone():
        doc_meta["department_id"] = "cuerpo_tecnico"

    try:
        cur.execute("""
            INSERT INTO documents (doc_code, title, department_id, author, date, file_path, confidentiality, tags, raw_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_code) DO UPDATE SET
                title=excluded.title,
                department_id=excluded.department_id,
                author=excluded.author,
                date=excluded.date,
                file_path=excluded.file_path,
                confidentiality=excluded.confidentiality,
                tags=excluded.tags,
                raw_content=excluded.raw_content
            RETURNING id;
        """, (
            doc_meta["doc_code"],
            doc_meta["title"],
            doc_meta["department_id"],
            doc_meta["author"],
            doc_meta["date"],
            doc_meta["file_path"],
            doc_meta["confidentiality"],
            doc_meta["tags"],
            doc_meta["raw_content"]
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
        return doc_id, len(chunks)
    except Exception:
        conn.rollback()
        raise


def ingest_document_file(file_path: Path, conn, embedder) -> Tuple[int, int]:
    """Ingests a markdown file from disk."""
    doc_meta = parse_markdown_file(file_path)
    return ingest_doc_metadata_dict(doc_meta, conn, embedder)


def delete_document_by_id(doc_id: int, conn) -> Optional[Dict[str, Any]]:
    """Deletes a document and cleans up its chunks and FTS5 index entries."""
    cur = conn.cursor()
    cur.execute("SELECT id, doc_code, title, file_path FROM documents WHERE id = ?", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        return None

    try:
        cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_ids = [r["id"] for r in cur.fetchall()]
        for cid in chunk_ids:
            cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return dict(doc)
    except Exception:
        conn.rollback()
        raise


def sync_relational_entities_to_rag(conn, embedder) -> Tuple[int, int]:
    """
    Synchronizes relational SQL tables (players, staff, departments)
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
    player_bullets = ["## Detalle Individual de Futbolistas del Plantel"]
    for p in players:
        roster_lines.append(
            f"| #{p['dorsal']} | {p['name']} | {p['nickname']} | {p['position']} | {p['age']} años | {p['height_m']} m | {p['weight_kg']} kg | {p['notes']} |"
        )
        player_bullets.append(
            f"- **Dorsal #{p['dorsal']} - {p['name']} (\"{p['nickname']}\")**: Juega como {p['position']}, tiene {p['age']} años, mide {p['height_m']} m y pesa {p['weight_kg']} kg. Nota oficial: {p['notes']}"
        )

    roster_doc = {
        "doc_code": "SQL-PLANTEL-2026",
        "title": "Padrón Oficial del Plantel Profesional - Peso, Altura, Edad y Dorsales",
        "department_id": "cuerpo_tecnico",
        "author": "Secretaría Técnica y Base de Datos SQL",
        "date": "2026-11-06",
        "confidentiality": "interno",
        "tags": "plantel, jugadores, dorsales, altura, peso, edad, posicion, arquero, delantero, defensor, mediocampista",
        "raw_content": "\n".join(roster_lines) + "\n\n" + "\n".join(player_bullets),
        "file_path": "sqlite://players"
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
    staff_bullets = ["## Contacto Directo y Responsabilidades del Staff"]
    for s in staff_rows:
        staff_lines.append(
            f"| {s['name']} | {s['role']} | {s['dept_name']} | {s['email']} | {s['phone']} |"
        )
        staff_bullets.append(
            f"- **{s['name']}** ({s['role']} en {s['dept_name']}): Email `{s['email']}` | Teléfono `{s['phone']}`. Área encargada de: {s['dept_desc']}"
        )

    staff_doc = {
        "doc_code": "SQL-STAFF-2026",
        "title": "Directorio Oficial del Staff Profesional, Correos, Teléfonos y Jefaturas de Área",
        "department_id": "disciplina_prensa",
        "author": "Gerencia de Operaciones y Recursos Humanos",
        "date": "2026-11-06",
        "confidentiality": "interno",
        "tags": "staff, directorio, telefono, email, correo, contacto, medico, nutricionista, kinesiologa, entrenador, preparador_fisico",
        "raw_content": "\n".join(staff_lines) + "\n\n" + "\n".join(staff_bullets),
        "file_path": "sqlite://staff"
    }

    _, c1 = ingest_doc_metadata_dict(roster_doc, conn, embedder)
    _, c2 = ingest_doc_metadata_dict(staff_doc, conn, embedder)
    return 2, (c1 + c2)


def run_full_ingestion(
    db_path: Path = DB_PATH,
    docs_dir: Path = RAW_DOCS_DIR
) -> Dict[str, Any]:
    """Runs end-to-end ingestion over all markdown files AND relational SQL tables, cleaning orphan docs."""
    print("1. Inicializando esquema SQLite + FTS5...")
    init_database(db_path)
    conn = get_db_connection(db_path)
    try:
        seed_metadata(conn)

        print("2. Cargando motor de embeddings multilingüe (FastEmbed ONNX)...")
        embedder = get_embedding_engine()

        total_docs = 0
        total_chunks = 0
        active_doc_codes = {"SQL-PLANTEL-2026", "SQL-STAFF-2026"}
        all_files = sorted(list(docs_dir.rglob("*.md"))) if docs_dir.exists() else []

        print(f"3. Indexando {len(all_files)} documentos Markdown desde {docs_dir}...")
        for f in all_files:
            doc_meta = parse_markdown_file(f)
            active_doc_codes.add(doc_meta["doc_code"])
            doc_id, count = ingest_doc_metadata_dict(doc_meta, conn, embedder)
            total_docs += 1
            total_chunks += count
            print(f"   -> [MD] {f.name} (Doc ID {doc_id}: {count} chunks)")

        print("4. Sincronizando tablas relacionales SQL (players, staff, departments) al índice RAG...")
        sql_docs, sql_chunks = sync_relational_entities_to_rag(conn, embedder)
        total_docs += sql_docs
        total_chunks += sql_chunks
        print(f"   -> [SQL] {sql_docs} documentos virtuales generados ({sql_chunks} chunks)")

        # 5. Clean up orphan documents whose files were removed from disk
        cur = conn.cursor()
        existing_docs = cur.execute("SELECT id, doc_code FROM documents").fetchall()
        removed_count = 0
        for row in existing_docs:
            if row["doc_code"] not in active_doc_codes:
                delete_document_by_id(row["id"], conn)
                removed_count += 1

        if removed_count:
            print(f"5. Limpieza completada: {removed_count} documentos huérfanos eliminados del índice.")

        print(f"Ingesta completada con éxito: {total_docs} documentos y {total_chunks} fragmentos activos.")
        return {
            "documents_count": total_docs,
            "chunks_count": total_chunks,
            "removed_orphans": removed_count,
            "db_path": str(db_path)
        }
    finally:
        conn.close()


if __name__ == "__main__":
    run_full_ingestion()

