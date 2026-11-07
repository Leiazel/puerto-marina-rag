"""
Ingestion pipeline module for Club Atlético Puerto Marina.
Parses markdown frontmatter, chunks documents intelligently with contextual prefixes,
generates dense vector embeddings, and populates SQLite relational + FTS5 tables.
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml

from src.config import RAW_DOCS_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, DB_PATH
from src.database import get_db_connection, init_database
from src.embeddings import get_embedding_engine

# Seed master data
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
    {"dorsal": 1, "name": "Mateo Rossi", "nickname": "Araña", "position": "Arquero Titular", "age": 29, "height_m": 1.89, "weight_kg": 84.2, "notes": "Especialista en mano a mano, excelente juego aéreo."},
    {"dorsal": 2, "name": "Bruno Benítez", "nickname": "Mariscal", "position": "Defensor Central Derecho", "age": 31, "height_m": 1.86, "weight_kg": 82.1, "notes": "Capitán del equipo, líder del vestuario, gran juego aéreo defensivo."},
    {"dorsal": 3, "name": "Facundo Medina", "nickname": "Zurdo", "position": "Lateral Izquierdo", "age": 24, "height_m": 1.77, "weight_kg": 73.0, "notes": "Recorrido de banda, precisión en centros, ejecutor de córners zurdos."},
    {"dorsal": 4, "name": "Rodrigo Aguirre", "nickname": "Vasco", "position": "Lateral Derecho", "age": 27, "height_m": 1.80, "weight_kg": 76.2, "notes": "Marcaje agresivo, potente físicamente, subida constante."},
    {"dorsal": 5, "name": "Thiago Almada", "nickname": "Pulpo", "position": "Mediocampista Central / Tapón", "age": 25, "height_m": 1.78, "weight_kg": 75.4, "notes": "Equilibrio defensivo, recuperador de pelotas, primer pase vertical."},
    {"dorsal": 6, "name": "Lautaro Carrizo", "nickname": "Toro", "position": "Defensor Central Izquierdo", "age": 23, "height_m": 1.88, "weight_kg": 83.0, "notes": "Lesionado: Esguince de tobillo grado II. En RTP Fase 3."},
    {"dorsal": 7, "name": "Franco Valenzuela", "nickname": "Rayo", "position": "Extremo Derecho", "age": 22, "height_m": 1.74, "weight_kg": 69.1, "notes": "Velocidad pura (34 km/h), desborde. Carga controlada por sobrecarga en bíceps femoral."},
    {"dorsal": 8, "name": "Gastón Pereyra", "nickname": "El Duende", "position": "Volante Mixto / Box-to-Box", "age": 26, "height_m": 1.82, "weight_kg": 77.8, "notes": "Despliegue aeróbico (8 km/partido), llegada al área rival y remate."},
    {"dorsal": 9, "name": "Tomás Cáceres", "nickname": "Flecha", "position": "Segundo Delantero / Extremo", "age": 20, "height_m": 1.79, "weight_kg": 72.4, "notes": "Canterano revelación, alta tasa de presión en campo contrario (5.4/90min)."},
    {"dorsal": 10, "name": "Julián Morales", "nickname": "La Joya", "position": "Mediapunta / Enganche", "age": 26, "height_m": 1.72, "weight_kg": 68.3, "notes": "Creador de juego, ejecutor N°1 de penales y tiros libres. Alta médica de pubalgia."},
    {"dorsal": 11, "name": "Emiliano Santos", "nickname": "Tanque", "position": "Centrodelantero Goleador", "age": 28, "height_m": 1.87, "weight_kg": 86.4, "notes": "Goleador del torneo (11 goles), juego de espaldas y fijación de centrales."},
    {"dorsal": 12, "name": "Lucas Barreto", "nickname": "Flaco", "position": "Arquero Suplente", "age": 22, "height_m": 1.91, "weight_kg": 85.0, "notes": "Gran envergadura, arquero de proyección."},
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
    
    # Departments
    for dept in INITIAL_DEPARTMENTS:
        cur.execute("""
            INSERT INTO departments (id, name, description, manager)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                manager=excluded.manager;
        """, (dept["id"], dept["name"], dept["description"], dept["manager"]))

    # Players
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

    # Staff
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

    # Fallback values
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


def chunk_document(doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Intelligently chunks markdown content by headers (##, ###) or sliding window,
    enriching each chunk with an explicit contextual prefix (Title, Department, Date, Author)
    so the embedding captures full semantic context.
    """
    content = doc_meta["raw_content"]
    title = doc_meta["title"]
    dept = doc_meta["department_id"]
    author = doc_meta["author"]
    date_str = doc_meta["date"]
    tags = doc_meta["tags"]

    # Split on markdown section headers (## or ###)
    section_pattern = re.compile(r'(?=^#{2,3}\s+)', re.MULTILINE)
    raw_sections = [s.strip() for s in section_pattern.split(content) if s.strip()]

    if not raw_sections:
        raw_sections = [content]

    final_chunks = []
    chunk_idx = 0

    for sec in raw_sections:
        # If section is within comfortable chunk length
        if len(sec) <= CHUNK_SIZE_CHARS:
            chunk_text = sec
            # Contextual prefix for high semantic accuracy in vector and lexical search
            augmented_content = (
                f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                f"TAGS: {tags}\n\n"
                f"{chunk_text}"
            )
            final_chunks.append({
                "chunk_index": chunk_idx,
                "title": f"{title} (Sección {chunk_idx + 1})",
                "department_id": dept,
                "content": augmented_content,
                "metadata": {
                    "doc_code": doc_meta["doc_code"],
                    "date": date_str,
                    "author": author,
                    "tags": tags,
                    "raw_section": chunk_text[:200]
                }
            })
            chunk_idx += 1
        else:
            # Sub-split long section with sliding window
            paragraphs = sec.split("\n\n")
            current_buffer = []
            current_len = 0

            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len > CHUNK_SIZE_CHARS and current_buffer:
                    chunk_text = "\n\n".join(current_buffer)
                    augmented_content = (
                        f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                        f"TAGS: {tags}\n\n"
                        f"{chunk_text}"
                    )
                    final_chunks.append({
                        "chunk_index": chunk_idx,
                        "title": f"{title} (Parte {chunk_idx + 1})",
                        "department_id": dept,
                        "content": augmented_content,
                        "metadata": {
                            "doc_code": doc_meta["doc_code"],
                            "date": date_str,
                            "author": author,
                            "tags": tags
                        }
                    })
                    chunk_idx += 1
                    # Sliding overlap: keep last paragraph if reasonable
                    if len(current_buffer[-1]) < CHUNK_OVERLAP_CHARS:
                        current_buffer = [current_buffer[-1], p]
                        current_len = len(current_buffer[0]) + p_len
                    else:
                        current_buffer = [p]
                        current_len = p_len
                else:
                    current_buffer.append(p)
                    current_len += p_len + 2

            if current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                augmented_content = (
                    f"DOCUMENTO: {title} | ÁREA: {dept} | FECHA: {date_str} | AUTOR: {author}\n"
                    f"TAGS: {tags}\n\n"
                    f"{chunk_text}"
                )
                final_chunks.append({
                    "chunk_index": chunk_idx,
                    "title": f"{title} (Parte {chunk_idx + 1})",
                    "department_id": dept,
                    "content": augmented_content,
                    "metadata": {
                        "doc_code": doc_meta["doc_code"],
                        "date": date_str,
                        "author": author,
                        "tags": tags
                    }
                })
                chunk_idx += 1

    return final_chunks


def ingest_document_file(file_path: Path, conn, embedder) -> Tuple[int, int]:
    """Ingests a single markdown document and inserts chunks into SQLite & FTS5."""
    doc_meta = parse_markdown_file(file_path)
    cur = conn.cursor()

    # Check if department exists, if not fallback to 'cuerpo_tecnico'
    cur.execute("SELECT id FROM departments WHERE id = ?", (doc_meta["department_id"],))
    if not cur.fetchone():
        doc_meta["department_id"] = "cuerpo_tecnico"

    # Insert or update document
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

    # Delete previous chunks and FTS5 entries for this document
    cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
    old_chunk_ids = [r["id"] for r in cur.fetchall()]
    for cid in old_chunk_ids:
        cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (str(cid),))
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    # Chunk content
    chunks = chunk_document(doc_meta)
    if not chunks:
        return doc_id, 0

    # Batch embed chunks
    chunk_texts = [c["content"] for c in chunks]
    vectors = embedder.embed_batch(chunk_texts)

    # Insert into chunks and chunks_fts
    for c, vec in zip(chunks, vectors):
        blob = embedder.vector_to_blob(vec)
        meta_json = json.dumps(c["metadata"], ensure_ascii=False)
        
        cur.execute("""
            INSERT INTO chunks (document_id, chunk_index, title, department_id, content, embedding, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id;
        """, (doc_id, c["chunk_index"], c["title"], c["department_id"], c["content"], blob, meta_json))
        chunk_id = cur.fetchone()[0]

        # Sync to FTS5 virtual table
        cur.execute("""
            INSERT INTO chunks_fts (chunk_id, title, department_id, content, tags)
            VALUES (?, ?, ?, ?, ?);
        """, (str(chunk_id), c["title"], c["department_id"], c["content"], doc_meta["tags"]))

    conn.commit()
    return doc_id, len(chunks)


def run_full_ingestion() -> Dict[str, Any]:
    """Runs end-to-end ingestion over all raw documents in the data folder."""
    print("Iniciando inicialización de la base de datos...")
    init_database(DB_PATH)
    conn = get_db_connection(DB_PATH)
    seed_metadata(conn)

    print("Cargando modelo de embeddings (FastEmbed multilingual)...")
    embedder = get_embedding_engine()

    total_docs = 0
    total_chunks = 0
    all_files = list(RAW_DOCS_DIR.rglob("*.md"))

    print(f"Encontrados {len(all_files)} documentos para indexar en {RAW_DOCS_DIR}...")
    for f in all_files:
        doc_id, count = ingest_document_file(f, conn, embedder)
        total_docs += 1
        total_chunks += count
        print(f" -> Indexado: {f.name} (Doc ID {doc_id}, {count} chunks generados)")

    conn.close()
    print(f"Ingesta finalizada: {total_docs} documentos, {total_chunks} fragmentos indexados.")
    return {
        "documents_count": total_docs,
        "chunks_count": total_chunks,
        "db_path": str(DB_PATH)
    }


if __name__ == "__main__":
    run_full_ingestion()
