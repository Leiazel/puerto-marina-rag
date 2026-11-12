"""
Interactive CLI for Club Atlético Puerto Marina RAG System (v2.2).
Allows instant queries, interactive terminal sessions, stats inspection, and re-indexing.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DB_PATH
from src.rag_pipeline import ClubRAGPipeline
from src.ingestion import run_full_ingestion

DEPT_PREFIX_MAP = {
    "medico": "servicio_medico",
    "servicio_medico": "servicio_medico",
    "gps": "preparacion_fisica",
    "preparacion_fisica": "preparacion_fisica",
    "tactica": "cuerpo_tecnico",
    "cuerpo_tecnico": "cuerpo_tecnico",
    "nutricion": "nutricion",
    "scouting": "scouting",
    "logistica": "logistica_viajes",
    "logistica_viajes": "logistica_viajes",
    "prensa": "disciplina_prensa",
    "disciplina_prensa": "disciplina_prensa",
}
VALID_DEPARTMENTS = set(DEPT_PREFIX_MAP.values())


def print_banner():
    print("=" * 70)
    print(" ⚽ CLUB ATLÉTICO PUERTO MARINA - SISTEMA RAG DE INTELIGENCIA OPERATIVA")
    print(" Base de datos SQLite Híbrida (FTS5 BM25 + Vectores Densos Multilingües)")
    print("=" * 70)


def display_result(result):
    print("\n" + "=" * 70)
    print(f"🔎 CONSULTA: {result['query']}")
    if result.get("department_filter"):
        print(f"🏢 FILTRO ÁREA: {result['department_filter']}")
    print(f"🧠 MOTOR: {result.get('model_used', 'local')} | 🎯 CONFIANZA: {int(result.get('confidence', 0) * 100)}%")
    if result.get("fallback_used"):
        print(f"⚠️  FALLBACK ACTIVO: {result.get('fallback_reason')}")
    print(f"⏱️  TIEMPO TOTAL: {result['timings']['total_ms']} ms "
          f"(Búsqueda: {result['timings']['retrieval_ms']} ms | Síntesis: {result['timings']['generation_ms']} ms)")
    print("=" * 70 + "\n")

    print(result["answer"])
    print("\n" + "-" * 70)
    print(f"📑 FRAGMENTOS RECUPERADOS (Top-{len(result['retrieved_chunks'])}):")
    for idx, ch in enumerate(result["retrieved_chunks"], 1):
        doc_ref = ch.get("metadata", {}).get("doc_code", "DOC")
        modalities = "+".join(ch.get("modalities", []))
        print(f"\n[{idx}] Ref: {doc_ref} | {ch['title']} | Modalidades: {modalities}")
        print(f"    RRF Score: {ch['rrf_score']:.5f} | Cosine Sim: {ch['cosine_sim']:.4f} | BM25: {ch['bm25_score']}")
        if ch.get("match_snippet"):
            print(f"    Coincidencia FTS: {ch['match_snippet']}")
        preview = ch['content'].replace('\n', ' ')[:160]
        print(f"    Texto: {preview}...")
    print("=" * 70 + "\n")


def _ensure_db_initialized():
    if not DB_PATH.exists():
        print("⚙️  Base de datos no encontrada. Ejecutando ingesta inicial...")
        run_full_ingestion()


def interactive_mode(pipeline):
    print_banner()
    print("Modo interactivo iniciado. Escribe tu pregunta (o 'salir' para terminar):")
    print("Ejemplos:")
    print("  - ¿Cuál es la evolución de la lesión de Lautaro Carrizo y cuándo puede jugar?")
    print("  - ¿Qué jugador tiene riesgo de sobrecarga según el reporte de GPS Catapult?")
    print("  - nutricion: ¿Quién es celíaco y qué menú especial requiere en el hotel?")
    print("  - logistica: ¿En qué hotel se concentra el plantel?")
    print("  - prensa: ¿Cuáles son las multas por llegar tarde al entrenamiento?")
    print("-" * 70)

    while True:
        try:
            q = input("\n[Puerto Marina RAG] > ").strip()
            if not q:
                continue
            if q.lower() in ["salir", "exit", "quit", "q"]:
                print("Cerrando sesión. ¡Vamos Puerto Marina!")
                break

            # Check for department filter syntax: e.g. "medico: ¿como esta carrizo?"
            dept_filter = None
            if ":" in q:
                prefix, rest = q.split(":", 1)
                mapped = DEPT_PREFIX_MAP.get(prefix.strip().lower())
                if mapped:
                    dept_filter = mapped
                    q = rest.strip()
                    if not q:
                        print("⚠️  Escribe una consulta después del prefijo de departamento.")
                        continue

            result = pipeline.ask(q, department_filter=dept_filter)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            break


def main():
    parser = argparse.ArgumentParser(description="CLI de RAG para Club Atlético Puerto Marina")
    parser.add_argument("--query", "-q", type=str, help="Ejecutar una consulta directa")
    parser.add_argument("--dept", "-d", type=str, default=None, help="Filtrar por departamento (ej: servicio_medico, cuerpo_tecnico)")
    parser.add_argument("--top_k", "-k", type=int, default=4, help="Número de fragmentos a recuperar (1-15, default: 4)")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Peso híbrido RRF: 0=vector, 1=lexical (default: 0.5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Iniciar sesión interactiva")
    parser.add_argument("--stats", "-s", action="store_true", help="Mostrar estadísticas de la base de datos")
    parser.add_argument("--reindex", "-r", action="store_true", help="Reindexar todos los documentos de la carpeta data")
    parser.add_argument("--upload", "-u", nargs="+", help="Subir e indexar uno o más archivos (PDF, DOCX, DOC, MD, TXT)")
    parser.add_argument("--author", type=str, default="Staff Técnico", help="Autor del documento subido")
    parser.add_argument("--tags", type=str, default="", help="Etiquetas para documentos subidos (separadas por coma)")

    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha debe estar entre 0.0 y 1.0")
    if not (1 <= args.top_k <= 15):
        parser.error("--top_k debe estar entre 1 y 15")
    if args.dept and args.dept not in VALID_DEPARTMENTS:
        parser.error(f"--dept inválido '{args.dept}'. Opciones: {', '.join(sorted(VALID_DEPARTMENTS))}")

    if args.reindex:
        print_banner()
        run_full_ingestion()
        return

    _ensure_db_initialized()

    if args.upload:
        print_banner()
        from src.config import UPLOADS_DIR, ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
        from src.database import get_db_connection
        from src.embeddings import get_embedding_engine
        from src.ingestion import ingest_uploaded_file

        dept_id = args.dept or "cuerpo_tecnico"
        print(f"📤 Iniciando subida de {len(args.upload)} archivo(s) al departamento '{dept_id}'...")
        conn = get_db_connection(DB_PATH)
        embedder = get_embedding_engine()
        uploads_root = UPLOADS_DIR.resolve()
        dest_dir = (uploads_root / dept_id).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            for file_str in args.upload:
                file_path = Path(file_str)
                if not file_path.exists() or not file_path.is_file():
                    print(f"❌ Archivo no encontrado: {file_str}")
                    continue
                if file_path.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
                    print(f"❌ Extensión no soportada '{file_path.suffix}' en {file_path.name}")
                    continue
                raw_bytes = file_path.read_bytes()
                if len(raw_bytes) == 0 or len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
                    print(f"❌ Tamaño inválido en {file_path.name} ({len(raw_bytes)} bytes)")
                    continue
                try:
                    safe_name = file_path.name
                    dest_file = (dest_dir / safe_name).resolve()
                    if not dest_file.is_relative_to(dest_dir):
                        print(f"❌ Ruta de destino inválida para {file_path.name}")
                        continue
                    doc_id, count, doc_code = ingest_uploaded_file(
                        file_path=dest_file,
                        original_filename=file_path.name,
                        department_id=dept_id,
                        author=args.author,
                        tags=args.tags,
                        conn=conn,
                        embedder=embedder,
                        file_bytes=raw_bytes,
                        write_to_disk=True
                    )
                    print(f"✅ [{file_path.name}] -> Código {doc_code} (Doc ID {doc_id}: {count} chunks indexados)")
                except Exception as e:
                    print(f"❌ Error al procesar {file_path.name}: {e}")
        finally:
            conn.close()

        print("\n✨ Proceso de carga y sincronización completado.")
        return

    pipeline = ClubRAGPipeline()

    if args.stats:
        print_banner()
        stats = pipeline.get_stats()
        print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
        print(f" - Motor de síntesis activo:        {stats['synthesis_mode']}")
        print(f" - Documentos operativos indexados: {stats['documents_count']}")
        print(f" - Fragmentos (chunks) en SQLite:   {stats['chunks_count']}")
        print(f" - Futbolistas registrados:         {stats['players_count']}")
        print(f" - Integrantes de staff técnico:    {stats['staff_count']}")
        print(f" - Departamentos operativos:        {stats['departments_count']}")
        print(f" - Consultas auditadas en logs:     {stats['total_queries_logged']}")
        print(f" - Latencia promedio de consulta:   {stats['average_query_latency_ms']} ms\n")
        print("📁 Desglose por Departamento:")
        for d in stats["departments"]:
            print(f"   • {d['name']}: {d['doc_count']} documentos")
        print("=" * 70)
        return

    if args.query is not None:
        if not args.query.strip():
            parser.error("--query no puede estar vacío")
        result = pipeline.ask(args.query.strip(), department_filter=args.dept, top_k=args.top_k, alpha=args.alpha)
        display_result(result)
        return

    # Default to interactive if no specific action provided
    interactive_mode(pipeline)


if __name__ == "__main__":
    main()

