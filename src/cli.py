"""
Interactive CLI for Club Atlético Puerto Marina RAG System.
Allows instant queries, interactive terminal sessions, stats inspection, and re-indexing.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag_pipeline import ClubRAGPipeline
from src.ingestion import run_full_ingestion


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


def interactive_mode(pipeline):
    print_banner()
    print("Modo interactivo iniciado. Escribe tu pregunta (o 'salir' para terminar):")
    print("Ejemplos:")
    print("  - ¿Cuál es la evolución de la lesión de Lautaro Carrizo y cuándo puede jugar?")
    print("  - ¿Qué jugador tiene riesgo de sobrecarga según el reporte de GPS Catapult?")
    print("  - ¿Quién es celíaco y qué menú especial requiere en el hotel?")
    print("  - ¿Cuáles son las multas por llegar tarde al entrenamiento o por sobrepeso?")
    print("  - ¿Cuál es el análisis táctico y las debilidades del rival Deportivo Los Andes?")
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
            if ":" in q and q.split(":", 1)[0].lower() in ["medico", "servicio_medico", "gps", "preparacion_fisica", "tactica", "cuerpo_tecnico", "nutricion", "scouting", "logistica", "prensa"]:
                tag, actual_q = q.split(":", 1)
                mapping = {
                    "medico": "servicio_medico", "servicio_medico": "servicio_medico",
                    "gps": "preparacion_fisica", "preparacion_fisica": "preparacion_fisica",
                    "tactica": "cuerpo_tecnico", "cuerpo_tecnico": "cuerpo_tecnico",
                    "nutricion": "nutricion", "scouting": "scouting",
                    "logistica": "logistica_viajes", "prensa": "disciplina_prensa"
                }
                dept_filter = mapping.get(tag.strip().lower())
                q = actual_q.strip()

            result = pipeline.ask(q, department_filter=dept_filter)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            break


def main():
    parser = argparse.ArgumentParser(description="CLI de RAG para Club Atlético Puerto Marina")
    parser.add_argument("--query", "-q", type=str, help="Ejecutar una consulta directa")
    parser.add_argument("--dept", "-d", type=str, default=None, help="Filtrar por departamento (ej: servicio_medico, cuerpo_tecnico)")
    parser.add_argument("--top_k", "-k", type=int, default=4, help="Número de fragmentos a recuperar (default: 4)")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Peso híbrido RRF: 0=vector, 1=lexical (default: 0.5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Iniciar sesión interactiva")
    parser.add_argument("--stats", "-s", action="store_true", help="Mostrar estadísticas de la base de datos")
    parser.add_argument("--reindex", "-r", action="store_true", help="Reindexar todos los documentos de la carpeta data")

    args = parser.parse_args()

    if args.reindex:
        print_banner()
        run_full_ingestion()
        return

    pipeline = ClubRAGPipeline()

    if args.stats:
        print_banner()
        stats = pipeline.get_stats()
        print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
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

    if args.query:
        result = pipeline.ask(args.query, department_filter=args.dept, top_k=args.top_k, alpha=args.alpha)
        display_result(result)
        return

    # Default to interactive if no specific action provided
    interactive_mode(pipeline)


if __name__ == "__main__":
    main()
