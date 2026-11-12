#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Clean proxy environment variable if needed
if [[ "${no_proxy:-}" == *"[::1]"* ]]; then
  export no_proxy="${no_proxy//,[::1]/}"
  export no_proxy="${no_proxy//[::1]/}"
  export NO_PROXY="$no_proxy"
fi

# Detect uvicorn and python binaries across local .venv, legacy rag_env, or global environment
if [[ -x "$DIR/.venv/bin/uvicorn" && -x "$DIR/.venv/bin/python" ]]; then
  UVICORN_BIN="$DIR/.venv/bin/uvicorn"
  PYTHON_BIN="$DIR/.venv/bin/python"
elif [[ -x "/home/antigravity/rag_env/bin/uvicorn" && -x "/home/antigravity/rag_env/bin/python" ]]; then
  UVICORN_BIN="/home/antigravity/rag_env/bin/uvicorn"
  PYTHON_BIN="/home/antigravity/rag_env/bin/python"
else
  UVICORN_BIN="$(command -v uvicorn || true)"
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${UVICORN_BIN:-}" || -z "${PYTHON_BIN:-}" ]]; then
  echo "❌ Error: no se encontró 'uvicorn' o 'python3'." >&2
  echo "   Crea un entorno virtual e instala las dependencias:" >&2
  echo "   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Auto-initialize database if club_knowledge.db does not exist yet
if [[ ! -f "$DIR/data/database/club_knowledge.db" ]]; then
  echo "⚙️  Base de datos no encontrada. Ejecutando ingesta inicial e indexación híbrida..."
  "$PYTHON_BIN" -m src.ingestion
fi

echo "======================================================================"
echo " ⚽ INICIANDO SERVIDOR RAG v2.2 - CLUB ATLÉTICO PUERTO MARINA"
echo " Host Web UI:       http://${HOST}:${PORT}"
echo " Documentación API: http://${HOST}:${PORT}/docs"
echo "======================================================================"

exec "$UVICORN_BIN" src.api:app --host "$HOST" --port "$PORT"
