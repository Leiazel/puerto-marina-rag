#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Clean proxy environment variable if needed
if [[ "$no_proxy" == *"[::1]"* ]]; then
  export no_proxy="${no_proxy//,[::1]/}"
  export no_proxy="${no_proxy//[::1]/}"
  export NO_PROXY="$no_proxy"
fi

# Detect uvicorn binary across local venv, custom path, or global environment
if [[ -x "$DIR/.venv/bin/uvicorn" ]]; then
  UVICORN_BIN="$DIR/.venv/bin/uvicorn"
  PYTHON_BIN="$DIR/.venv/bin/python"
elif [[ -x "/home/antigravity/rag_env/bin/uvicorn" ]]; then
  UVICORN_BIN="/home/antigravity/rag_env/bin/uvicorn"
  PYTHON_BIN="/home/antigravity/rag_env/bin/python"
else
  UVICORN_BIN="$(command -v uvicorn)"
  PYTHON_BIN="$(command -v python3)"
fi

# Auto-initialize database if club_knowledge.db does not exist yet
if [[ ! -f "$DIR/data/database/club_knowledge.db" ]]; then
  echo "⚙️  Base de datos no encontrada. Ejecutando ingesta inicial e indexación híbrida..."
  "$PYTHON_BIN" -m src.ingestion
fi

echo "======================================================================"
echo " ⚽ INICIANDO SERVIDOR RAG v2.0 - CLUB ATLÉTICO PUERTO MARINA"
echo " Host Web UI:       http://127.0.0.1:8000"
echo " Documentación API: http://127.0.0.1:8000/docs"
echo "======================================================================"

"$UVICORN_BIN" src.api:app --host 0.0.0.0 --port 8000
