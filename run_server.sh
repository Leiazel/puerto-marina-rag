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

echo "======================================================================"
echo " ⚽ INICIANDO SERVIDOR RAG - CLUB ATLÉTICO PUERTO MARINA"
echo " Host: http://127.0.0.1:8000"
echo " Documentación API: http://127.0.0.1:8000/docs"
echo "======================================================================"

/home/antigravity/rag_env/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
