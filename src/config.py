"""
Configuration module for Club Atlético Puerto Marina RAG System (v2.1).
Includes hybrid RRF constants, Spanish stopwords for FTS5, and resilient LLM environment settings.
"""
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _load_env_file(env_path: Path) -> None:
    """Loads .env using python-dotenv if available, with a native parser fallback."""
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val


_load_env_file(ENV_FILE)

DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
UPLOADS_DIR = DATA_DIR / "uploads"
DATABASE_DIR = DATA_DIR / "database"
DB_PATH = DATABASE_DIR / "club_knowledge.db"

# Upload and Security settings
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per file
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt"}

# Embedding Model (FastEmbed Multilingual ONNX - 384 dimensions)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Chunking settings
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Hybrid Search Settings
DEFAULT_TOP_K = 4
DEFAULT_HYBRID_ALPHA = 0.5  # 0.5 = 50% BM25 lexical, 50% dense vector
RRF_K_CONSTANT = 60


def _resolve_llm_config():
    """
    Resolves LLM settings supporting multiple standard env var names
    (LLM_*, OPENAI_*, GROQ_*, GEMINI_*) and normalizes endpoints/models
    to prevent HTTP 404 errors caused by mismatched provider defaults.
    """
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    ).strip()

    raw_base = (
        os.getenv("LLM_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or ""
    ).strip().rstrip("/")

    # Strip accidental endpoint suffixes if user pasted full chat completions URL
    for suffix in ("/chat/completions", "/models"):
        if raw_base.endswith(suffix):
            raw_base = raw_base[: -len(suffix)].rstrip("/")

    # Infer or normalize base URL based on key prefix / host
    if not raw_base:
        if api_key.startswith("gsk_"):
            raw_base = "https://api.groq.com/openai/v1"
        elif api_key.startswith("AIza"):
            raw_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        else:
            raw_base = "https://api.openai.com/v1"
    elif "api.groq.com" in raw_base and not raw_base.endswith("/openai/v1"):
        if raw_base.endswith("/v1") and "/openai/" not in raw_base:
            raw_base = raw_base[:-3] + "/openai/v1"
        elif not raw_base.endswith("/v1"):
            raw_base = raw_base.rstrip("/") + "/openai/v1"

    # Resolve model name supporting LLM_MODEL, OPENAI_MODEL, GROQ_MODEL, MODEL_NAME
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("GROQ_MODEL")
        or os.getenv("MODEL_NAME")
        or ""
    ).strip()

    if not model:
        if "groq.com" in raw_base or api_key.startswith("gsk_"):
            model = "qwen/qwen3.8-27b"
        elif "googleapis.com" in raw_base or api_key.startswith("AIza"):
            model = "gemini-2.5-flash"
        elif "localhost:11434" in raw_base or "127.0.0.1:11434" in raw_base:
            model = "llama3.1:8b"
        else:
            model = "gpt-4o-mini"

    use_ext = os.getenv("USE_EXTERNAL_LLM", "false").lower() in ("true", "1", "yes")
    timeout = int(os.getenv("LLM_TIMEOUT_SEC", "25"))
    return use_ext, raw_base, api_key, model, timeout


USE_EXTERNAL_LLM, LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT_SEC = _resolve_llm_config()

# Fallback candidate models for Groq / OpenAI-compatible servers if the configured model is deprecated/not found
GROQ_FALLBACK_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b"
]

# Spanish Stopwords to prevent FTS5 BM25 noise on natural language queries
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo",
    "de", "en", "por", "para", "con", "sin", "sobre", "entre", "hacia", "desde",
    "hasta", "ante", "bajo", "tras", "durante", "mediante", "que", "quien",
    "quienes", "cual", "cuales", "como", "cuando", "donde", "cuanto", "cuanta",
    "cuantos", "cuantas", "este", "esta", "estos", "estas", "ese", "esa", "esos",
    "esas", "aquel", "aquella", "aquellos", "aquellas", "mi", "mis", "tu", "tus",
    "su", "sus", "nuestro", "nuestra", "es", "son", "fue", "fueron", "era",
    "eran", "ser", "estar", "esta", "estan", "estuvo", "hay", "tiene", "tienen",
    "tener", "hacer", "hace", "hacen", "puede", "pueden", "podria", "mas", "menos",
    "muy", "tambien", "pero", "sino", "porque", "pues", "si", "no", "ya", "todo",
    "toda", "todos", "todas", "otro", "otra", "otros", "otras", "mismo", "misma"
}

# Proxy fix for huggingface / httpx environment
if "no_proxy" in os.environ and "[::1]" in os.environ.get("no_proxy", ""):
    clean_no_proxy = os.environ["no_proxy"].replace(",[::1]", "").replace("[::1]", "")
    os.environ["no_proxy"] = clean_no_proxy
    os.environ["NO_PROXY"] = clean_no_proxy
