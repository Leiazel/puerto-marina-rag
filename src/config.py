"""
Configuration module for Club Atlético Puerto Marina RAG System (v2.0).
Includes hybrid RRF constants, Spanish stopwords for FTS5, and LLM environment settings.
"""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
DATABASE_DIR = DATA_DIR / "database"
DB_PATH = DATABASE_DIR / "club_knowledge.db"

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

# External / Local LLM Settings (Ready for future plug-and-play)
USE_EXTERNAL_LLM = os.getenv("USE_EXTERNAL_LLM", "false").lower() in ("true", "1", "yes")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "20"))

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
