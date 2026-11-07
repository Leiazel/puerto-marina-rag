"""
Configuration module for Club Atlético Puerto Marina RAG System.
"""
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
DATABASE_DIR = DATA_DIR / "database"
DB_PATH = DATABASE_DIR / "club_knowledge.db"

# Embedding Model
# FastEmbed Multilingual model (384 dimensions, trained for 50+ languages including Spanish)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Chunking settings
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Hybrid Search Settings
DEFAULT_TOP_K = 4
DEFAULT_HYBRID_ALPHA = 0.5  # 0.5 = 50% BM25 lexical, 50% dense vector
RRF_K_CONSTANT = 60

# Proxy fix for huggingface / httpx environment
if "no_proxy" in os.environ and "[::1]" in os.environ.get("no_proxy", ""):
    clean_no_proxy = os.environ["no_proxy"].replace(",[::1]", "").replace("[::1]", "")
    os.environ["no_proxy"] = clean_no_proxy
    os.environ["NO_PROXY"] = clean_no_proxy
