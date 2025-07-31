# utils/config.py
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_PATH = DATA_DIR / "chroma_db"

# --- RAG Pipeline Configuration ---
# EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Sentence Transformers embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_COLLECTION_NAME = "zurich_policy_s655"

# --- Document Processing Parameters ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200 # Overlapping characters between chunks

# --- LLM (Ollama) Configuration ---
OLLAMA_MODELS = {
    "Mistral": "mistral:7b",
    "Llama2": "llama2:7b"
}
DEFAULT_MODEL = "Mistral"
LLM_TEMPERATURE = 0.1 # Model creativity (0.0 = deterministic)
LLM_TOP_P = 0.9 # Cumulative probability of tokens to consider

# --- User Interface (Streamlit) Configuration ---
APP_TITLE = "Zurich Policy Manager Assistant"
PAGE_ICON = "🤖"