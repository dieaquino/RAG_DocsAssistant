# models/llm_manager.py
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils.config import OLLAMA_MODELS, LLM_TEMPERATURE, LLM_TOP_P

def _get_common_config(model_key: str) -> dict:
    """Helper to get shared model configuration."""
    model_name = OLLAMA_MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Model '{model_key}' not found in configuration.")
    return {
        "model": model_name,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
    }

def get_llm(model_key: str) -> OllamaLLM:
    """
    Gets a configured instance of a base LLM (for chains like RetrievalQA).
    Ideal for the console demo.
    """
    print(f"Loading base LLM model: {model_key}")
    config = _get_common_config(model_key)
    return OllamaLLM(
        **config,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_chat_model(model_key: str) -> ChatOllama:
    """
    Gets a configured instance of a Chat Model (for conversational chains).
    Ideal for the Streamlit application.
    """
    print(f"Loading Chat Model: {model_key}")
    config = _get_common_config(model_key)
    # For Streamlit, we don't use the stdout callback as it has its own way of handling streaming.
    return ChatOllama(**config)