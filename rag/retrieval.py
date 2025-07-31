# rag/retrieval.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Sequence, Any, Dict

from rag.vector_store import VectorStoreManager
from models.llm_manager import get_llm, get_chat_model
from utils.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

# The prompt template is already in English.
PROMPT_TEMPLATE = """
**Instruction:** You are the "Zurich Policy Manager Assistant", an AI assistant specializing in insurance policies.
Your task is to answer the user's questions accurately and concisely, based EXCLUSIVELY on the following context extracted from the policy document.
The context has been carefully selected for its relevance.
Always cite the source (e.g., Page 5, Section: Eligibility) where you get the information, if it is available in the metadata.
If the information is not found in the provided context, state clearly: "I could not find information about that in the policy." Do not invent answers.

**Policy Context:**
{context}

**User's Question:**
{question}

**Assistant's Answer (accurate and based on context):**
"""


# Class for the local Cross-Encoder re-ranker (REFACTORED AND ROBUST VERSION)
class LocalCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses a local Cross-Encoder model to re-rank results."""
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    top_n: int = 4
    cross_encoder: Any = None  # Kept for Pydantic model definition

    def __init__(self, **data: Any):
        """Explicit initializer to ensure the cross-encoder is loaded."""
        super().__init__(**data)  # Calls the Pydantic base class initializer
        try:
            # Initializes the CrossEncoder model and assigns it to the instance attribute
            print(f"Initializing CrossEncoder: {self.model_name}...")
            self.cross_encoder = CrossEncoder(self.model_name, max_length=512)
        except Exception as e:
            print(f"Error initializing CrossEncoder: {e}")
            # Leave self.cross_encoder as None to fail safely if it doesn't load

    def compress_documents(
            self, documents: Sequence[Document], query: str, callbacks=None
    ) -> Sequence[Document]:
        if not documents:
            return []

        if self.cross_encoder is None:
            raise RuntimeError("CrossEncoder model is not initialized. Cannot perform re-ranking.")

        doc_list = list(documents)
        passages = [d.page_content for d in doc_list]

        print(f"Re-ranking {len(passages)} documents with CrossEncoder...")
        scores = self.cross_encoder.predict([(query, passage) for passage in passages], show_progress_bar=False)

        docs_with_scores = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in docs_with_scores[:self.top_n]]


def get_advanced_retriever(vector_store_manager: VectorStoreManager, model_key: str):
    """
    Creates and returns an advanced retriever that combines hybrid search and re-ranking.
    This component can be reused to build different types of chains.
    """
    print(f"Initializing advanced retriever with model {model_key}...")
    all_docs = vector_store_manager.get_all_documents()
    if not all_docs:
        print("WARNING: No documents found in the database. The retriever will not work.")
        return None

    # Keyword-based retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    # Semantic retriever (Vector-based)
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        client=vector_store_manager.client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    chroma_retriever = db.as_retriever(search_kwargs={"k": 10})

    # Hybrid Search with EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # --- ACCURACY ENHANCEMENT: MultiQueryRetriever ---
    # We use an LLM to rephrase the user's question into several questions from different perspectives.
    print("Initializing Multi-Query-Retriever to expand the user's question...")
    llm_for_query_expansion = get_chat_model(model_key)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, llm=llm_for_query_expansion
    )

    # Re-ranker with Cross-Encoder
    reranker = LocalCrossEncoderReranker(top_n=4)

    # Final retrieval pipeline with compression and re-ranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=multi_query_retriever  # We use the retriever that expands the question
    )
    print("✅ Advanced retriever initialized successfully.")
    return compression_retriever


def get_rag_chain(model_key: str, vector_store_manager: VectorStoreManager):
    """
    Configures and returns the RAG chain (RetrievalQA) for stateless question-answering.
    """
    print(f"Configuring RAG (QA) chain with model: {model_key}")

    advanced_retriever = get_advanced_retriever(vector_store_manager, model_key)
    if not advanced_retriever:
        return None

    llm = get_llm(model_key)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=advanced_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("✅ Advanced RAG (QA) chain configured successfully.")
    return qa_chain