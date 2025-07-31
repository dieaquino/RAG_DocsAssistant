# simple_demo.py
import sys
import os

# --- START: Add project root to Python path ---
# This is crucial for absolute imports to work when running the script directly.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END ---

from utils.document_processor import process_pdf
from rag.vector_store import VectorStoreManager
from rag.retrieval import get_rag_chain  # <-- IMPORTANT! We are using the advanced chain
from utils.config import DOCS_DIR, DEFAULT_MODEL


def run_advanced_demo():
    """
    Runs a console demo using the advanced RAG pipeline
    with hybrid search and re-ranking.
    """
    print("--- STARTING ZURICH POLICY ASSISTANT ADVANCED CONSOLE DEMO ---")

    # 1. Initialize the vector store manager
    vector_store_manager = VectorStoreManager()

    # 2. Check and process documents if the database is empty
    try:
        collection_status = vector_store_manager.get_collection_status()
        if collection_status.get('count', 0) == 0:
            print("The vector database is empty. Processing documents...")
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            policy_files = [f for f in DOCS_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
            if not policy_files:
                print(f"\nERROR: No PDF files found in '{DOCS_DIR}'.")
                return

            pdf_path = policy_files[0]
            print(f"Processing document: {pdf_path.name}")
            doc_chunks = process_pdf(str(pdf_path))
            if doc_chunks:
                vector_store_manager.add_documents(doc_chunks)
            else:
                print("Could not process documents. Exiting.")
                return
    except Exception as e:
        print(f"\nERROR: Could not initialize the vector database: {e}")
        return

    # 3. Configure the ADVANCED RAG chain
    print("\nConfiguring advanced RAG chain (Hybrid Search + Re-ranker)...")
    rag_chain = get_rag_chain(
        model_key=DEFAULT_MODEL,
        vector_store_manager=vector_store_manager
    )

    if not rag_chain:
        print("Failed to initialize RAG chain. Exiting.")
        return

    # 4. Question and answer loop
    print("\n✅ Advanced Policy Assistant ready. Type 'exit' to finish.")
    while True:
        try:
            query = input("\nAsk about your policy: ")
            if query.lower().strip() == 'exit':
                break
            if not query.strip():
                continue

            print("\nExpanding query, searching, re-ranking, and generating response...")
            # The RetrievalQA chain expects the 'query' key
            response = rag_chain.invoke({"query": query})

            print("\n--- Assistant's Response ---")
            # The response key is 'result' in RetrievalQA
            print(response['result'].strip())

            sources = response.get('source_documents', [])
            if sources:
                print("\n--- Sources (Re-ranked and verified) ---")
                unique_sources = set()
                for doc in sources:
                    meta = doc.metadata
                    # Build a more detailed source string
                    source_str = f"Page {meta.get('page', 'N/A')}, Section: {meta.get('section', 'N/A')}"
                    unique_sources.add(source_str)

                for source in sorted(list(unique_sources)):
                    print(f"- {source}")
            print("-" * 25)

        except (KeyboardInterrupt, EOFError):
            break

    print("\n--- DEMO FINISHED ---")


if __name__ == "__main__":
    run_advanced_demo()