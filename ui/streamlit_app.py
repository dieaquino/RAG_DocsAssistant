# ui/streamlit_app.py
import streamlit as st
import os
import time

# --- START: Add project root to Python path ---
import sys
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END ---

# --- APPLICATION IMPORTS ---
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Import the correct function for chat models
from models.llm_manager import get_chat_model
from rag.retrieval import get_advanced_retriever
from utils.config import APP_TITLE, PAGE_ICON, OLLAMA_MODELS, DEFAULT_MODEL, DOCS_DIR
from utils.document_processor import process_pdf
from rag.vector_store import VectorStoreManager

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {APP_TITLE}")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = VectorStoreManager()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose an LLM model:", options=list(OLLAMA_MODELS.keys()), key="model_selector"
    )
    if st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.rag_chain = None # Force chain restart
        st.success(f"Model changed to {selected_model}. The assistant will restart.")
        st.rerun()

    st.markdown("---")
    st.header("🗂️ Policy Status")
    try:
        status = st.session_state.vector_store_manager.get_collection_status()
        st.metric(label="Chunks in DB", value=status.get('count', 0))
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")

# --- UI TABS ---
tab1, tab2 = st.tabs(["💬 Policy Chat", "📄 Document Manager"])

# --- TAB 1: POLICY CHAT ---
with tab1:
    st.header("Chat with your Policy")

    # Initialize the RAG chain if it doesn't exist
    if st.session_state.rag_chain is None:
        with st.spinner(f"Initializing advanced assistant with the {st.session_state.selected_model} model..."):
            advanced_retriever = get_advanced_retriever(st.session_state.vector_store_manager, st.session_state.selected_model)
            if advanced_retriever:
                # Use the chat model, which is more suitable for conversation
                llm = get_chat_model(st.session_state.selected_model)
                st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=advanced_retriever,
                    memory=st.session_state.memory,
                    return_source_documents=True,
                )
                st.success("Advanced assistant ready!")
            else:
                st.error("Could not initialize the assistant. Please upload and process a document in the 'Document Manager' tab.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What do you want to know about your policy?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                with st.spinner("Expanding query, searching, re-ranking, and generating response..."):
                    response = st.session_state.rag_chain.invoke({"question": prompt})
                    full_response = response.get('answer', "I could not get a response from the model.")
                    st.markdown(full_response)
            else:
                full_response = "The assistant is not ready. Please ensure a document has been processed."
                st.warning(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- TAB 2: DOCUMENT MANAGER ---
with tab2:
    st.header("Policy Management")
    st.markdown("Upload a new policy document in PDF format. This will replace the current database.")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_file = st.file_uploader("Upload your policy (PDF)", type="pdf")

    if uploaded_file is not None:
        file_path = DOCS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        if st.button("Process and Vectorize Document"):
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                # You could add logic here to clear the previous DB if desired
                doc_chunks = process_pdf(str(file_path))
                if doc_chunks:
                    st.session_state.vector_store_manager.add_documents(doc_chunks)
                    # Force the chain and conversation to restart
                    st.session_state.rag_chain = None
                    st.session_state.messages = []
                    st.session_state.memory.clear()
                    st.success("Document processed! The assistant has been reset with the new knowledge.")
                    st.rerun()
                else:
                    st.error("Could not extract any content from the document.")

    st.markdown("---")
    st.subheader("Processed Documents")
    processed_files = [f.name for f in DOCS_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
    if processed_files:
        st.table(processed_files)
    else:
        st.info("No documents have been uploaded yet.")