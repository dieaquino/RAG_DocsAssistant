# rag/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from utils.config import CHROMA_PATH, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME


class VectorStoreManager:
    def __init__(self):
        """
        Initializes the vector store manager.
        It now creates and maintains a LangChain-compatible vector_store object.
        """
        # 1. Initialize the embedding function that LangChain will use
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}  # You can change this to 'cuda' if you have a GPU
        )

        # 2. Configure the ChromaDB client (to disable telemetry)
        client_settings = Settings(anonymized_telemetry=False, is_persistent=True)

        # We save the client as a private attribute to be used in other modules
        self._client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=client_settings)

        # 3. Create the LangChain Chroma object, which will manage the collection
        # We use the client we just saved
        self.vector_store = Chroma(
            client=self._client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
        )

        print("VectorStoreManager initialized successfully.")
        print(f"Collection '{CHROMA_COLLECTION_NAME}' loaded with {self.vector_store._collection.count()} documents.")

    @property
    def client(self) -> chromadb.Client:
        """Property that safely exposes the underlying ChromaDB client."""
        return self._client

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the collection using the LangChain method.
        This greatly simplifies the previous logic.
        """
        if not documents:
            print("No documents to add.")
            return

        print(f"Adding {len(documents)} chunks to the collection...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully.")

    def get_collection_status(self):
        """Returns the current status of the collection."""
        # We access the underlying Chroma collection through the LangChain object
        count = self.vector_store._collection.count()
        return {
            "name": self.vector_store._collection.name,
            "count": count
        }

    def get_all_documents(self) -> List[Document]:
        """Retrieves all documents from the collection."""
        print("Retrieving all documents from the vector database...")
        # The get() method without filters returns everything
        all_data = self.vector_store.get(include=["metadatas", "documents"])

        docs = []
        for i, content in enumerate(all_data.get('documents', [])):
            doc = Document(
                page_content=content,
                metadata=all_data['metadatas'][i]
            )
            docs.append(doc)

        print(f"{len(docs)} documents retrieved.")
        return docs