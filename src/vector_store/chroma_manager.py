import chromadb
from chromadb.utils import embedding_functions
import os

# To run this file directly for testing, we need to handle the Python path.
if __name__ == '__main__':
    import sys
    import json
    from dotenv import load_dotenv

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data_ingestion.pubmed_fetcher import get_paper_details
    from src.processing.text_splitter import chunk_text_semantically


class ChromaManager:
    """
    A manager class for handling all interactions with a ChromaDB vector store.

    This class encapsulates the setup of the ChromaDB client, the embedding
    function, and common database operations, providing a clean interface for
    the rest of the application.
    """

    def __init__(self, path: str = "db/chroma_db", openai_api_key: str = None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for ChromaManager.")

        # Using PersistentClient ensures that our database is saved to disk.
        self.client = chromadb.PersistentClient(path=path)

        # Define the embedding function that Chroma will use to vectorize documents.
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )

    def create_or_get_collection(self, name: str):
        """Creates a new collection or gets it if it already exists."""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )

    def check_documents_exist(self, collection, doc_ids: list[str]) -> set[str]:
        """
        Checks which documents from a list of IDs already exist in the collection.

        This is a crucial helper for our "just-in-time" ingestion strategy.
        It allows the app to avoid re-processing papers that are already in the database.
        It returns a set for efficient membership checking.

        Args:
            collection: The ChromaDB collection object to check against.
            doc_ids: A list of document IDs to check for.

        Returns:
            A set containing only the IDs that were found in the collection.
        """
        if not doc_ids:
            return set()

        # The `get` method is an efficient way to check for existence by ID.
        # It will only return the documents that it finds.
        existing_docs = collection.get(ids=doc_ids)
        return set(existing_docs['ids'])


# This block is for direct script testing and demonstration.
if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print("--- Running ChromaManager self-test ---")
    manager = ChromaManager(openai_api_key=api_key)

    # For a clean test, we delete the collection if it exists.
    try:
        manager.client.delete_collection(name="pubmed_papers_test")
    except Exception:
        pass  # Collection didn't exist, which is fine.

    collection = manager.create_or_get_collection("pubmed_papers_test")

    # 1. Test adding a document
    test_ids = ["test-doc-0", "test-doc-1"]
    collection.add(documents=["This is a test", "This is another test"], ids=test_ids)
    print(f"Successfully added {collection.count()} documents to the test collection.")

    # 2. Test the existence check
    ids_to_check = ["test-doc-0", "test-doc-99"]  # One exists, one does not
    found_ids = manager.check_documents_exist(collection, ids_to_check)

    print(f"\nChecking for IDs: {ids_to_check}")
    print(f"Found IDs: {found_ids}")
    assert "test-doc-0" in found_ids
    assert "test-doc-99" not in found_ids
    print("Document existence check passed successfully.")