import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# These imports are only for the __main__ block.
# Keeping them here is fine for a self-contained module.
from src.data_ingestion.pubmed_fetcher import get_paper_details
from src.processing.text_splitter import chunk_text_semantically


class ChromaManager:
    """
    A manager class for handling all interactions with a ChromaDB vector store.

    This class encapsulates the setup of the ChromaDB client, the embedding
    function, and the creation/retrieval of collections, providing a clean
    interface for the rest of the application.
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


# This block is for direct script testing and demonstration.
if __name__ == '__main__':
    import json

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print("Running ChromaManager self-test...")
    manager = ChromaManager(openai_api_key=api_key)

    # For a clean test, we delete the collection if it exists.
    try:
        manager.client.delete_collection(name="pubmed_papers_test")
    except ValueError:
        pass  # Collection didn't exist, which is fine.

    collection = manager.create_or_get_collection("pubmed_papers_test")

    # Test adding a single paper's chunks
    paper_id = "29276945"
    paper_data = get_paper_details(paper_id)

    if "error" not in paper_data and paper_data.get("abstract"):
        chunks = chunk_text_semantically(paper_data["abstract"], api_key)
        ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]

        collection.add(documents=chunks, ids=ids)
        print(f"Successfully added {collection.count()} documents to the test collection.")
    else:
        print("Failed to fetch or process paper for the test.")