import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# Clean imports that will now work because the directory names are correct.
from src.data_ingestion.pubmed_fetcher import get_paper_details
from src.processing.text_splitter import chunk_text_semantically


class ChromaManager:
    """A class to manage interactions with a ChromaDB vector store."""

    def __init__(self, path: str = "db/chroma_db", openai_api_key: str = None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings.")

        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        print(f"ChromaDB client initialized. Database path: {path}")

    def create_or_get_collection(self, name: str):
        print(f"Creating or getting collection: {name}")
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    chroma_manager = ChromaManager(openai_api_key=api_key)

    try:
        chroma_manager.client.delete_collection(name="pubmed_papers")
        print("Existing collection 'pubmed_papers' deleted for a clean run.")
    except Exception:
        print("Collection 'pubmed_papers' does not exist yet. Creating a new one.")

    collection = chroma_manager.create_or_get_collection("pubmed_papers")

    paper_id = "29276945"
    paper_data = get_paper_details(paper_id)

    if "error" not in paper_data and paper_data.get("abstract"):
        abstract = paper_data.get("abstract")
        chunks = chunk_text_semantically(abstract, api_key)

        documents_to_add = chunks
        metadatas_to_add = [{"pubmed_id": paper_id, "title": paper_data.get("title", ""), "chunk_number": i} for i in
                            range(len(chunks))]
        ids_to_add = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]

        print(f"\nAdding {len(documents_to_add)} chunks to the collection...")
        collection.add(documents=documents_to_add, metadatas=metadatas_to_add, ids=ids_to_add)
        print("Chunks added successfully.")

        count = collection.count()
        print(f"\nCollection now contains {count} documents.")

        retrieved_data = collection.get(ids=ids_to_add)
        print("\n--- Verification: Retrieved Data ---")
        print(retrieved_data['metadatas'])
    else:
        print(f"Failed to fetch paper data or abstract was empty for PubMed ID {paper_id}")