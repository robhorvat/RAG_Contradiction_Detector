import cohere
import os
from dotenv import load_dotenv

from src.vector_store.chroma_manager import ChromaManager


class AdvancedRetriever:
    """
    Implements a two-stage retrieval process for enhanced accuracy.

    Stage 1: Fast candidate retrieval from ChromaDB using vector similarity.
    Stage 2: Sophisticated re-ranking of candidates using Cohere's Re-rank model
             to find the most contextually relevant results.
    """

    def __init__(self, chroma_manager: ChromaManager, cohere_api_key: str):
        self.chroma_manager = chroma_manager
        self.collection = self.chroma_manager.create_or_get_collection("pubmed_papers")
        self.cohere_client = cohere.Client(cohere_api_key)

    def retrieve(self, query: str, top_n_candidates: int = 10, top_n_reranked: int = 3) -> list[dict]:
        """
        Performs candidate retrieval and re-ranking.

        Args:
            query: The user's search query.
            top_n_candidates: Number of documents to fetch from ChromaDB.
            top_n_reranked: Final number of documents to return after re-ranking.

        Returns:
            A list of the top re-ranked documents with their relevance scores.
        """
        # Stage 1: Retrieve a broad set of candidate documents from the vector store.
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n_candidates
        )

        candidate_docs = results['documents'][0]
        if not candidate_docs:
            return []

        candidate_metadatas = results['metadatas'][0]

        # Stage 2: Use a more powerful model to re-rank the candidates for relevance.
        reranked_results = self.cohere_client.rerank(
            model='rerank-english-v3.0',
            query=query,
            documents=candidate_docs,
            top_n=top_n_reranked
        )

        # Combine the re-ranked results with their original text and metadata.
        final_results = []
        for hit in reranked_results.results:
            original_index = hit.index
            final_results.append({
                'text': candidate_docs[original_index],
                'metadata': candidate_metadatas[original_index],
                'relevance_score': hit.relevance_score
            })

        return final_results


# Example of how to run this script directly for testing
if __name__ == '__main__':
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not (openai_key and cohere_key):
        raise ValueError("API keys for OpenAI and Cohere must be set in .env file.")

    # Initialize our database manager
    chroma_manager = ChromaManager(openai_api_key=openai_key)

    # For this test to work, the DB must have some data.
    # Let's add two papers' abstracts.
    from src.data_ingestion.pubmed_fetcher import get_paper_details
    from src.processing.text_splitter import chunk_text_semantically

    paper_id_1 = "29276945"  # Coffee and CVD
    paper_id_2 = "12032036"  # Epinephrine in surgery

    # --- DATA PREPARATION PHASE ---
    # Clear the collection for a clean run
    try:
        chroma_manager.client.delete_collection(name="pubmed_papers")
        print("Deleted existing collection.")
    except Exception:
        print("Collection does not exist. Creating a new one.")
    collection = chroma_manager.create_or_get_collection("pubmed_papers")

    all_chunks, all_metadatas, all_ids = [], [], []

    for paper_id in [paper_id_1, paper_id_2]:
        print(f"\nProcessing paper: {paper_id}")
        paper_data = get_paper_details(paper_id)
        if "error" not in paper_data and paper_data.get("abstract"):
            chunks = chunk_text_semantically(paper_data["abstract"], openai_key)
            metadatas = [{"pubmed_id": paper_id, "title": paper_data["title"], "chunk_num": i} for i, chunk in
                         enumerate(chunks)]
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
            all_chunks.extend(chunks), all_metadatas.extend(metadatas), all_ids.extend(ids)

    if all_chunks:
        collection.add(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
        print(f"\nAdded a total of {len(all_chunks)} chunks to the database.")

    # --- RETRIEVAL PHASE ---
    # ** THE FIX: Initialize the retriever AFTER the database is ready **
    retriever = AdvancedRetriever(chroma_manager=chroma_manager, cohere_api_key=cohere_key)

    # Now, perform a search
    test_query = "What are the effects of caffeine on the heart?"

    final_documents = retriever.retrieve(test_query)

    print("\n--- Final Search Results ---")
    for doc in final_documents:
        print(f"Score: {doc['relevance_score']:.4f}")
        print(f"Text: {doc['text']}")
        print(f"Source: {doc['metadata']['title']} (ID: {doc['metadata']['pubmed_id']})")
        print("-" * 30)