import cohere
import os
from dotenv import load_dotenv

# Python path
if __name__ == '__main__':
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vector_store.chroma_manager import ChromaManager


class AdvancedRetriever:
    """
    Implements a two-stage retrieval process with metadata filtering.

    This retriever is designed for accuracy by first using an efficient vector search
    to find candidate documents and then applying a more powerful, computationally
    intensive re-ranking model to find the true best matches.
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        cohere_api_key: str | None,
        cohere_model: str = "rerank-english-v3.0",
    ):
        self.chroma_manager = chroma_manager
        self.collection = self.chroma_manager.create_or_get_collection("pubmed_papers")
        self.cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None
        self.cohere_model = cohere_model

    def retrieve(self, query: str, paper_id: str, top_n_candidates: int = 10, top_n_reranked: int = 3) -> list[dict]:
        """
        Performs retrieval and re-ranking for a specific paper.

        The `where` clause is a powerful feature of ChromaDB that allows us to
        pre-filter the search space. This is much more efficient than retrieving
        documents and then filtering them in Python.

        Args:
            query: The user's search query (e.g., the research topic).
            paper_id: The PubMed ID to restrict the search to.
            top_n_candidates: How many initial candidates to pull from the vector store.
            top_n_reranked: The final number of documents to return after re-ranking.

        Returns:
            A sorted list of the most relevant document chunks.
        """
        # Stage 1: Fast candidate retrieval from Chroma, filtered by paper_id.
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n_candidates,
            where={"pubmed_id": paper_id}
        )

        candidate_docs = results['documents'][0]
        if not candidate_docs:
            return []

        candidate_metadatas = results['metadatas'][0]

        if self.cohere_client is None:
            return [
                {
                    "text": candidate_docs[i],
                    "metadata": candidate_metadatas[i],
                    "relevance_score": None,
                }
                for i in range(min(int(top_n_reranked), len(candidate_docs)))
            ]

        # Stage 2: Use Cohere's powerful re-ranker to find the best matches.
        # This step is slower but much more accurate than vector similarity alone.
        reranked_results = self.cohere_client.rerank(
            model=self.cohere_model,
            query=query,
            documents=candidate_docs,
            top_n=int(top_n_reranked)
        )

        # Finally, we combine the re-ranked results with their original metadata
        # to pass a clean, structured output to the next part of the pipeline.
        final_results = []
        for hit in reranked_results.results:
            original_index = hit.index
            final_results.append({
                'text': candidate_docs[original_index],
                'metadata': candidate_metadatas[original_index],
                'relevance_score': hit.relevance_score
            })

        return final_results


# This block is for direct script testing and demonstration of the retriever's capabilities.
if __name__ == '__main__':
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not (openai_key and cohere_key):
        raise ValueError("API keys for OpenAI and Cohere must be set in the .env file.")


    from src.data_ingestion.pubmed_fetcher import get_paper_details
    from src.processing.text_splitter import chunk_text_semantically

    # 1. Setup: Ingest data from two different papers into a fresh collection.
    print("--- Setting up test database ---")
    chroma_manager = ChromaManager(openai_api_key=openai_key)
    try:
        chroma_manager.client.delete_collection(name="pubmed_papers")
    except Exception:
        pass  # Collection didn't exist, which is fine.

    collection = chroma_manager.create_or_get_collection("pubmed_papers")

    paper_id_1 = "29276945"  # Coffee and CVD
    paper_id_2 = "31405892"  # Vitamin D and mortality

    for paper_id in [paper_id_1, paper_id_2]:
        paper_data = get_paper_details(paper_id)
        if "error" not in paper_data and paper_data.get("abstract"):
            chunks = chunk_text_semantically(paper_data["abstract"], openai_key)
            metadatas = [{"pubmed_id": paper_id, "title": paper_data["title"]} for _ in chunks]
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)

    print(f"Database setup complete. Total documents: {collection.count()}")

    # 2. Execution: Initialize the retriever and perform a filtered search.
    print("\n--- Testing filtered retrieval ---")
    retriever = AdvancedRetriever(chroma_manager=chroma_manager, cohere_api_key=cohere_key)

    test_query = "What are the effects of caffeine on the heart?"

    # We are ONLY searching within the coffee paper.
    final_documents = retriever.retrieve(query=test_query, paper_id=paper_id_1)

    # 3. Verification: Print the results.
    print(f"\n--- Final Search Results for query within Paper {paper_id_1} ---")
    for doc in final_documents:
        print(f"Score: {doc['relevance_score']:.4f}")
        print(f"Text: {doc['text'][:100]}...")
        print(f"Source ID: {doc['metadata']['pubmed_id']}")
        print("-" * 30)
