from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os


def chunk_text_semantically(text: str, openai_api_key: str) -> list[str]:
    """
    Splits text into semantically coherent chunks using an embedding model.

    This advanced splitting method is preferred over simple character-based chunking
    because it preserves the contextual integrity of the text, which is crucial
    for the accuracy of the downstream RAG pipeline.

    Args:
        text: The text content to be chunked.
        openai_api_key: Your OpenAI API key.

    Returns:
        A list of text chunks.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    # Using the "percentile" threshold is a more adaptive strategy than a fixed
    # similarity score. It determines splits based on statistical outliers in
    # sentence similarity, making it robust to different text styles.
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    docs = text_splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]
    return chunks


# This block is for direct script testing and demonstration.
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the .env file for this demo.")

    sample_abstract = (
        "Coffee is one of the most widely consumed beverages, and some studies have suggested "
        "it may be related to cardiovascular disease (CVD), the leading cause of poor health in the world. "
        "This review evaluates the evidence on the effect of habitual coffee consumption on CVD incidence and mortality."
    )

    print("--- Original Text ---")
    print(sample_abstract)

    semantic_chunks = chunk_text_semantically(sample_abstract, api_key)

    print(f"\n--- Found {len(semantic_chunks)} Semantic Chunks ---")
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\n[Chunk {i}]")
        print(chunk)