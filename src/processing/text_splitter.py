from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os


def chunk_text_semantically(text: str, openai_api_key: str) -> list[str]:
    """
    Splits a block of text into semantically coherent chunks.

    This function uses LangChain's SemanticChunker, which groups related sentences
    together into larger chunks, preserving the context and meaning.

    Args:
        text (str): The text content to be chunked (e.g., a paper's abstract).
        openai_api_key (str): Your OpenAI API key.

    Returns:
        list[str]: A list of text chunks.
    """
    # Initialize the OpenAI embeddings model. This is used by the chunker
    # to understand the semantic meaning of sentences.
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize the SemanticChunker.
    # The 'breakpoint_threshold_type="percentile"' tells the chunker to decide
    # when to split based on the distribution of semantic similarity scores.
    # This is a more robust method than using a fixed similarity score.
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    # The 'create_documents' method takes a list of texts. Here, we just have one.
    docs = text_splitter.create_documents([text])

    # Extract the page_content from each created document.
    chunks = [doc.page_content for doc in docs]

    return chunks


# Example of how to run this script directly for testing
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    # You must have your OPENAI_API_KEY in the .env file for this to work
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    # Example abstract text from our previous step
    sample_abstract = """
    Coffee is one of the most widely consumed beverages, and some studies have suggested it may be related to cardiovascular disease (CVD), the leading cause of poor health in the world. This review evaluates the evidence on the effect of habitual coffee consumption on CVD incidence and mortality. The review is based mostly on observational studies and meta-analyses of the literature. In healthy people, in comparison to not consuming coffee, habitual consumption of 3-5 cups of coffee per day is associated with a 15% reduction in the risk of CVD, and higher consumption has not been linked to elevated CVD risk. Moreover, in comparison to no coffee intake, usual consumption of 1-5 cups/day is associated with a lower risk of death. In people who have already suffered a CVD event, habitual consumption does not increase the risk of a recurrent CVD or death. However, hypertensive patients with uncontrolled blood pressure should avoid consuming large doses of caffeine. In persons with well-controlled blood pressure, coffee consumption is probably safe, but this hypothesis should be confirmed by further investigations.
    """

    print("--- Original Abstract ---")
    print(sample_abstract)
    print("\n" + "=" * 50 + "\n")

    # Get the semantic chunks
    semantic_chunks = chunk_text_semantically(sample_abstract, api_key)

    print("--- Semantic Chunks ---")
    for i, chunk in enumerate(semantic_chunks):
        print(f"Chunk {i + 1}:")
        print(chunk)
        print("-" * 20)

    print(f"\nSuccessfully split the abstract into {len(semantic_chunks)} semantic chunks.")