import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.retriever import AdvancedRetriever
from src.vector_store.chroma_manager import ChromaManager
from src.data_ingestion.pubmed_fetcher import get_paper_details


class ContradictionChecker:
    """The core agent that orchestrates the retrieval and analysis of scientific papers."""

    def __init__(self, retriever: AdvancedRetriever, openai_api_key: str):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=openai_api_key)
        print("ContradictionChecker initialized.")

    def _get_system_prompt(self):
        return """
        You are a meticulous biomedical researcher. Your task is to analyze the key claims from two provided scientific abstracts regarding a specific topic.
        Based on the context from Paper 1 and Paper 2, perform the following actions:
        1. Extract Key Claim from Paper 1: Identify and state the single most important scientific claim made in the text from Paper 1.
        2. Extract Key Claim from Paper 2: Identify and state the single most important scientific claim made in the text from Paper 2.
        3. Analyze Claims: Compare the two claims. Determine if they are: Contradictory, Supporting, or Unrelated.
        4. Provide Justification: Briefly explain your reasoning for the analysis in one or two sentences.
        Respond ONLY with a valid JSON object in the following format:
        {
          "paper_1_claim": "The central claim from Paper 1's text.",
          "paper_2_claim": "The central claim from Paper 2's text.",
          "analysis": {
            "verdict": "Contradictory" | "Supporting" | "Unrelated",
            "justification": "Your reasoning here."
          }
        }
        """

    def check(self, paper_1_id: str, paper_2_id: str, topic: str) -> dict:
        print(f"\n--- Starting Contradiction Check ---")
        print(f"Paper 1: {paper_1_id}, Paper 2: {paper_2_id}")
        print(f"Topic: {topic}")

        paper_1_info = get_paper_details(paper_1_id)
        paper_2_info = get_paper_details(paper_2_id)

        if "error" in paper_1_info or "error" in paper_2_info or not paper_1_info.get(
                "abstract") or not paper_2_info.get("abstract"):
            return {"error": "Could not fetch details or abstract for one or both papers."}

        context_1_text = paper_1_info.get("abstract")
        context_2_text = paper_2_info.get("abstract")

        user_message = f"**Topic of Interest:** {topic}\n\n**Paper 1 (ID: {paper_1_id}) Abstract:**\n{context_1_text}\n\n**Paper 2 (ID: {paper_2_id}) Abstract:**\n{context_2_text}"

        print("Sending request to OpenAI for analysis...")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": f"Failed to parse LLM response: {e}", "raw_response": response.choices[0].message.content}


if __name__ == '__main__':
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not (openai_key and cohere_key):
        raise ValueError("API keys for OpenAI and Cohere must be set in .env file.")

    chroma_manager = ChromaManager(openai_api_key=openai_key)
    retriever = AdvancedRetriever(chroma_manager=chroma_manager, cohere_api_key=cohere_key)
    checker = ContradictionChecker(retriever=retriever, openai_api_key=openai_key)

    # --- THE FINAL, TRIPLE-VERIFIED, CORRECT PAPER IDs ---
    # A large consortium study finding both low and high Vitamin D levels are associated with higher mortality.
    paper1 = "25023936"
    # A meta-analysis of RCTs finding no significant effect of supplementation on mortality.
    paper2 = "31405892"

    analysis_result = checker.check(
        paper_1_id=paper1,
        paper_2_id=paper2,
        topic="What is the relationship between Vitamin D levels/supplementation and total mortality?"
    )

    print("\n--- Analysis Complete ---")
    print(json.dumps(analysis_result, indent=2))