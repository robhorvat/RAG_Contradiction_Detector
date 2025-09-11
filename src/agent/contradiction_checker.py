import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.retriever import AdvancedRetriever
from src.vector_store.chroma_manager import ChromaManager
from src.data_ingestion.pubmed_fetcher import get_paper_details


class ContradictionChecker:
    """
    The core agent that orchestrates the process of identifying contradictions.

    This class leverages a retriever to find relevant information and an LLM to
    perform the nuanced task of claim extraction and comparison.
    """

    def __init__(self, retriever: AdvancedRetriever, openai_api_key: str):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=openai_api_key)
        # The system prompt is a constant, so we can define it once.
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self):
        """
        Creates the master prompt for the LLM, instructing it to act as a
        researcher and return a structured JSON response.
        """
        # Using a multi-line string with clear instructions and a defined
        # JSON schema is a key part of reliable prompt engineering.
        return """
        You are a meticulous biomedical researcher. Your task is to analyze the key scientific claims from two provided abstracts on a specific topic.

        1.  From the text of Paper 1, extract its single most important scientific claim.
        2.  From the text of Paper 2, extract its single most important scientific claim.
        3.  Compare the two claims. Classify their relationship as "Contradictory", "Supporting", or "Unrelated".
        4.  Provide a brief justification for your classification in one or two sentences.

        Respond ONLY with a valid JSON object matching this schema:
        {
          "paper_1_claim": "The central claim from Paper 1.",
          "paper_2_claim": "The central claim from Paper 2.",
          "analysis": {
            "verdict": "Contradictory" | "Supporting" | "Unrelated",
            "justification": "Your reasoning here."
          }
        }
        """

    def check(self, paper_1_id: str, paper_2_id: str, topic: str) -> dict:
        """
        Performs the full contradiction check for two papers.

        Args:
            paper_1_id: The PubMed ID of the first paper.
            paper_2_id: The PubMed ID of the second paper.
            topic: The topic to focus the analysis on.

        Returns:
            A dictionary containing the structured analysis from the LLM.
        """
        paper_1_info = get_paper_details(paper_1_id)
        paper_2_info = get_paper_details(paper_2_id)

        # Robustness check: Ensure we have abstracts to analyze.
        if "error" in paper_1_info or not paper_1_info.get("abstract"):
            return {"error": f"Could not retrieve a valid abstract for Paper 1 (ID: {paper_1_id})."}
        if "error" in paper_2_info or not paper_2_info.get("abstract"):
            return {"error": f"Could not retrieve a valid abstract for Paper 2 (ID: {paper_2_id})."}

        user_message = (
            f"**Topic of Interest:** {topic}\n\n"
            f"**Paper 1 (ID: {paper_1_id}) Abstract:**\n{paper_1_info['abstract']}\n\n"
            f"**Paper 2 (ID: {paper_2_id}) Abstract:**\n{paper_2_info['abstract']}"
        )

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # Enable OpenAI's JSON mode
        )

        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": f"Failed to parse LLM JSON response: {e}",
                    "raw_response": response.choices[0].message.content}



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