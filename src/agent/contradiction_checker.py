import os
import json
from dotenv import load_dotenv
from pydantic import ValidationError

# To run this file directly for testing, we need to handle the Python path.
if __name__ == '__main__':
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# We now import our Pydantic schema to enforce the structure of the LLM's output.
from src.agent.schemas import LLMResponse
from src.agent.llm_clients import OpenAIJSONClient
from src.retrieval.retriever import AdvancedRetriever
from src.verifier.heuristic_verifier import HeuristicContradictionVerifier
from src.verifier.verdict_arbitration import arbitrate_verdict
from src.vector_store.chroma_manager import ChromaManager
from src.data_ingestion.pubmed_fetcher import get_paper_details
from src.processing.text_splitter import chunk_text_semantically


def _format_context_for_prompt(retrieved_chunks: list[dict]) -> str:
    """A helper function to format the retrieved chunks into a clean string for the LLM."""
    if not retrieved_chunks:
        return "No relevant information was found for this paper."
    return "\n---\n".join([chunk['text'] for chunk in retrieved_chunks])


class ContradictionChecker:
    """
    The core RAG agent that orchestrates the retrieval and analysis of scientific papers.
    """

    def __init__(
        self,
        retriever: AdvancedRetriever,
        llm_client,
        *,
        torch_verifier=None,
        verifier_strategy: str = "llm_only",
        verifier_override_confidence: float = 0.65,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.heuristic_verifier = HeuristicContradictionVerifier()
        self.torch_verifier = torch_verifier
        self.verifier_strategy = verifier_strategy
        self.verifier_override_confidence = float(verifier_override_confidence)
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self):
        """Creates the master prompt for the LLM, instructing it to act as a researcher and return a structured JSON response."""
        # By providing the full JSON schema within the prompt, we drastically increase
        # the likelihood that the LLM's output will be perfectly structured and parsable.
        return """
        You are a meticulous biomedical researcher. Your task is to analyze key claims from text passages retrieved from two scientific abstracts.

        Based ONLY on the context provided in the "Retrieved Passages" for each paper below, perform the following actions:
        1.  Extract the single most important scientific claim from the passages for Paper 1. If no relevant claim can be found, you MUST return the string "No specific claim could be extracted from the provided text."
        2.  Extract the single most important scientific claim from the passages for Paper 2. If no relevant claim can be found, you MUST return the string "No specific claim could be extracted from the provided text."
        3.  Compare the two extracted claims. Classify their relationship as "Contradictory", "Supporting", or "Unrelated". If either claim could not be extracted, the verdict MUST be "Unrelated".
        4.  Provide a brief justification for your classification in one or two sentences.

        You MUST respond ONLY with a valid JSON object that strictly follows this schema:
        {
          "paper_1_claim": "Your extracted claim for Paper 1 or the specified failure message.",
          "paper_2_claim": "Your extracted claim for Paper 2 or the specified failure message.",
          "analysis": {
            "verdict": "Contradictory" | "Supporting" | "Unrelated",
            "justification": "Your reasoning here."
          }
        }
        """

    def check(self, paper_1_id: str, paper_2_id: str, topic: str) -> dict:
        """Performs the full RAG-based contradiction check with schema validation."""
        context_1_chunks = self.retriever.retrieve(query=topic, paper_id=paper_1_id)
        context_2_chunks = self.retriever.retrieve(query=topic, paper_id=paper_2_id)

        formatted_context_1 = _format_context_for_prompt(context_1_chunks)
        formatted_context_2 = _format_context_for_prompt(context_2_chunks)

        user_message = (
            f"**Topic of Interest:** {topic}\n\n"
            f"**Retrieved Passages from Paper 1 (ID: {paper_1_id}):**\n{formatted_context_1}\n\n"
            f"**Retrieved Passages from Paper 2 (ID: {paper_2_id}):**\n{formatted_context_2}"
        )

        try:
            llm_output, raw_response = self.llm_client.generate_json(
                system_prompt=self.system_prompt,
                user_message=user_message,
            )
            # This is the crucial validation step. It will raise a ValidationError
            # if the JSON from the LLM is malformed or missing fields.
            validated_data = LLMResponse.model_validate(llm_output)

            # Convert the validated Pydantic model back to a dict for the rest of the app.
            analysis_result = validated_data.model_dump()
            baseline = self.heuristic_verifier.predict(
                analysis_result["paper_1_claim"],
                analysis_result["paper_2_claim"],
            )
            torch_prediction = None
            if self.torch_verifier is not None:
                torch_prediction = self.torch_verifier.predict(
                    analysis_result["paper_1_claim"],
                    analysis_result["paper_2_claim"],
                )

            llm_verdict = analysis_result["analysis"]["verdict"]
            final_verdict, arbitration = arbitrate_verdict(
                llm_verdict=llm_verdict,
                verifier_prediction=torch_prediction,
                strategy=self.verifier_strategy,
                override_confidence=self.verifier_override_confidence,
            )
            analysis_result["analysis"]["llm_verdict"] = llm_verdict
            analysis_result["analysis"]["verdict"] = final_verdict
            analysis_result["analysis"]["verdict_source"] = arbitration["selected_source"]

            analysis_result['evidence_paper_1'] = [chunk['text'] for chunk in context_1_chunks]
            analysis_result['evidence_paper_2'] = [chunk['text'] for chunk in context_2_chunks]
            analysis_result['baseline_verifier'] = baseline
            analysis_result['torch_verifier'] = torch_prediction
            analysis_result['verifier_arbitration'] = arbitration
            return analysis_result
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            # We now catch both JSON parsing errors and Pydantic validation errors.
            return {
                "error": f"LLM output failed validation: {e}",
                "raw_response": raw_response if "raw_response" in locals() else None,
            }


# The __main__ block for testing remains the same.
if __name__ == '__main__':
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not (openai_key and cohere_key):
        raise ValueError("API keys must be set for the test run.")

    print("--- Setting up test database ---")
    chroma_manager = ChromaManager(openai_api_key=openai_key)
    try:
        chroma_manager.client.delete_collection(name="pubmed_papers")
    except Exception:
        pass
    collection = chroma_manager.create_or_get_collection("pubmed_papers")

    paper1_id = "34292771"
    paper2_id = "35852509"

    for paper_id in [paper1_id, paper2_id]:
        paper_data = get_paper_details(paper_id)
        if "error" not in paper_data and paper_data.get("abstract"):
            chunks = chunk_text_semantically(paper_data["abstract"], openai_key)
            metadatas = [{"pubmed_id": paper_id, "title": paper_data["title"]} for _ in chunks]
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print("Test data ingestion complete.")

    print("\n--- Running full RAG contradiction check ---")
    retriever = AdvancedRetriever(chroma_manager=chroma_manager, cohere_api_key=cohere_key)
    llm = OpenAIJSONClient(api_key=openai_key, model="gpt-4o")
    checker = ContradictionChecker(retriever=retriever, llm_client=llm)

    analysis_result = checker.check(
        paper_1_id=paper1_id,
        paper_2_id=paper2_id,
        topic="Does Ivermectin reduce mortality in patients with COVID-19?"
    )

    print("\n--- Final RAG Analysis Output ---")
    print(json.dumps(analysis_result, indent=2))
