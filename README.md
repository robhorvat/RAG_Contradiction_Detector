# 🔎 Biomedical Contradiction Detector

This project is an AI-powered agent designed to tackle a critical challenge for researchers: identifying contradictory claims in scientific literature. It uses a robust, evidence-grounded Retrieval-Augmented Generation (RAG) pipeline to analyze and compare any two papers from PubMed.

This repository is the result of a rigorous development process that went far beyond a simple tutorial. It involved iterative debugging of a complex software stack, hardening the system against inconsistent real-world data, and refactoring the core logic to meet production-grade standards.

---

## 🚀 Live Demo

![Contradiction Detector Demo GIF](https://github.com/RomanceLIVE/RAG_Contradiction_Detector/raw/main/RAG_Demo.gif)

---

## The Problem

The volume of biomedical research is staggering. For any given topic, there are often dozens of studies with conflicting findings. Manually tracking these is a massive bottleneck. This tool was built to automate the first-pass analysis, using an AI agent to flag potential contradictions and, crucially, to show the exact evidence for its reasoning.

## My Approach: A Production-Minded RAG System

I designed this project to be a robust prototype, focusing on engineering decisions that a senior developer would make.

*   **Evidence-Grounded Reasoning:** The core principle is that the LLM's final verdict **must** be based on retrieved text, not its own internal knowledge. The final version of this app implements a true RAG pipeline where only the most relevant, re-ranked passages from each paper are sent to the LLM for analysis.

*   **Building for Real-World Data:** The initial PubMed data fetching was fragile. I re-engineered it to be resilient to PubMed's varied XML schemas and to gracefully handle entries with missing abstracts—a common real-world data problem.

*   **A Dynamic "Just-in-Time" Database:** The application doesn't require a pre-populated database. When a user enters new PubMed IDs, the system automatically fetches the abstracts, performs semantic chunking, and ingests them into the local ChromaDB vector store on the fly.

*   **High-Fidelity Retrieval:** A simple vector search is not enough. This pipeline uses a two-stage process for accuracy:
    1.  **Semantic Chunking & Candidate Retrieval:** Abstracts are split into contextually-aware chunks and retrieved from **ChromaDB**.
    2.  **High-Relevance Re-ranking:** **Cohere's** powerful re-ranker is used to ensure only the most relevant passages are passed to the agent, filtering out noise.

*   **Trust and Transparency in the UI:** The Streamlit UI was designed to build user trust. It doesn't just show a verdict; it includes expandable sections that display the **exact evidence passages** the agent used for its analysis, making the reasoning process transparent.

## Key Learnings from Development

*   **The Importance of a Stateless Architecture:** Early versions of the Streamlit app suffered from "zombie" database connections due to a conflict between caching and file deletion. I re-architected the app to be stateless on startup, guaranteeing a clean, predictable database state for every session.
*   **Data Curation is King:** The biggest challenge was not the code, but the data. My AI mentor repeatedly provided faulty PubMed IDs. The robustness of the final application was proven by its ability to correctly analyze this "garbage" data and return a logical "Unrelated" verdict every time. This led me to build my own `verify_papers.py` script for data curation.

## Technical Stack

*   **LLM & Agent Logic:** OpenAI GPT-4o (with JSON Mode for structured output)
*   **Retrieval:** ChromaDB, Cohere Re-ranker
*   **Text Processing:** LangChain (for Semantic Chunking)
*   **Data Ingestion:** PubMed API
*   **Frontend:** Streamlit

## Running the App Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RomanceLIVE/RAG_Contradiction_Detector.git
    cd RAG_Contradiction_Detector
    ```
2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Add API Keys:**
    *   Create a `.env` file in the project root with your keys:
      ```
      OPENAI_API_KEY="sk-..."
      COHERE_API_KEY="..."
      ```
4.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```

## Future Work

This project is a strong proof-of-concept. To evolve it into a production-ready tool, I would focus on:

*   **Quantitative Evaluation:** Implementing a formal evaluation framework like RAGAs to benchmark the system's accuracy on a curated dataset of paper pairs.
*   **Full-Text Analysis:** Moving beyond abstracts to ingest and analyze full-text PDFs using a tool like Grobid or Unstructured.
*   **Production Hygiene:** Containerizing the application with Docker and setting up a CI/CD pipeline for automated testing and deployment.