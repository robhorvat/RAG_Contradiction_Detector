# 🔎 Biomedical Contradiction Detector

In the fast-paced world of scientific research, keeping up with conflicting findings is a major challenge. This project was my attempt to tackle that problem head-on by building an AI agent that can read two scientific papers and intelligently flag potential contradictions.

This is more than just a summarizer, it's a tool for nuanced analysis, powered by a modern, robust Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Live Demo

![Contradiction Detector Demo GIF](https://github.com/RomanceLIVE/RAG_Contradiction_Detector/raw/main/RAG_Demo.gif)

---

## The Problem

Researchers are drowning in papers. For any given topic, like the health effects of a specific vitamin, there can be dozens of studies with subtly different conclusions. Manually tracking these is a huge time sink and a major hurdle for clinical decision-making and future research. I wanted to build a tool that could automate the first pass of this critical analysis.

## My Approach & Key Features

I designed this project to be a robust prototype, focusing on making thoughtful engineering decisions that go beyond a standard tutorial.

*   **🎯 Solving a Specific, Hard Problem:** I deliberately avoided building a generic "research bot." The goal was to solve one thing well: detecting contradictions, a task that requires a deep understanding of the text.

*   **🦾 Building for Messy, Real-World Data:** Early tests showed that real-world PubMed data is inconsistent. The final data ingestion module was hardened to reliably parse different XML formats (like journal articles vs. book chapters) and gracefully handle entries with missing abstracts—a problem we discovered and solved during development.

*   **🧠 An Advanced, Two-Stage RAG Pipeline:** A simple vector search isn't enough for nuanced scientific topics. That's why this project uses a two-stage process for high-quality retrieval:
    1.  **Semantic Chunking & Candidate Retrieval:** First, abstracts are split into contextually-aware chunks using semantic analysis. These are stored in a local **ChromaDB** instance.
    2.  **High-Relevance Re-ranking:** The initial search results are then re-ranked using **Cohere's** powerful re-ranker, ensuring that only the most relevant passages are considered for the final analysis.

*   **🤖 A Controllable and Reliable AI Core:** The heart of the system is an agent powered by **OpenAI's GPT-4o**. By using a carefully designed system prompt and enabling OpenAI's JSON mode, I can ensure the agent's output is structured, predictable, and directly usable, avoiding the randomness of a standard text generation.

*   **🌐 An Interactive Front-End:** To make the tool usable, I built a clean and simple front-end with **Streamlit**, turning a complex backend pipeline into an interactive web app.

## Tech Stack

This project integrates a modern set of tools chosen for their specific strengths:

*   **LLM & Agent Logic:** OpenAI GPT-4o, LangChain
*   **Retrieval:** ChromaDB (Vector Store), Cohere (Re-ranker)
*   **Data Handling:** Pandas, PubMed API
*   **Frontend & App:** Streamlit
*   **Development Environment:** Python, PyCharm, venv

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
    *   Create a `.env` file in the project root.
    *   Add your keys:
      ```
      OPENAI_API_KEY="sk-..."
      COHERE_API_KEY="..."
      ```
4.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

## Future Improvements

No project is ever truly finished. If I were to continue developing this, my next steps would be:

*   **Full-Text Analysis:** Extend the system to download and analyze the full PDF text of papers, not just the abstracts.
*   **Quantitative Evaluation:** Implement a formal evaluation framework like RAGAs to quantitatively score the accuracy and relevance of the agent's findings.
*   **UI Enhancements:** Add features to the Streamlit app to visualize the source text alongside the claims and allow users to explore related papers.