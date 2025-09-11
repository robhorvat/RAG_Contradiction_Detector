# 🔎 Biomedical Contradiction Detector

**The Biomedical Contradiction Detector is an advanced AI agent designed to identify potentially contradictory claims between two scientific papers. This tool leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide a nuanced analysis, moving beyond simple summarization to genuine semantic understanding.**

---

## 🚀 Live Demo

![Contradiction Detector Demo GIF](https://github.com/RomanceLIVE/RAG_Contradiction_Detector/raw/main/RAG_Demo.gif)

---

## Problem Statement

Biomedical researchers and clinicians are faced with a deluge of new studies, many of which present conflicting findings. Manually identifying and analyzing these contradictions is a time-consuming and critical task that can impact the direction of future research and clinical guidelines. A simple keyword search or summarization tool is insufficient, as it lacks the ability to discern the nuanced claims made in a paper's abstract. This project solves that problem with a targeted AI agent.

## Features & Technical Highlights

This project was built to be a production-grade prototype, showcasing advanced techniques and robust engineering practices that go far beyond a typical tutorial project.

*   **🎯 Specific, High-Value Use Case:** Instead of a generic "research agent," this tool solves a concrete, difficult problem: detecting contradictions in scientific literature.
*   **🦾 Robust Data Ingestion:** The system uses the official PubMed API for reliable data fetching, with built-in resilience to handle different article formats (journal vs. book) and gracefully manage missing data, a common real-world challenge.
*   **🧠 Advanced RAG Pipeline:**
    *   **Semantic Chunking:** Abstracts are split using `langchain_experimental.text_splitter.SemanticChunker`, which preserves the contextual meaning of text, a critical step for accurate analysis.
    *   **Two-Stage Retrieval:** The system employs a sophisticated retrieval process. It first fetches candidate chunks from a local **ChromaDB** vector store and then uses **Cohere's Re-ranker** (`rerank-english-v3.0`) to ensure only the most contextually relevant information is passed to the LLM.
*   **🤖 Agentic Core Logic:** The heart of the system is an agent powered by **OpenAI's GPT-4o** in JSON mode. It uses a carefully engineered prompt to act as a biomedical researcher, extract key claims from two sources, and render a structured verdict (`Contradictory`, `Supporting`, `Unrelated`) with a justification.
*   **🌐 Interactive User Interface:** A clean and intuitive UI built with **Streamlit** allows for easy interaction, transforming a complex backend into a usable tool.

## Technical Stack

*   **LLM & Agent Logic:** OpenAI GPT-4o, LangChain
*   **Retrieval:** ChromaDB (Vector Store), Cohere (Re-ranker)
*   **Data Handling:** Pandas, PubMed API
*   **Frontend:** Streamlit
*   **Environment:** Python, PyCharm, venv

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RomanceLIVE/RAG_Contradiction_Detector.git
    cd contradiction-detector
    ```
2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Add API Keys:**
    *   Create a file named `.env` in the project root.
    *   Add your API keys to this file:
      ```
      OPENAI_API_KEY="sk-..."
      COHERE_API_KEY="..."
      ```
4.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

---