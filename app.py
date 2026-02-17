import streamlit as st
from dotenv import load_dotenv

from src.agent.contradiction_checker import ContradictionChecker
from src.agent.llm_clients import OpenAIJSONClient, GeminiJSONClient, LocalRuleBasedJSONClient
from src.config import load_settings
from src.retrieval.retriever import AdvancedRetriever
from src.vector_store.chroma_manager import ChromaManager
from src.data_ingestion.pubmed_fetcher import get_paper_details
from src.processing.text_splitter import chunk_text_semantically

# Page Configuration
st.set_page_config(page_title="Biomedical Contradiction Detector", page_icon="🔎", layout="wide")


def _build_llm_client(settings):
    provider = settings.llm_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
        return OpenAIJSONClient(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY.")
        return GeminiJSONClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )

    if provider == "local":
        return LocalRuleBasedJSONClient()

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")


@st.cache_resource
def initialize_system():
    """
    Loads API keys and initializes core, non-UI components.
    This function is cached, so it only runs once per session, making the app fast.
    """
    load_dotenv()
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings/chunking.")

    chroma_manager = ChromaManager(openai_api_key=settings.openai_api_key)
    retriever = AdvancedRetriever(
        chroma_manager=chroma_manager,
        cohere_api_key=settings.cohere_api_key,
        cohere_model=settings.cohere_rerank_model,
    )
    llm_client = _build_llm_client(settings)
    checker = ContradictionChecker(retriever=retriever, llm_client=llm_client)

    return checker, chroma_manager, settings.openai_api_key, settings


def ensure_papers_are_in_db(paper_ids: list[str], chroma_manager: ChromaManager, openai_key: str):
    """
    Checks if papers are in the DB and ingests them if they are not.
    """
    collection = chroma_manager.create_or_get_collection("pubmed_papers")
    ids_to_check = [f"{pid}-chunk-0" for pid in paper_ids]
    found_ids_set = chroma_manager.check_documents_exist(collection, ids_to_check)

    missing_paper_ids = [pid for i, pid in enumerate(paper_ids) if ids_to_check[i] not in found_ids_set]

    if missing_paper_ids:
        with st.spinner(f"First time seeing paper(s) {', '.join(missing_paper_ids)}. Ingesting..."):
            for paper_id in missing_paper_ids:
                paper_data = get_paper_details(paper_id)
                if "error" not in paper_data and paper_data.get("abstract"):
                    chunks = chunk_text_semantically(paper_data["abstract"], openai_key)
                    metadatas = [{"pubmed_id": paper_id, "title": paper_data["title"]} for _ in chunks]
                    ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
                    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            st.toast("New papers ingested successfully!")


# --- Main App Execution ---
try:
    components = initialize_system()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

checker_agent, chroma_manager, openai_key, settings = components

st.title("🔎 Biomedical Contradiction Detector")
st.markdown("This app uses a RAG pipeline to analyze scientific papers. Enter any two PubMed IDs to begin.")
st.caption(f"LLM provider: `{settings.llm_provider}`")

if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# Using our personally verified contradictory papers.
default_id1, default_id2 = "19307517", "8554248"
default_topic = "Does Vitamin D supplementation prevent fractures in the elderly?"

with st.form("input_form"):
    st.subheader("Enter papers to analyze:")
    col1, col2 = st.columns(2)
    with col1:
        paper_id_1 = st.text_input("PubMed ID for Paper 1", value=default_id1)
    with col2:
        paper_id_2 = st.text_input("PubMed ID for Paper 2", value=default_id2)
    topic = st.text_input("Topic of Analysis", value=default_topic)
    submit_button = st.form_submit_button("Analyze")

if submit_button:
    if not all([paper_id_1, paper_id_2, topic]):
        st.warning("Please fill in all fields.")
    else:
        paper_ids = [paper_id_1.strip(), paper_id_2.strip()]
        ensure_papers_are_in_db(paper_ids, chroma_manager, openai_key)

        with st.spinner("Running RAG pipeline... This may take a moment."):
            st.session_state.analysis_result = checker_agent.check(
                paper_1_id=paper_ids[0],
                paper_2_id=paper_ids[1],
                topic=topic.strip()
            )
            st.session_state.paper_ids_for_display = paper_ids

# --- Results Display ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    pids = st.session_state.get('paper_ids_for_display', ['', ''])
    st.divider()
    st.header("Analysis Results")
    if "error" in result:
        st.error(f"**Error:** {result['error']}")
    else:
        verdict = result.get("analysis", {}).get("verdict", "Unknown")
        color = "red" if verdict == "Contradictory" else "orange" if verdict == "Unrelated" else "green"
        st.markdown(f"### Verdict: <span style='color:{color};'>{verdict}</span>", unsafe_allow_html=True)
        st.info(f"**Justification:** {result.get('analysis', {}).get('justification', 'N/A')}")
        baseline = result.get("baseline_verifier")
        if baseline:
            st.caption(
                "Heuristic baseline: "
                f"{baseline.get('verdict', 'Unknown')} "
                f"(confidence={baseline.get('confidence', 'n/a')}, "
                f"overlap={baseline.get('overlap_ratio', 'n/a')})"
            )
        col1, col2 = st.columns(2)
        with col1, st.container(border=True):
            st.subheader(f"Claim from Paper 1 ({pids[0]})")
            st.write(result.get("paper_1_claim", "N/A"))
            with st.expander("Show Evidence from Paper 1"):
                st.caption("The LLM based its claim on these retrieved passages:")
                for evidence in result.get('evidence_paper_1', []): st.markdown(f"> {evidence}")
        with col2, st.container(border=True):
            st.subheader(f"Claim from Paper 2 ({pids[1]})")
            st.write(result.get("paper_2_claim", "N/A"))
            with st.expander("Show Evidence from Paper 2"):
                st.caption("The LLM based its claim on these retrieved passages:")
                for evidence in result.get('evidence_paper_2', []): st.markdown(f"> {evidence}")
