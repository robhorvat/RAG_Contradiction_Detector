import streamlit as st
import os
import json
from dotenv import load_dotenv

# --- IMPORTANT: LOAD ALL OUR MODULES ---
# This works because our project structure is sound.
from src.agent.contradiction_checker import ContradictionChecker
from src.retrieval.retriever import AdvancedRetriever
from src.vector_store.chroma_manager import ChromaManager

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Contradiction Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- LOAD API KEYS AND INITIALIZE AGENT (Cached) ---
# Use Streamlit's caching to initialize our agent only once.
@st.cache_resource
def initialize_agent():
    """Loads API keys and initializes all necessary components."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not (openai_key and cohere_key):
        st.error("API keys for OpenAI and Cohere must be set in your .env file.")
        st.stop()

    try:
        chroma_manager = ChromaManager(openai_api_key=openai_key)
        retriever = AdvancedRetriever(chroma_manager=chroma_manager, cohere_api_key=cohere_key)
        checker = ContradictionChecker(retriever=retriever, openai_api_key=openai_key)
        return checker
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        st.stop()


checker_agent = initialize_agent()

# --- UI COMPONENTS ---
st.title("🔎 Biomedical Contradiction Detector")
st.markdown("Enter two PubMed IDs and a research topic to analyze their abstracts for contradictory claims.")

# Use session state to remember inputs and results
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Let's use the final, triple-verified, correct paper IDs as defaults
# A large consortium study finding both low and high Vitamin D levels are associated with higher mortality.
default_id1 = "25023936"
# A meta-analysis of RCTs finding no significant effect of supplementation on mortality.
default_id2 = "31405892"
default_topic = "What is the relationship between Vitamin D levels/supplementation and total mortality?"

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        paper_id_1 = st.text_input("PubMed ID for Paper 1", value=default_id1)
    with col2:
        paper_id_2 = st.text_input("PubMed ID for Paper 2", value=default_id2)

    topic = st.text_input("Topic of Analysis", value=default_topic)

    submit_button = st.form_submit_button(label="Analyze for Contradictions")

# --- ANALYSIS LOGIC ---
if submit_button:
    if not paper_id_1 or not paper_id_2 or not topic:
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Analyzing papers... This may take a moment."):
            try:
                result = checker_agent.check(
                    paper_1_id=paper_id_1.strip(),
                    paper_2_id=paper_id_2.strip(),
                    topic=topic.strip()
                )
                st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.session_state.analysis_result = None

# --- DISPLAY RESULTS ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    st.divider()
    st.header("Analysis Results")

    if "error" in result:
        st.error(f"An error occurred: {result['error']}")
    else:
        verdict = result.get("analysis", {}).get("verdict", "Unknown")

        if verdict == "Contradictory":
            st.error(f"**Verdict: {verdict}**")
        elif verdict == "Supporting":
            st.success(f"**Verdict: {verdict}**")
        else:
            st.warning(f"**Verdict: {verdict}**")

        st.info(f"**Justification:** {result.get('analysis', {}).get('justification', 'N/A')}")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Paper 1 Claim")
                st.write(result.get("paper_1_claim", "N/A"))
        with col2:
            with st.container(border=True):
                st.subheader("Paper 2 Claim")
                st.write(result.get("paper_2_claim", "N/A"))

        with st.expander("Show Raw JSON Output"):
            st.json(result)