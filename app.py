import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import time
from src.rag_pipeline import run_rag

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Hybrid RAG System",
    layout="wide"
)

st.title("Hybrid Retrieval-Augmented Generation System")

st.markdown(
"""
This system combines **Dense Vector Search**, **BM25 Sparse Retrieval** and 
**Reciprocal Rank Fusion (RRF)** to answer questions from Wikipedia corpus.
"""
)

# ---------------- SIDEBAR CONTROLS ----------------

st.sidebar.header("Retrieval Settings")

mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["hybrid", "dense", "sparse"]
)

top_k = st.sidebar.slider(
    "Top-K Retrieval (Dense + Sparse)",
    min_value=5,
    max_value=20,
    value=10
)

final_k = st.sidebar.slider(
    "Final Context Chunks (After Fusion)",
    min_value=3,
    max_value=10,
    value=5
)

# ---------------- SESSION STATE ----------------

if "result" not in st.session_state:
    st.session_state.result = None

if "latency" not in st.session_state:
    st.session_state.latency = None

# ---------------- QUERY INPUT ----------------

query = st.text_input(
    "Enter your question",
    placeholder="Example: Who invented the World Wide Web?"
)

# ---------------- RUN BUTTON ----------------

if st.button("Run Hybrid RAG") and query.strip():

    start = time.time()

    st.session_state.result = run_rag(
        query,
        mode=mode,
        top_k=top_k,
        final_k=final_k
    )

    st.session_state.latency = round(time.time() - start, 3)

# =================================================
# ---------------- DISPLAY OUTPUT ------------------
# =================================================

if st.session_state.result:

    result = st.session_state.result
    latency = st.session_state.latency

    col1, col2 = st.columns([3,1])

    with col1:
        st.subheader("Generated Answer")
        st.success(result["answer"])

    with col2:
        st.metric("Latency (seconds)", latency)
        st.metric("Retrieval Mode", mode.upper())

    # ---------------- SOURCES ----------------

    st.subheader("Source Documents")

    for url in result["sources"]:
        st.markdown(f"- {url}")

    # ---------------- CONTEXT ----------------

    st.subheader("Final Context Used (Post Fusion)")

    for i, item in enumerate(result["final_context"]):

        with st.expander(f"Chunk {i+1} | RRF Score: {round(item.get('rrf_score',0),4)}"):

            st.markdown(f"**Source:** {item['url']}")
            st.write(item["chunk"])

    # ---------------- RETRIEVAL TABS ----------------

    st.subheader("Retrieval Transparency")

    tab1, tab2, tab3 = st.tabs(
        ["Dense Retrieval", "Sparse Retrieval (BM25)", "RRF Fusion"]
    )

    # ---------- Dense ----------

    with tab1:

        dense = result["dense_results"]

        if dense:
            st.table([
                {
                    "Rank": d["rank"],
                    "Score": round(d["score"],4),
                    "URL": d["url"]
                }
                for d in dense[:10]
            ])
        else:
            st.info("Dense retrieval not used")

    # ---------- Sparse ----------

    with tab2:

        sparse = result["sparse_results"]

        if sparse:
            st.table([
                {
                    "Rank": s["rank"],
                    "Score": round(s["score"],4),
                    "URL": s["url"]
                }
                for s in sparse[:10]
            ])
        else:
            st.info("Sparse retrieval not used")

    # ---------- RRF ----------

    with tab3:

        rrf = result["rrf_results"]

        if rrf:
            st.table([
                {
                    "Final Rank": i+1,
                    "RRF Score": round(r["rrf_score"],4),
                    "URL": r["url"]
                }
                for i, r in enumerate(rrf[:10])
            ])
        else:
            st.info("Fusion not used")
