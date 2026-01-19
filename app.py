import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import time
from src.rag_pipeline import run_rag

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hybrid RAG System", layout="wide")

st.title("Hybrid RAG System")

# ---------------- SESSION STATE INIT ----------------
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "latency" not in st.session_state:
    st.session_state.latency = None

# ---------------- INPUT ----------------
query = st.text_input("Ask your question:")

# ---------------- SEARCH BUTTON ----------------
if st.button("Search") and query.strip():

    start_time = time.time()

    st.session_state.rag_result = run_rag(query)

    st.session_state.latency = round(time.time() - start_time, 2)

# ---------------- DISPLAY RESULTS ----------------
if st.session_state.rag_result:

    result = st.session_state.rag_result
    latency = st.session_state.latency

    # ---------------- ANSWER ----------------
    st.subheader("Answer")
    st.write(result.get("answer", "No answer generated."))

    st.caption(f"Response Time: {latency} seconds")

    # ---------------- SOURCES ----------------
    st.subheader("Top Sources")

    for url in result.get("sources", [])[:5]:
        st.write(url)

    # ---------------- CONTEXT USED ----------------
    st.subheader("Context Used For Answer")

    final_context = result.get("final_context", [])

    if len(final_context) == 0:
        st.warning("No context returned from RAG pipeline")
    else:
        for item in final_context:
            st.markdown(
                f"""
**Source:** {item['url']}  
**RRF Score:** {round(item['rrf_score'], 4)}

{item['chunk'][:600]}...
---
"""
            )

    # ---------------- RETRIEVAL DETAILS ----------------
    st.subheader("Retrieval Details")

    tab1, tab2, tab3 = st.tabs(
        ["Dense Retrieval", "Sparse Retrieval (BM25)", "RRF Fusion"]
    )

    # ---------- Dense ----------
    with tab1:
        dense_results = result.get("dense_results", [])

        if len(dense_results) == 0:
            st.info("No dense retrieval results.")
        else:
            for item in dense_results[:5]:
                st.markdown(
                    f"""
**Rank:** {item['rank']}  
**Score:** {round(item['score'], 4)}  
**Source:** {item['url']}  

{item['chunk'][:300]}...
---
"""
                )

    # ---------- Sparse ----------
    with tab2:
        sparse_results = result.get("sparse_results", [])

        if len(sparse_results) == 0:
            st.info("No sparse retrieval results.")
        else:
            for item in sparse_results[:5]:
                st.markdown(
                    f"""
**Rank:** {item['rank']}  
**Score:** {round(item['score'], 4)}  
**Source:** {item['url']}  

{item['chunk'][:300]}...
---
"""
                )

    # ---------- RRF ----------
    with tab3:
        rrf_results = result.get("rrf_results", [])

        if len(rrf_results) == 0:
            st.info("No fusion results.")
        else:
            for i, item in enumerate(rrf_results[:5]):
                st.markdown(
                    f"""
**Final Rank:** {i+1}  
**RRF Score:** {round(item['rrf_score'], 4)}  
**Source:** {item['url']}  

{item['chunk'][:300]}...
---
"""
                )
