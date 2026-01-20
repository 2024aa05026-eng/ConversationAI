import os

# -------- MAC + FAISS SAFE MODE (MUST BE FIRST) --------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ======================================================
# ---------------- EMBEDDING MODEL ---------------------
# ======================================================

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# ======================================================
# ---------------- GENERATION MODEL --------------------
# ======================================================

GEN_MODEL_NAME = "google/flan-t5-base"

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

gen_model.to("cpu")
gen_model.eval()


# ======================================================
# ---------------- LOAD DATA ---------------------------
# ======================================================

with open("data/corpus_chunks.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

texts = [d["text"] for d in corpus]

bm25 = BM25Okapi([t.split() for t in texts])


# ======================================================
# ---------------- LOAD FAISS --------------------------
# ======================================================

index = faiss.read_index("data/faiss.index")
faiss.omp_set_num_threads(1)

print("FAISS index dimension:", index.d)


# ======================================================
# ---------------- RRF FUSION --------------------------
# ======================================================

def rrf_fusion(dense_ids, sparse_results, k=60):

    scores = {}

    # Dense contribution
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sparse contribution
    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ======================================================
# ---------------- MAIN RAG PIPELINE -------------------
# ======================================================

def run_rag(query, mode="hybrid", top_k=10, final_k=5):

    if not query.strip():
        return {
            "answer": "Empty query",
            "sources": [],
            "mode": mode
        }

    dense_results = []
    sparse_results = []
    rrf_results = []


    # ==================================================
    # ---------------- DENSE RETRIEVAL ----------------
    # ==================================================

    if mode in ["dense", "hybrid"]:

        q_emb = embed_model.encode(query, show_progress_bar=False)

        q_emb = np.array(q_emb).astype("float32").reshape(1, -1)

        # cosine similarity
        faiss.normalize_L2(q_emb)

        dense_scores, dense_ids = index.search(q_emb, top_k)

        for rank, (idx, score) in enumerate(zip(dense_ids[0], dense_scores[0])):

            dense_results.append({
                "rank": rank + 1,
                "chunk": texts[idx],
                "url": corpus[idx]["url"],
                "score": float(score),
                "chunk_id": idx
            })


    # ==================================================
    # ---------------- SPARSE RETRIEVAL ----------------
    # ==================================================

    if mode in ["sparse", "hybrid"]:

        bm25_scores = bm25.get_scores(query.split())

        sparse_top = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for rank, (idx, score) in enumerate(sparse_top):

            sparse_results.append({
                "rank": rank + 1,
                "chunk": texts[idx],
                "url": corpus[idx]["url"],
                "score": float(score),
                "chunk_id": idx
            })


    # ==================================================
    # ---------------- FINAL CONTEXT -------------------
    # ==================================================

    # HYBRID (RRF)
    if mode == "hybrid":

        dense_ids_list = [d["chunk_id"] for d in dense_results]
        sparse_ids_list = [(s["chunk_id"], s["score"]) for s in sparse_results]

        fused = rrf_fusion(dense_ids_list, sparse_ids_list)

        for idx, score in fused:

            rrf_results.append({
                "chunk": texts[idx],
                "url": corpus[idx]["url"],
                "rrf_score": float(score),
                "chunk_id": idx
            })

        final_context = rrf_results[:final_k]

    # DENSE ONLY
    elif mode == "dense":

        final_context = dense_results[:final_k]

    # SPARSE ONLY
    else:

        final_context = sparse_results[:final_k]


    contexts = [item["chunk"] for item in final_context]
    sources = [item["url"] for item in final_context]


    # ==================================================
    # ---------------- LLM GENERATION ------------------
    # ==================================================

    if contexts:

        context_text = "\n\n".join(contexts)

        prompt = (
            "Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        inputs = gen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():

            outputs = gen_model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                num_beams=2,
                do_sample=False
            )

        answer = gen_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    else:
        answer = "No relevant answer found."


    # ==================================================
    # ---------------- FINAL RETURN --------------------
    # ==================================================

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,

        # Final context
        "final_context": final_context,
        "retrieved_chunks": final_context,

        # Debug retrieval outputs
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "rrf_results": rrf_results
    }
