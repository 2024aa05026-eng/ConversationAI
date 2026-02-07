import json
import time
from tqdm import tqdm
import sys
import os

# ---------- PATH FIX ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import run_rag
from evaluation.metrics import mean_reciprocal_rank, recall_at_k


# ---------- PATHS ----------

QUESTIONS_PATH = "data/eval_questions.json"
RESULTS_PATH = "data/eval_results.json"
SUMMARY_PATH = "data/eval_summary.json"

MODES = ["dense", "sparse", "hybrid"]


# ---------- LOAD QUESTIONS ----------

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("Loaded", len(questions), "evaluation questions")


all_results = []
latencies = {mode: [] for mode in MODES}


# ======================================================
# ---------------- RUN EVALUATION ----------------------
# ======================================================

for item in tqdm(questions):

    question = item["question"]
    gt_url = item["source_url"]

    for mode in MODES:

        start = time.time()

        rag_output = run_rag(question, mode=mode)

        latency = round(time.time() - start, 3)
        latencies[mode].append(latency)

        predicted_urls = rag_output.get("sources", [])

        rank = None

        for i, url in enumerate(predicted_urls):
            if url == gt_url:
                rank = i + 1
                break

        rr = 1 / rank if rank else 0

        all_results.append({
            "mode": mode,
            "question": question,
            "ground_truth_url": gt_url,
            "ground_truth_answer": item.get("answer", ""),
            "generated_answer": rag_output.get("answer", ""),
            "retrieved_urls": predicted_urls,
            "context": " ".join(rag_output.get("final_context", [{}])[:3] if rag_output.get("final_context") else []),
            "rank": rank,
            "reciprocal_rank": rr,
            "latency": latency
        })


# ======================================================
# ---------------- METRICS PER MODE --------------------
# ======================================================

summary = {}

print("\n========== FINAL RESULTS ==========")

for mode in MODES:

    mode_results = [r for r in all_results if r["mode"] == mode]

    mrr = mean_reciprocal_rank(mode_results)
    recall5 = recall_at_k(mode_results, k=5)

    avg_latency = round(sum(latencies[mode]) / len(latencies[mode]), 3)

    summary[mode] = {
        "MRR": round(mrr, 4),
        "Recall@5": round(recall5, 4),
        "Average_Latency": avg_latency,
        "Total_Questions": len(mode_results)
    }

    print(f"\nMODE: {mode.upper()}")
    print("MRR:", round(mrr, 4))
    print("Recall@5:", round(recall5, 4))
    print("Avg Latency:", avg_latency, "sec")


# ======================================================
# ---------------- SAVE RESULTS ------------------------
# ======================================================

os.makedirs("evaluation/results", exist_ok=True)

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved detailed results →", RESULTS_PATH)


with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved summary metrics →", SUMMARY_PATH)
