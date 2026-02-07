"""
Ablation Studies: Compare performance with different RRF k values,
dense-only, sparse-only, and hybrid retrieval methods.
"""

import json
import time
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import run_rag
from evaluation.metrics import mean_reciprocal_rank, recall_at_k


# ======================================================
# ABLATION CONFIGURATIONS
# ======================================================

QUESTIONS_PATH = "data/eval_questions.json"
ABLATION_RESULTS_PATH = "data/ablation_results.json"
ABLATION_SUMMARY_PATH = "data/ablation_summary.json"

# Test different RRF k values and retrieval modes
ABLATION_CONFIGS = [
    {"name": "hybrid_k60", "mode": "hybrid", "top_k": 10, "final_k": 5},
    {"name": "hybrid_k30", "mode": "hybrid", "top_k": 10, "final_k": 5, "rrf_k": 30},
    {"name": "hybrid_k100", "mode": "hybrid", "top_k": 10, "final_k": 5, "rrf_k": 100},
    {"name": "dense_only", "mode": "dense", "top_k": 10, "final_k": 5},
    {"name": "sparse_only", "mode": "sparse", "top_k": 10, "final_k": 5},
    {"name": "hybrid_topk5", "mode": "hybrid", "top_k": 5, "final_k": 5},
    {"name": "hybrid_topk15", "mode": "hybrid", "top_k": 15, "final_k": 5},
    {"name": "hybrid_finalfk3", "mode": "hybrid", "top_k": 10, "final_k": 3},
    {"name": "hybrid_finalfk10", "mode": "hybrid", "top_k": 10, "final_k": 10},
]


# ======================================================
# LOAD QUESTIONS
# ======================================================

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} evaluation questions")


# ======================================================
# RUN ABLATION STUDIES
# ======================================================

all_ablation_results = []
ablation_summary = {}

for config in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running ablation: {config['name']}")
    print(f"{'='*60}")
    
    config_results = []
    latencies = []
    
    for item in tqdm(questions):
        question = item["question"]
        gt_url = item["source_url"]
        
        # Run RAG with ablation parameters
        start = time.time()
        rag_output = run_rag(
            question,
            mode=config["mode"],
            top_k=config.get("top_k", 10),
            final_k=config.get("final_k", 5)
        )
        latency = round(time.time() - start, 3)
        latencies.append(latency)
        
        predicted_urls = rag_output.get("sources", [])
        
        # Calculate rank
        rank = None
        for i, url in enumerate(predicted_urls):
            if url == gt_url:
                rank = i + 1
                break
        
        rr = 1 / rank if rank else 0
        
        config_results.append({
            "ablation_config": config["name"],
            "question": question,
            "ground_truth_url": gt_url,
            "retrieved_urls": predicted_urls,
            "rank": rank,
            "reciprocal_rank": rr,
            "latency": latency
        })
        
        all_ablation_results.append(config_results[-1])
    
    # Calculate metrics for this ablation
    mrr = mean_reciprocal_rank(config_results)
    recall5 = recall_at_k(config_results, k=5)
    avg_latency = round(sum(latencies) / len(latencies), 3)
    
    ablation_summary[config["name"]] = {
        "MRR": round(mrr, 4),
        "Recall@5": round(recall5, 4),
        "Average_Latency": avg_latency,
        "Config": config,
        "Total_Questions": len(config_results)
    }
    
    print(f"\n{config['name'].upper()}")
    print(f"MRR: {round(mrr, 4)}")
    print(f"Recall@5: {round(recall5, 4)}")
    print(f"Avg Latency: {avg_latency} sec")


# ======================================================
# SAVE RESULTS
# ======================================================

os.makedirs("data", exist_ok=True)

with open(ABLATION_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_ablation_results, f, indent=2)

print(f"\nSaved detailed ablation results → {ABLATION_RESULTS_PATH}")

with open(ABLATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(ablation_summary, f, indent=2)

print(f"Saved ablation summary → {ABLATION_SUMMARY_PATH}")

# Print comparison
print(f"\n{'='*60}")
print("ABLATION STUDY SUMMARY")
print(f"{'='*60}")

for config_name, metrics in ablation_summary.items():
    print(f"\n{config_name}:")
    print(f"  MRR: {metrics['MRR']}")
    print(f"  Recall@5: {metrics['Recall@5']}")
    print(f"  Avg Latency: {metrics['Average_Latency']}s")
