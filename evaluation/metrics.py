import numpy as np


# -------------------------
# MRR @ URL LEVEL
# -------------------------

def mean_reciprocal_rank(results):
    """
    results = list of dicts
    Each item:
      {
        "retrieved_urls": [...],
        "ground_truth_url": "..."
      }
    """

    rr_scores = []

    for item in results:

        gt = item["ground_truth_url"]
        retrieved = item["retrieved_urls"]

        rank = 0
        for i, url in enumerate(retrieved):
            if url == gt:
                rank = i + 1
                break

        if rank == 0:
            rr_scores.append(0)
        else:
            rr_scores.append(1 / rank)

    return np.mean(rr_scores)


# -------------------------
# Recall@K (URL LEVEL)
# -------------------------

def recall_at_k(results, k=5):

    hits = 0

    for item in results:
        gt = item["ground_truth_url"]
        retrieved = item["retrieved_urls"][:k]

        if gt in retrieved:
            hits += 1

    return hits / len(results)


# -------------------------
# Hit Rate@K
# -------------------------

def hit_rate_at_k(results, k=5):
    return recall_at_k(results, k)
