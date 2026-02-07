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

# -------------------------
# ROUGE Score (Answer Quality)
# -------------------------

def rouge_score(generated, reference):
    """
    Simple ROUGE-L implementation (longest common subsequence)
    Compares generated answer with reference answer
    """
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    
    lcs = lcs_length(gen_tokens, ref_tokens)
    
    if len(ref_tokens) == 0:
        return 0.0
    
    # ROUGE-L F1 score
    precision = lcs / len(gen_tokens) if len(gen_tokens) > 0 else 0
    recall = lcs / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_rouge_batch(results):
    """
    Compute ROUGE-L for batch of results
    results: list of dicts with 'generated_answer' and 'ground_truth_answer'
    """
    rouge_scores = []
    
    for item in results:
        gen = item.get("generated_answer", "")
        ref = item.get("ground_truth_answer", "")
        
        if gen and ref:
            score = rouge_score(gen, ref)
            rouge_scores.append(score)
    
    return np.mean(rouge_scores) if rouge_scores else 0.0


# -------------------------
# BLEU Score (Answer Quality)
# -------------------------

def bleu_score(generated, reference, max_n=2):
    """
    Simple BLEU score implementation (n-gram precision)
    Uses unigrams and bigrams by default
    """
    from collections import Counter
    
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    
    if not gen_tokens:
        return 0.0
    
    bleu_scores = []
    
    # Calculate n-gram precision
    for n in range(1, max_n + 1):
        gen_ngrams = [" ".join(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
        ref_ngrams = Counter([" ".join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        
        if not gen_ngrams:
            continue
        
        matches = sum(min(ref_ngrams.get(g, 0), 1) for g in gen_ngrams)
        precision = matches / len(gen_ngrams) if gen_ngrams else 0
        bleu_scores.append(precision)
    
    # Geometric mean
    if bleu_scores:
        import math
        return math.exp(sum(math.log(s) if s > 0 else -10 for s in bleu_scores) / len(bleu_scores))
    
    return 0.0


def compute_bleu_batch(results):
    """
    Compute BLEU score for batch of results
    """
    bleu_scores = []
    
    for item in results:
        gen = item.get("generated_answer", "")
        ref = item.get("ground_truth_answer", "")
        
        if gen and ref:
            score = bleu_score(gen, ref)
            bleu_scores.append(score)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


# -------------------------
# Semantic Similarity (Answer Quality)
# -------------------------

def semantic_similarity(generated, reference):
    """
    Compute cosine similarity between generated and reference answers
    using sentence embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        gen_emb = model.encode(generated, convert_to_tensor=False)
        ref_emb = model.encode(reference, convert_to_tensor=False)
        
        # Cosine similarity
        dot_product = np.dot(gen_emb, ref_emb)
        norm_gen = np.linalg.norm(gen_emb)
        norm_ref = np.linalg.norm(ref_emb)
        
        if norm_gen * norm_ref == 0:
            return 0.0
        
        return float(dot_product / (norm_gen * norm_ref))
    
    except Exception as e:
        print(f"Warning: Semantic similarity computation failed: {e}")
        return 0.0


def compute_semantic_similarity_batch(results):
    """
    Compute semantic similarity for batch of results
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        sim_scores = []
        
        for item in results:
            gen = item.get("generated_answer", "")
            ref = item.get("ground_truth_answer", "")
            
            if gen and ref:
                gen_emb = model.encode(gen, convert_to_tensor=False)
                ref_emb = model.encode(ref, convert_to_tensor=False)
                
                dot_product = np.dot(gen_emb, ref_emb)
                norm_gen = np.linalg.norm(gen_emb)
                norm_ref = np.linalg.norm(ref_emb)
                
                if norm_gen * norm_ref > 0:
                    sim_scores.append(float(dot_product / (norm_gen * norm_ref)))
        
        return np.mean(sim_scores) if sim_scores else 0.0
    
    except Exception as e:
        print(f"Warning: Batch semantic similarity failed: {e}")
        return 0.0