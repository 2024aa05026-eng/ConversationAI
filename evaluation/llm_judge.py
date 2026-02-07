"""
LLM-as-Judge Evaluation: Use LLM to evaluate answer quality on multiple dimensions
- Factual Accuracy
- Completeness
- Relevance to question
- Coherence and clarity
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ======================================================
# LOAD JUDGE MODEL
# ======================================================

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cpu")
model.eval()


# ======================================================
# JUDGE PROMPTS
# ======================================================

JUDGE_PROMPTS = {
    "factual_accuracy": """Evaluate the factual accuracy of this answer.
Context: {context}
Ground Truth Answer: {reference}
Generated Answer: {generated}
Rate on scale 1-5 (1=completely wrong, 5=completely accurate): """
    
    "completeness": """Evaluate how complete the answer is compared to the reference.
Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}
Does it address all parts? Rate 1-5 (1=missing key points, 5=fully complete): """
    
    "relevance": """Evaluate how relevant the generated answer is to the question.
Question: {question}
Answer: {generated}
Is the answer directly addressing the question? Rate 1-5 (1=irrelevant, 5=highly relevant): """
    
    "coherence": """Evaluate the coherence and clarity of this answer.
Answer: {generated}
Is the answer clear, logical and well-structured? Rate 1-5 (1=incoherent, 5=very coherent): """
}


# ======================================================
# EVALUATE ANSWER WITH LLM
# ======================================================

def evaluate_answer(question, generated, reference, context=""):
    """
    Evaluate answer across multiple dimensions using LLM
    """
    
    judgments = {}
    
    for dimension, prompt_template in JUDGE_PROMPTS.items():
        # Create prompt
        prompt = prompt_template.format(
            question=question,
            generated=generated[:500],  # Limit length
            reference=reference[:500],
            context=context[:300]
        )
        
        try:
            # Tokenize and generate
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Extract score (1-5)
            score = extract_score(response)
            judgments[dimension] = score
        
        except Exception as e:
            print(f"Error evaluating {dimension}: {e}")
            judgments[dimension] = 3  # Default middle score
    
    return judgments


def extract_score(response):
    """
    Extract numeric score from LLM response
    """
    # Look for digits 1-5 in response
    for char in response:
        if char in "12345":
            return int(char)
    
    # Default to middle score if no digit found
    return 3


# ======================================================
# BATCH EVALUATE RESULTS
# ======================================================

EVAL_RESULTS_PATH = "data/eval_results.json"
LLM_JUDGE_RESULTS_PATH = "data/llm_judge_results.json"
LLM_JUDGE_SUMMARY_PATH = "data/llm_judge_summary.json"

# Load evaluation results
with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
    eval_results = json.load(f)

print(f"Loaded {len(eval_results)} evaluation results")
print(f"Running LLM-as-Judge evaluation on {len(eval_results)} answers...")

llm_judge_results = []
mode_scores = {
    "dense": [],
    "sparse": [],
    "hybrid": []
}

for result in tqdm(eval_results):
    
    question = result["question"]
    generated = result.get("generated_answer", "No answer provided")
    reference = result.get("ground_truth_answer", "")
    context = result.get("context", "")
    mode = result["mode"]
    
    # Skip if no reference answer
    if not reference:
        continue
    
    # Evaluate with LLM-as-Judge
    judgments = evaluate_answer(question, generated, reference, context)
    
    # Calculate average score across dimensions
    avg_score = sum(judgments.values()) / len(judgments) if judgments else 0
    
    llm_judge_results.append({
        "mode": mode,
        "question": question,
        "generated_answer": generated,
        "ground_truth_answer": reference,
        "judge_scores": judgments,
        "average_score": avg_score
    })
    
    if mode in mode_scores:
        mode_scores[mode].append(avg_score)


# ======================================================
# COMPUTE STATISTICS
# ======================================================

import numpy as np

llm_judge_summary = {}

for mode, scores in mode_scores.items():
    if scores:
        llm_judge_summary[mode] = {
            "average_judge_score": round(np.mean(scores), 2),
            "median_judge_score": round(np.median(scores), 2),
            "std_judge_score": round(np.std(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "evaluated_questions": len(scores)
        }


# ======================================================
# SAVE RESULTS
# ======================================================

os.makedirs("data", exist_ok=True)

with open(LLM_JUDGE_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(llm_judge_results, f, indent=2)

print(f"Saved LLM judge results → {LLM_JUDGE_RESULTS_PATH}")

with open(LLM_JUDGE_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(llm_judge_summary, f, indent=2)

print(f"Saved LLM judge summary → {LLM_JUDGE_SUMMARY_PATH}")

# ======================================================
# PRINT SUMMARY
# ======================================================

print("\n" + "="*60)
print("LLM-AS-JUDGE EVALUATION SUMMARY")
print("="*60)

for mode, stats in llm_judge_summary.items():
    print(f"\n{mode.upper()} MODE")
    print("-" * 40)
    print(f"Average Judge Score: {stats['average_judge_score']}/5.0")
    print(f"Median Judge Score: {stats['median_judge_score']}/5.0")
    print(f"Score Std. Dev: {stats['std_judge_score']}")
    print(f"Range: {stats['min_score']} - {stats['max_score']}")
    print(f"Questions Evaluated: {stats['evaluated_questions']}")
