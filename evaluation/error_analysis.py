"""
Error Analysis: Categorize and analyze failures by question type,
retrieval method, and other patterns.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ======================================================
# LOAD RESULTS AND QUESTIONS
# ======================================================

EVAL_RESULTS_PATH = "data/eval_results.json"
EVAL_QUESTIONS_PATH = "data/eval_questions.json"
ERROR_ANALYSIS_PATH = "data/error_analysis.json"

with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
    all_results = json.load(f)

with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions_lookup = {q["question"]: q for q in json.load(f)}


# ======================================================
# ERROR CATEGORIZATION
# ======================================================

def categorize_error(result):
    """
    Categorize retrieval failures:
    - success: ground truth in top-5
    - rank_6_10: found but ranked 6-10
    - rank_11_20: found but ranked 11-20
    - not_retrieved: not in top-20
    - not_found: ground truth URL not in results at all
    """
    
    rank = result.get("rank")
    
    if rank is None:
        return "not_found"
    elif rank <= 5:
        return "success"
    elif rank <= 10:
        return "rank_6_10"
    elif rank <= 20:
        return "rank_11_20"
    else:
        return "rank_21_plus"


def get_question_type(question):
    """Simple question type detection"""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ["what", "who", "where", "which"]):
        return "factual"
    elif any(word in q_lower for word in ["why", "how"]):
        return "explanatory"
    elif any(word in q_lower for word in ["compare", "different", "vs", "versus"]):
        return "comparative"
    elif question.count("and") > 1 or "," in question:
        return "complex"
    else:
        return "other"


# ======================================================
# ANALYZE ERRORS BY MODE
# ======================================================

error_analysis = {}

for mode in ["dense", "sparse", "hybrid"]:
    mode_results = [r for r in all_results if r["mode"] == mode]
    
    error_breakdown = defaultdict(list)
    question_type_breakdown = defaultdict(lambda: defaultdict(int))
    
    for result in mode_results:
        question = result["question"]
        error_type = categorize_error(result)
        question_type = get_question_type(question)
        
        error_breakdown[error_type].append(result)
        question_type_breakdown[question_type][error_type] += 1
    
    # Calculate statistics
    total = len(mode_results)
    stats = {
        "Total_Questions": total,
        "Success_Count": len(error_breakdown.get("success", [])),
        "Success_Rate": round(len(error_breakdown.get("success", [])) / total * 100, 2) if total > 0 else 0,
        "Failure_Count": total - len(error_breakdown.get("success", [])),
        "Failure_Rate": round((total - len(error_breakdown.get("success", []))) / total * 100, 2) if total > 0 else 0,
    }
    
    # Error type distribution
    error_distribution = {}
    for error_type, results in error_breakdown.items():
        error_distribution[error_type] = {
            "count": len(results),
            "percentage": round(len(results) / total * 100, 2) if total > 0 else 0
        }
    
    # Question type analysis
    question_type_stats = {}
    for qtype, error_counts in question_type_breakdown.items():
        total_qtype = sum(error_counts.values())
        success_count = error_counts.get("success", 0)
        question_type_stats[qtype] = {
            "total": total_qtype,
            "success": success_count,
            "success_rate": round(success_count / total_qtype * 100, 2) if total_qtype > 0 else 0,
            "error_distribution": dict(error_counts)
        }
    
    error_analysis[mode] = {
        "statistics": stats,
        "error_distribution": error_distribution,
        "question_type_analysis": question_type_stats,
        "failed_examples": {
            "not_found": [
                {
                    "question": r["question"],
                    "ground_truth": r["ground_truth_url"],
                    "retrieved": r["retrieved_urls"][:5]
                }
                for r in error_breakdown.get("not_found", [])[:3]
            ],
            "low_rank": [
                {
                    "question": r["question"],
                    "rank": r["rank"],
                    "ground_truth": r["ground_truth_url"],
                    "retrieved": r["retrieved_urls"][:5]
                }
                for r in error_breakdown.get("rank_11_20", [])[:3]
            ]
        }
    }


# ======================================================
# SAVE ERROR ANALYSIS
# ======================================================

os.makedirs("data", exist_ok=True)

with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
    json.dump(error_analysis, f, indent=2)

print(f"Saved error analysis → {ERROR_ANALYSIS_PATH}")

# ======================================================
# PRINT SUMMARY
# ======================================================

print("\n" + "="*60)
print("ERROR ANALYSIS SUMMARY")
print("="*60)

for mode, analysis in error_analysis.items():
    print(f"\n{mode.upper()} MODE")
    print("-" * 40)
    
    stats = analysis["statistics"]
    print(f"Total Questions: {stats['Total_Questions']}")
    print(f"Success Rate: {stats['Success_Rate']}%")
    print(f"Failure Rate: {stats['Failure_Rate']}%")
    
    print(f"\nError Distribution:")
    for error_type, dist in analysis["error_distribution"].items():
        print(f"  {error_type}: {dist['count']} ({dist['percentage']}%)")
    
    print(f"\nBy Question Type:")
    for qtype, qstats in analysis["question_type_analysis"].items():
        print(f"  {qtype}: {qstats['success']}/{qstats['total']} correct ({qstats['success_rate']}%)")
