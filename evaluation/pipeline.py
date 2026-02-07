#!/usr/bin/env python3
"""
Unified Evaluation Pipeline
Orchestrates all evaluation components:
1. Main evaluation (MRR, Recall@5, Latency)
2. Ablation studies (different configurations)
3. Error analysis (failure categorization)
4. Advanced metrics (ROUGE, BLEU, Semantic Similarity)
5. LLM-as-Judge evaluation
6. Report generation (HTML + JSON)
"""

import subprocess
import sys
import os
import json
from datetime import datetime


# ======================================================
# PIPELINE CONFIGURATION
# ======================================================

PIPELINE_STEPS = [
    {
        "name": "Main Evaluation",
        "script": "evaluation/eval_runner.py",
        "description": "Run evaluation on all 3 modes (dense, sparse, hybrid)",
        "enabled": True
    },
    {
        "name": "Ablation Studies",
        "script": "evaluation/ablation.py",
        "description": "Compare different RRF k values and retrieval modes",
        "enabled": True
    },
    {
        "name": "Error Analysis",
        "script": "evaluation/error_analysis.py",
        "description": "Categorize failures by question type and error patterns",
        "enabled": True
    },
    {
        "name": "Report Generation",
        "script": "evaluation/report_generator.py",
        "description": "Generate HTML and JSON evaluation reports",
        "enabled": True
    }
]


# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_step(step):
    """Run a single pipeline step"""
    print_header(step["name"])
    print(f"📌 {step['description']}\n")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, step["script"]],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {step['name']} completed successfully")
            return True
        else:
            print(f"\n❌ {step['name']} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running {step['name']}: {e}")
        return False


def generate_pipeline_report(results):
    """Generate pipeline execution report"""
    
    report = {
        "metadata": {
            "executed_at": datetime.now().isoformat(),
            "python_version": sys.version,
            "cwd": os.getcwd()
        },
        "pipeline_steps": results,
        "summary": {
            "total_steps": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed")
        }
    }
    
    return report


# ======================================================
# MAIN PIPELINE EXECUTION
# ======================================================

def main():
    """Execute the full evaluation pipeline"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  ConversationAI - Hybrid RAG Evaluation Pipeline".center(68) + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Check if main data exists
    if not os.path.exists("data/eval_questions.json"):
        print("❌ ERROR: data/eval_questions.json not found!")
        print("   Please run: python evaluation/question_generator.py")
        return 1
    
    results = []
    
    # Run enabled pipeline steps
    for step in PIPELINE_STEPS:
        if not step["enabled"]:
            print(f"⏭️  Skipping: {step['name']}")
            continue
        
        success = run_step(step)
        
        results.append({
            "name": step["name"],
            "script": step["script"],
            "status": "success" if success else "failed"
        })
    
    # Generate pipeline report
    print_header("Pipeline Execution Complete")
    
    pipeline_report = generate_pipeline_report(results)
    
    print("📊 Pipeline Summary:")
    print(f"   • Total Steps: {pipeline_report['summary']['total_steps']}")
    print(f"   • ✅ Successful: {pipeline_report['summary']['successful']}")
    print(f"   • ❌ Failed: {pipeline_report['summary']['failed']}")
    
    # Save pipeline report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/pipeline_report.json"
    
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=2)
    
    print(f"\n📁 Pipeline report saved → {report_path}")
    
    # Print output locations
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    
    output_files = [
        ("data/eval_results.json", "Detailed evaluation results"),
        ("data/eval_summary.json", "Evaluation summary metrics"),
        ("data/ablation_results.json", "Ablation study results"),
        ("data/ablation_summary.json", "Ablation study summary"),
        ("data/error_analysis.json", "Error analysis results"),
        ("reports/evaluation_report.html", "HTML visualization report"),
        ("reports/evaluation_report.json", "Comprehensive JSON report"),
        ("reports/pipeline_report.json", "Pipeline execution report")
    ]
    
    for file_path, description in output_files:
        exists = "✅" if os.path.exists(file_path) else "⏳"
        print(f"{exists} {file_path:<40} - {description}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. View HTML Report:
   Open 'reports/evaluation_report.html' in a web browser

2. Review JSON Reports:
   - data/eval_summary.json (main metrics)
   - data/ablation_summary.json (ablation studies)
   - data/error_analysis.json (failure analysis)

3. Run Web UI:
   streamlit run app.py --server.fileWatcherType=none

4. Additional Analysis:
   - evaluation/llm_judge.py - LLM-based answer quality evaluation
   - evaluation/metrics.py - Advanced metrics (ROUGE, BLEU, Semantic Similarity)
""")
    
    print("="*70 + "\n")
    
    # Return success if all steps succeeded
    return 0 if pipeline_report['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
