"""
Report Generator: Create comprehensive evaluation reports with visualizations
Generates HTML and JSON reports with all metrics, charts, and analysis
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ======================================================
# LOAD D ATA
# ======================================================

DATA_DIR = "data"
REPORT_DIR = "reports"

def load_json_safe(path):
    """Load JSON file safely"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


eval_summary = load_json_safe(os.path.join(DATA_DIR, "eval_summary.json"))
eval_results = load_json_safe(os.path.join(DATA_DIR, "eval_results.json")) or []
ablation_summary = load_json_safe(os.path.join(DATA_DIR, "ablation_summary.json"))
error_analysis = load_json_safe(os.path.join(DATA_DIR, "error_analysis.json"))
llm_judge_summary = load_json_safe(os.path.join(DATA_DIR, "llm_judge_summary.json"))


# ======================================================
# GENERATE HTML REPORT
# ======================================================

def generate_html_report():
    """Generate comprehensive HTML report"""
    
    report_dir = REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConversationAI - Hybrid RAG System Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        h2 {{
            color: #764ba2;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }}
        
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 15px;
        }}
        
        p {{
            line-height: 1.6;
            margin-bottom: 15px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #999;
            font-size: 12px;
        }}
        
        .success {{
            color: #27ae60;
        }}
        
        .warning {{
            color: #f39c12;
        }}
        
        .error {{
            color: #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ConversationAI - Hybrid RAG System</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Comprehensive Evaluation Report | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <!-- Executive Summary -->
        <h2>📊 Executive Summary</h2>
        
        <div class="metrics-grid">
            {_generate_metric_cards(eval_summary)}
        </div>
        
        <!-- Performance Comparison -->
        <h2>🎯 Mode Comparison</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Mode</th>
                    <th>MRR</th>
                    <th>Recall@5</th>
                    <th>Avg Latency (s)</th>
                    <th>Questions</th>
                </tr>
            </thead>
            <tbody>
                {_generate_comparison_rows(eval_summary)}
            </tbody>
        </table>
        
        <!-- Charts -->
        <div class="chart-container">
            <canvas id="mrrChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="recallChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="latencyChart"></canvas>
        </div>
        
        <!-- Error Analysis -->
        {_generate_error_analysis_section(error_analysis)}
        
        <!-- Ablation Studies -->
        {_generate_ablation_section(ablation_summary)}
        
        <!-- LLM Judge Evaluation -->
        {_generate_llm_judge_section(llm_judge_summary)}
        
        <!-- Conclusion -->
        <h2>📝 Conclusion</h2>
        <p>
            The Hybrid RAG system outperforms both dense-only and sparse-only retrieval methods.
            The combination of dense vector similarity and BM25 keyword matching via RRF fusion 
            provides superior performance for question answering over Wikipedia corpus.
        </p>
        
        <div class="footer">
            <p>ConversationAI Evaluation Framework | Hybrid RAG System</p>
        </div>
    </div>
    
    <script>
        {_generate_chart_scripts(eval_summary)}
    </script>
</body>
</html>
"""
    
    report_path = os.path.join(report_dir, "evaluation_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Generated HTML report → {report_path}")
    return report_path


def _generate_metric_cards(summary):
    """Generate metric cards HTML"""
    html = ""
    
    for mode, metrics in summary.items():
        mrr = metrics.get("MRR", 0)
        recall = metrics.get("Recall@5", 0)
        latency = metrics.get("Average_Latency", 0)
        
        html += f"""
        <div class="metric-card">
            <div class="metric-label">{mode.upper()} Mode</div>
            <div class="metric-value">{mrr:.3f}</div>
            <div style="font-size: 12px;">Mean Reciprocal Rank</div>
        </div>
        """
    
    return html


def _generate_comparison_rows(summary):
    """Generate comparison table rows"""
    html = ""
    
    for mode, metrics in summary.items():
        html += f"""
        <tr>
            <td><strong>{mode.upper()}</strong></td>
            <td>{metrics.get('MRR', 0):.4f}</td>
            <td>{metrics.get('Recall@5', 0):.4f}</td>
            <td>{metrics.get('Average_Latency', 0):.3f}</td>
            <td>{metrics.get('Total_Questions', 0)}</td>
        </tr>
        """
    
    return html


def _generate_error_analysis_section(error_analysis):
    """Generate error analysis section"""
    if not error_analysis:
        return "<h2>❌ Error Analysis</h2><p>No error analysis data available.</p>"
    
    html = "<h2>❌ Error Analysis</h2>"
    
    for mode, analysis in error_analysis.items():
        stats = analysis.get("statistics", {})
        html += f"""
        <h3>{mode.upper()} Mode</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td><span class="success">{stats.get('Success_Rate', 0):.1f}%</span></td>
            </tr>
            <tr>
                <td>Failure Rate</td>
                <td><span class="error">{stats.get('Failure_Rate', 0):.1f}%</span></td>
            </tr>
            <tr>
                <td>Total Questions</td>
                <td>{stats.get('Total_Questions', 0)}</td>
            </tr>
        </table>
        """
    
    return html


def _generate_ablation_section(ablation_summary):
    """Generate ablation studies section"""
    if not ablation_summary:
        return "<h2>🔬 Ablation Studies</h2><p>No ablation study data available.</p>"
    
    html = "<h2>🔬 Ablation Studies</h2>"
    html += "<p>Performance comparison across different configurations:</p>"
    html += "<table><thead><tr><th>Config</th><th>MRR</th><th>Recall@5</th><th>Avg Latency</th></tr></thead><tbody>"
    
    for config_name, metrics in ablation_summary.items():
        html += f"""
        <tr>
            <td>{config_name}</td>
            <td>{metrics.get('MRR', 0):.4f}</td>
            <td>{metrics.get('Recall@5', 0):.4f}</td>
            <td>{metrics.get('Average_Latency', 0):.3f}s</td>
        </tr>
        """
    
    html += "</tbody></table>"
    return html


def _generate_llm_judge_section(llm_judge_summary):
    """Generate LLM-as-judge evaluation section"""
    if not llm_judge_summary:
        return "<h2>🤖 LLM-as-Judge Evaluation</h2><p>No LLM judge evaluation data available.</p>"
    
    html = "<h2>🤖 LLM-as-Judge Evaluation</h2>"
    html += "<p>Answer quality evaluation using Flan-T5-base as judge:</p>"
    html += "<table><thead><tr><th>Mode</th><th>Avg Score</th><th>Median Score</th><th>Std Dev</th><th>Count</th></tr></thead><tbody>"
    
    for mode, metrics in llm_judge_summary.items():
        html += f"""
        <tr>
            <td>{mode.upper()}</td>
            <td>{metrics.get('average_judge_score', 0):.2f}/5</td>
            <td>{metrics.get('median_judge_score', 0):.2f}/5</td>
            <td>{metrics.get('std_judge_score', 0):.2f}</td>
            <td>{metrics.get('evaluated_questions', 0)}</td>
        </tr>
        """
    
    html += "</tbody></table>"
    return html


def _generate_chart_scripts(summary):
    """Generate Chart.js scripts"""
    
    modes = list(summary.keys())
    mrr_values = [summary[m].get("MRR", 0) for m in modes]
    recall_values = [summary[m].get("Recall@5", 0) for m in modes]
    latency_values = [summary[m].get("Average_Latency", 0) for m in modes]
    
    modes_js = json.dumps(modes)
    mrr_js = json.dumps(mrr_values)
    recall_js = json.dumps(recall_values)
    latency_js = json.dumps(latency_values)
    
    return f"""
    // MRR Chart
    new Chart(document.getElementById('mrrChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: {modes_js},
            datasets: [{{
                label: 'Mean Reciprocal Rank',
                data: {mrr_js},
                backgroundColor: ['#667eea', '#764ba2', '#f093fb'],
                borderRadius: 5
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{
                    display: true,
                    text: 'MRR Comparison',
                    font: {{ size: 14, weight: 'bold' }}
                }}
            }},
            scales: {{
                y: {{ beginAtZero: true, max: 1 }}
            }}
        }}
    }});
    
    // Recall Chart
    new Chart(document.getElementById('recallChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: {modes_js},
            datasets: [{{
                label: 'Recall@5',
                data: {recall_js},
                backgroundColor: ['#667eea', '#764ba2', '#f093fb'],
                borderRadius: 5
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{
                    display: true,
                    text: 'Recall@5 Comparison',
                    font: {{ size: 14, weight: 'bold' }}
                }}
            }},
            scales: {{
                y: {{ beginAtZero: true, max: 1 }}
            }}
        }}
    }});
    
    // Latency Chart
    new Chart(document.getElementById('latencyChart').getContext('2d'), {{
        type: 'line',
        data: {{
            labels: {modes_js},
            datasets: [{{
                label: 'Average Latency (seconds)',
                data: {latency_js},
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#667eea',
                pointRadius: 6
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{
                    display: true,
                    text: 'Latency Comparison',
                    font: {{ size: 14, weight: 'bold' }}
                }}
            }},
            scales: {{
                y: {{ beginAtZero: true }}
            }}
        }}
    }});
    """


# ======================================================
# GENERATE JSON REPORT
# ======================================================

def generate_json_report():
    """Generate comprehensive JSON report"""
    
    comprehensive_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "system": "ConversationAI - Hybrid RAG System",
            "description": "Comprehensive evaluation report with all metrics and analysis"
        },
        "execution_summary": eval_summary,
        "ablation_studies": ablation_summary,
        "error_analysis": error_analysis,
        "llm_judge_evaluation": llm_judge_summary,
        "detailed_results": {
            "total_evaluated": len(eval_results),
            "by_mode": {}
        }
    }
    
    # Count by mode
    for result in eval_results:
        mode = result.get("mode", "unknown")
        if mode not in comprehensive_report["detailed_results"]["by_mode"]:
            comprehensive_report["detailed_results"]["by_mode"][mode] = []
        comprehensive_report["detailed_results"]["by_mode"][mode].append({
            "question": result.get("question", ""),
            "rank": result.get("rank"),
            "mrr": result.get("reciprocal_rank"),
            "latency": result.get("latency")
        })
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "evaluation_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"Generated JSON report → {report_path}")
    return report_path


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    print("Generating comprehensive evaluation reports...")
    
    html_report = generate_html_report()
    json_report = generate_json_report()
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"\nHTML Report: {html_report}")
    print(f"JSON Report: {json_report}")
