#!/usr/bin/env python3
"""
Main runner script for the DeepEval evaluation framework.
Supports running evaluations for chatbot, RAG, or both with configurable LLMs.

Usage:
  python run_eval.py                          # Run all evaluations
  python run_eval.py --target chatbot         # Chatbot only
  python run_eval.py --target rag             # RAG only
  python run_eval.py --judge openai           # Use OpenAI as judge
  python run_eval.py --judge grok             # Use Grok as judge
  python run_eval.py --judge gemma            # Use local Gemma as judge
  python run_eval.py --metric toxicity        # Run specific metric only
  python run_eval.py --list-metrics           # List all available metrics
"""
import argparse
import subprocess
import sys
import os

METRICS = [
    "answer_relevancy", "faithfulness", "hallucination", "toxicity", "bias",
    "contextual_precision", "contextual_recall", "contextual_relevancy",
    "correctness", "coherence", "completeness", "conciseness",
    "helpfulness", "politeness", "safety",  # chatbot-specific
    "groundedness", "retrieval_quality",     # RAG-specific
]


def main():
    parser = argparse.ArgumentParser(description="DeepEval Evaluation Runner")
    parser.add_argument("--target", choices=["chatbot", "rag", "all"], default="all", help="What to evaluate")
    parser.add_argument("--judge", choices=["openai", "grok", "groq", "groq_oss120b", "groq_scout", "groq_qwen", "oss_120b", "gemma"], help="Judge LLM to use")
    parser.add_argument("--generator", choices=["openai", "grok", "groq", "groq_oss120b", "groq_scout", "groq_qwen", "gemma"], help="Generator LLM to use")
    parser.add_argument("--metric", help="Run specific metric only (e.g., toxicity, hallucination)")
    parser.add_argument("--list-metrics", action="store_true", help="List all available metrics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.list_metrics:
        print("Available metrics:")
        for m in METRICS:
            print(f"  • {m}")
        return

    # Set environment variables for LLM selection
    if args.judge:
        os.environ["JUDGE_LLM"] = args.judge
    if args.generator:
        os.environ["GENERATOR_LLM"] = args.generator

    print("=" * 60)
    print("  DeepEval Evaluation Framework")
    print(f"  Judge LLM:     {os.environ.get('JUDGE_LLM', 'openai')}")
    print(f"  Generator LLM: {os.environ.get('GENERATOR_LLM', 'gemma')}")
    print(f"  Target:        {args.target}")
    if args.metric:
        print(f"  Metric:        {args.metric}")
    print("=" * 60)

    test_files = []
    if args.target in ("chatbot", "all"):
        test_files.append("test_chatbot_eval.py")
    if args.target in ("rag", "all"):
        test_files.append("test_rag_eval.py")

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(test_files)

    if args.metric:
        cmd.extend(["-k", args.metric])

    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-v")

    cmd.append("--tb=short")

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
