"""Summarize an experiment results file: overall stats plus a random
sample of cases to manually read through.

Usage:
    python3 analyze_results.py ../results/experiment_qwen3-8b_linux_1_results.json
    python3 analyze_results.py ../results/experiment_qwen3-8b_linux_1_results.json --sample 20
"""

import argparse
import json
import random


def input_text_of(r):
    """Older results files have a "title" field, newer ones (that can run
    on either title or abstract) have "input_text" instead. Handle both."""

    return r["input_text"] if "input_text" in r else r["title"]


def print_stats(results):
    print(f"Total documents: {len(results)}\n")

    for label in ("old_matching", "new_matching"):
        counts = {"success": 0, "misclassified": 0, "hallucination": 0}
        for r in results:
            counts[r[label]["category"]] += 1

        total = len(results)
        print(f"{label}:")
        for category, count in counts.items():
            print(f"  {category}: {count} ({count / total:.1%})")
        print()

    differing = [r for r in results if r["old_matching"]["category"] != r["new_matching"]["category"]]
    print(f"Cases where the matching fix changed the outcome: {len(differing)}")
    for r in differing:
        print(f"  {input_text_of(r)}")
        print(f"    raw answer: {r['raw_answer']!r}")
        print(f"    old: {r['old_matching']['category']} -> new: {r['new_matching']['category']}")


def print_sample(results, sample_size, seed):
    random.seed(seed)
    sample = random.sample(results, min(sample_size, len(results)))

    print(f"\n--- Random sample of {len(sample)} cases ---\n")
    for r in sample:
        print(f"Input: {input_text_of(r)}")
        print(f"  ground truth: {r['ground_truth_subjects']}")
        print(f"  raw answer: {r['raw_answer']!r}")
        print(f"  new matching: {r['new_matching']['category']} ({r['new_matching']['topic']})")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="Path to an experiment results JSON file")
    parser.add_argument("--sample", type=int, default=20, help="Number of random cases to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the sample")
    args = parser.parse_args()

    with open(args.results_path) as f:
        results = json.load(f)

    print_stats(results)
    print_sample(results, args.sample, args.seed)


if __name__ == "__main__":
    main()
