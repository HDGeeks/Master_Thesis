"""Print summary stats for an experiment results file: for each matching
label (old vs new), how many results fall into each category.

Usage:
    python3 analyze_results.py ../results/Qwen3.5-9B/title/normal_bit/experiment_Qwen3.5-9B_title_<timestamp>_results.json
"""

import argparse
import json


def count_categories(results, label):
    """Count how many results fall into each category ("success",
    "misclassified", "hallucination") for a given matching label
    ("old_matching" or "new_matching")."""

    counts = {"success": 0, "misclassified": 0, "hallucination": 0}
    for r in results:
        counts[r[label]["category"]] += 1
    return counts


def print_stats(results):
    total = len(results)
    print(f"Total documents: {total}\n")

    for label in ("old_matching", "new_matching"):
        counts = count_categories(results, label)

        print(f"{label}:")
        for category, count in counts.items():
            print(f"  {category}: {count} ({count / total:.1%})")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="Path to an experiment results JSON file")
    args = parser.parse_args()

    with open(args.results_path) as f:
        results = json.load(f)

    print_stats(results)


if __name__ == "__main__":
    main()