"""Dry-run cost estimate for the single-call version of Experiment 1.

This makes zero API calls and needs no API key. It builds the exact same
prompts the real script would send, then estimates token counts locally
using a word-based heuristic instead of Anthropic's real tokenizer.

Heuristic: English technical prose runs at roughly 1.3 tokens per word
(short common words are often 1 token, longer or technical words can split
into 2 or more). This is an approximation, expect it to land within
roughly 10 to 20 percent of the real number. Once you have an API key,
rerun the real sampling script (estimate_cost_single_call.py) on a small
batch to calibrate against actual billed tokens.

Requires only the same metadata.json / targets.json files as the main
experiment script. No network access needed.
"""

import argparse
import json
import os

# Rough Anthropic-style tokens per English word, based on published token
# to word ratios for technical text. Adjust this if you calibrate against
# real usage later and find it consistently off.
TOKENS_PER_WORD = 1.3

# Pricing as of August 2026, dollars per million tokens.
INPUT_PRICE_PER_MTOK = 1.0
OUTPUT_PRICE_PER_MTOK = 5.0

# The model replies with a single topic label, this is a fixed small
# estimate for output length since there is no real response to measure.
ESTIMATED_OUTPUT_TOKENS = 8

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)


def load_vocabulary():
    """Load the topic vocabulary the model is asked to choose from."""
    path = os.path.join(DATASET_DIR, "targets.json")
    with open(path, "r") as f:
        return json.load(f)


def load_documents():
    """Load the document metadata (titles, abstracts, ground-truth subjects)."""
    path = os.path.join(DATASET_DIR, "metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(text, vocabulary, is_abstract=False):
    """Same prompt text the single-call experiment script actually sends.

    This is either/or, not both at once: text is the title in the title
    case, or the abstract in the abstract case, never both combined."""

    targets_string = ", ".join(vocabulary)

    label = "abstract" if is_abstract else "title"
    body = f"Given the following {label}: {text}"

    return (
        "We want to create a list of topics. We call this list targets_list.\n"
        f"The targets_list contains the following topics: {targets_string}. "
        "Please use the exact spelling that I provide to you.\n\n"
        "We now want to annotate a title with the topics provided in the targets_list.\n"
        f"{body}\n"
        "Please assign 1 suitable topic from the targets_list to the title.\n"
        "This topic should be contained in the targets_list we created earlier and use the exact "
        "spelling of the topic in the targets_list.\n"
        "Please respond only with the 1 topic without any further text."
    )


def estimate_tokens(text):
    """Local, offline token estimate. Word count times a fixed ratio,
    no API call and no real tokenizer involved."""
    word_count = len(text.split())
    return round(word_count * TOKENS_PER_WORD)


def call_cost(input_tokens, output_tokens):
    """Convert estimated token counts into a dollar amount for one call."""
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )


def estimate_case(vocabulary, documents, use_abstract, num_docs):
    """Estimate average per-call cost for one case over every document
    actually available, then scale to num_docs.

    use_abstract picks which single field goes into the prompt, title or
    abstract, never both at the same time."""

    costs = []
    for doc in documents:
        text = doc.get("abstract") if use_abstract else doc["title"]
        prompt = build_prompt(text, vocabulary, is_abstract=use_abstract)
        input_tokens = estimate_tokens(prompt)
        costs.append(call_cost(input_tokens, ESTIMATED_OUTPUT_TOKENS))

    average_cost = sum(costs) / len(costs)
    return average_cost, average_cost * num_docs


def main():
    parser = argparse.ArgumentParser(
        description="Offline dry-run cost estimate, no API key or network needed."
    )
    parser.add_argument(
        "--num-docs", type=int, default=2500,
        help="total documents you plan to run per case (default: 2500)",
    )
    args = parser.parse_args()

    vocabulary = load_vocabulary()
    documents = load_documents()

    print(f"Estimating from {len(documents)} real documents in metadata.json, no API calls made.\n")

    title_avg, title_total = estimate_case(vocabulary, documents, use_abstract=False, num_docs=args.num_docs)
    abstract_avg, abstract_total = estimate_case(vocabulary, documents, use_abstract=True, num_docs=args.num_docs)

    print(f"Title only:     ~${title_avg:.5f} / call  ->  ~${title_total:,.2f} for {args.num_docs} docs")
    print(f"Abstract only:  ~${abstract_avg:.5f} / call  ->  ~${abstract_total:,.2f} for {args.num_docs} docs")
    print(f"\nCombined estimated total (both cases run separately): ~${title_total + abstract_total:,.2f}")
    print("\nThis is a local word-count estimate, not real tokenizer output.")
    print("Treat it as a ballpark to decide whether to get an API key at all.")


if __name__ == "__main__":
    main()