"""Experiment 1: check output quality of a newer model (Claude) on the
paper's own zero-shot topic labeling task, and compare the paper's original
exact-match evaluation against the improved matching in matching.py.

Single-prompt approach, 1 API call per document, matching the structure of
the Qwen3-8B scripts (INPUT_FIELD toggle, incremental save). The original
version reproduced the paper's exact 3-turn chat_session structure, this
version drops that reproduction requirement in favor of one call per
document, which is both cheaper and easier to compare directly against the
Qwen runs.

Zero-shot: no extended thinking, no system prompt, no examples, just the
single instruction prompt and the model's direct answer.

Requires:
- ANTHROPIC_API_KEY environment variable
- The D3 example dataset downloaded via the cloned reference repo
  (reference-repo/data/assets_example/metadata.json and targets.json)
"""

import datetime
import json
import os
import random

import anthropic
from dotenv import load_dotenv
from matching import match_topic, canonicalize

load_dotenv()

MODEL_NAME = "claude-haiku-4-5-20251001"
NUM_SAMPLE_DOCS = 2500
RANDOM_SEED = 42

# Which field of each document to feed the model as input.
# Set to "title" to use only the title, or "abstract" to use only the abstract.
INPUT_FIELD = "title"  # "title" or "abstract"

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results",
    f"experiment_1_claude_{INPUT_FIELD}_{RUN_TIMESTAMP}_results.json",
)


def load_sample_documents(num_docs):
    metadata_path = os.path.join(DATASET_DIR, "metadata.json")
    with open(metadata_path, "r") as f:
        publications = json.load(f)

    random.seed(RANDOM_SEED)
    return random.sample(publications, num_docs)


def load_vocabulary():
    targets_path = os.path.join(DATASET_DIR, "targets.json")
    with open(targets_path, "r") as f:
        targets = json.load(f)
    return [canonicalize(t) for t in targets]


def build_prompt(input_text, vocabulary):
    """Build the prompt. input_text is whatever field INPUT_FIELD points
    to (title or abstract), the wording adapts to name that field so the
    model gets an accurate description of what it is reading."""

    targets_string = ", ".join(vocabulary)
    field_label = INPUT_FIELD  # "title" or "abstract", used directly in the prompt text

    return (
        "We want to create a list of topics. We call this list targets_list.\n"
        "Here are the topics in the targets_list: " + targets_string + ". "
        "Please use the exact spelling that I provide to you.\n"
        f"We now want to annotate a {field_label} with the topics provided in the targets_list.\n"
        f"Given the following {field_label}: {input_text}\n"
        "Please assign 1 suitable topic from the targets_list to the "
        f"{field_label}.\n"
        "This topic should be contained in the targets_list and use the exact "
        "spelling of the topic in the targets_list.\n"
        "Please respond only with the 1 topic without any further text."
    )


def ask_model(client, prompt):
    """Run the model once on a single prompt and return the answer.

    Zero-shot, plain single-turn call. No thinking parameter is passed, so
    extended thinking stays off by default, no reasoning block to strip
    from the output (unlike the Qwen version, which needs /no_think and
    strip_thinking for the same effect)."""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=100,  # short cap, the answer is just a topic name, no reasoning to account for
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def classify_answer(raw_answer, vocabulary, ground_truth_subjects):
    """Return (category, matched_topic) for both the old exact-match-only
    rule and the new matching.py rule."""

    candidates = [c.strip() for c in raw_answer.split(",")]

    def evaluate(match_fn):
        for candidate in candidates:
            topic = match_fn(candidate)
            if topic is not None:
                if topic in ground_truth_subjects:
                    return "success", topic
                return "misclassified", topic
        return "hallucination", None

    old_category, old_topic = evaluate(
        lambda c: canonicalize(c) if canonicalize(c) in vocabulary else None
    )
    new_category, new_topic = evaluate(lambda c: match_topic(c, vocabulary))

    return old_category, old_topic, new_category, new_topic


def main():
    client = anthropic.Anthropic()
    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        # Pull whichever field INPUT_FIELD points to (title or abstract) as the model input.
        input_text = doc[INPUT_FIELD]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {input_text}")

        prompt = build_prompt(input_text, vocabulary)
        raw_answer = ask_model(client, prompt)

        old_cat, old_topic, new_cat, new_topic = classify_answer(
            raw_answer, vocabulary, ground_truth
        )

        print(f"  raw answer: {raw_answer!r}")
        print(f"  old matching: {old_cat} ({old_topic})")
        print(f"  new matching: {new_cat} ({new_topic})")

        results.append({
            "D3 ID": doc["D3 ID"],
            "input_field": INPUT_FIELD,
            "input_text": input_text,
            "ground_truth_subjects": ground_truth,
            "raw_answer": raw_answer,
            "old_matching": {"category": old_cat, "topic": old_topic},
            "new_matching": {"category": new_cat, "topic": new_topic},
        })

        # Save after every document, not just at the end, so a crash midway
        # through the run doesn't lose everything done so far.
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {RESULTS_PATH}")
    print_summary(results)


def print_summary(results):
    for label in ("old_matching", "new_matching"):
        counts = {"success": 0, "misclassified": 0, "hallucination": 0}
        for r in results:
            counts[r[label]["category"]] += 1
        print(f"{label}: {counts}")


if __name__ == "__main__":
    main()