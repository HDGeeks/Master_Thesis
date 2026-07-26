"""Experiment 1: check output quality of a newer open model (Qwen3-8B,
run locally) on the paper's own zero-shot topic labeling task, and compare
the paper's original exact-match evaluation against matching.py.

Runs entirely on this machine via mlx-lm (Apple's local inference library,
Metal-accelerated), no server, no cloud API, no network calls at inference
time. The model weights live in models/Qwen3-8B-4bit-mlx.

Uses the same content as the paper's 3-turn prompt (Figure 2), combined
into a single prompt so the model only runs once per title instead of
three times, the paper split it into 3 turns to manage prompt length for
their model, we don't need that here.

Requires:
- The D3 example dataset (reference-repo/data/assets_example/metadata.json
  and targets.json)
- models/Qwen3-8B-4bit-mlx, downloaded automatically on first run if it's
  not already there (~4.4GB, so first run on a new machine takes a while)
"""

import json
import os
import random
import re

from mlx_lm import generate, load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from matching import match_topic, canonicalize

HF_REPO = "mlx-community/Qwen3-8B-4bit"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-8B-4bit-mlx")
NUM_SAMPLE_DOCS = 20
RANDOM_SEED = 42

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_qwen3-8b_1_results.json"
)


def ensure_model_downloaded():
    """Download the model into models/ if it isn't there yet, so this
    script can just be run on a fresh laptop without a manual setup step."""

    weights_file = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(weights_file):
        return

    print(f"Model not found at {MODEL_PATH}, downloading {HF_REPO} (~4.4GB) ...")
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, local_dir=MODEL_PATH)
    print("Download complete.")


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


def build_prompt(title, vocabulary):
    targets_string = ", ".join(vocabulary)

    return (
        "We want to create a list of topics. We call this list targets_list.\n"
        "Here are the topics in the targets_list: " + targets_string + ". "
        "Please use the exact spelling that I provide to you.\n"
        "We now want to annotate a title with the topics provided in the targets_list.\n"
        f"Given the following title: {title}\n"
        "Please assign 1 suitable topic from the targets_list to the title.\n"
        "This topic should be contained in the targets_list and use the exact "
        "spelling of the topic in the targets_list.\n"
        "Please respond only with the 1 topic without any further text."
    )


def strip_thinking(text):
    """Qwen3 can emit a <think>...</think> reasoning block before the real
    answer. Strip it, we only want the final answer text, same as the paper
    only wanted the plain topic string.

    If <think> appears without a closing </think> (generation got cut off
    mid-thought), there is no real answer to extract. Return an empty
    string rather than leaking the raw reasoning text into matching, an
    unclosed block matching a topic by accident is not a real answer."""

    if "<think>" in text and "</think>" not in text:
        return ""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(model, tokenizer: TokenizerWrapper, prompt):
    """Run the model once on a single prompt and return the answer."""

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    raw_reply = generate(model, tokenizer, prompt=chat_prompt, max_tokens=500, verbose=False)
    return strip_thinking(raw_reply)


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
    ensure_model_downloaded()

    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    print(f"Loading model from {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)

    results = []
    for i, doc in enumerate(documents, start=1):
        title = doc["title"]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {title}")

        prompt = build_prompt(title, vocabulary)
        raw_answer = ask_model(model, tokenizer, prompt)

        old_cat, old_topic, new_cat, new_topic = classify_answer(
            raw_answer, vocabulary, ground_truth
        )

        print(f"  raw answer: {raw_answer!r}")
        print(f"  old matching: {old_cat} ({old_topic})")
        print(f"  new matching: {new_cat} ({new_topic})")

        results.append({
            "D3 ID": doc["D3 ID"],
            "title": title,
            "ground_truth_subjects": ground_truth,
            "raw_answer": raw_answer,
            "old_matching": {"category": old_cat, "topic": old_topic},
            "new_matching": {"category": new_cat, "topic": new_topic},
        })

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
