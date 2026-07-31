"""Experiment 1: check output quality of a newer open model (Qwen3-8B) on
the paper's own zero-shot topic labeling task, and compare the paper's
original exact-match evaluation against matching.py.

Linux/CPU version, for a server-grade machine with no GPU. Uses
llama-cpp-python (CPU backend, multi-threaded) instead of Ollama
(experiment_qwen3-8b_linux_2.py) or mlx-lm (Mac-only, macos_1.py). Same
single-prompt approach, 1 model call per document.

Serves the GGUF model straight from the shared institutional path, no
download or import step, the file is expected to already exist there
(see /localdata/dagstuhl/ai_models/huggingface-selective on helmholtz,
downloaded once via the institutional `hf download` procedure).
"""

import datetime
import json
import os
import random
import re

from llama_cpp import Llama
from matching import match_topic, canonicalize

# Shared institutional model path, per helmholtz's Hugging Face storage
# procedure. Downloaded once via `hf download`, shared across all users.
GGUF_PATH = "/localdata/dagstuhl/ai_models/huggingface-selective/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

NUM_SAMPLE_DOCS = 2500
RANDOM_SEED = 42

# Which field of each document to feed the model as input.
# Set to "title" to use only the title, or "abstract" to use only the abstract.
INPUT_FIELD = "abstract"  # "title" or "abstract"

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results",
    f"experiment_qwen3-8b_linux_3_{INPUT_FIELD}_{RUN_TIMESTAMP}_results.json",
)


def load_model():
    """Load the GGUF model once, kept in memory for all 2500 calls
    (one long-lived process instead of Ollama's daemon + keep_alive)."""

    if not os.path.exists(GGUF_PATH):
        raise FileNotFoundError(
            f"Model not found at {GGUF_PATH}. Download it once via the "
            "institutional hf download procedure, this script does not "
            "fetch it automatically."
        )

    return Llama(
        model_path=GGUF_PATH,
        n_ctx=4096,      # abstracts run up to ~1700 tokens plus the ~200 token vocab list
        n_threads=12,    # use all available CPU cores instead of relying on auto-detection
        n_batch=1024,    # process more prompt tokens in parallel, speeds up longer prompts
        verbose=False,   # suppress llama.cpp's per-call load logging
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
        "Please respond only with the 1 topic without any further text. /no_think"
    )


def strip_thinking(text):
    """Same safety net as the other versions: Qwen3 can emit a
    <think>...</think> reasoning block. If it's unclosed (cut off by the
    token limit), there is no real answer to extract, return empty rather
    than leaking the raw reasoning text into matching."""

    if "<think>" in text and "</think>" not in text:
        return ""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(llm, prompt):
    """Run the model once on a single prompt and return the answer."""

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,  # short cap, the answer is just a topic name, no reasoning block to account for
    )
    raw_text = response["choices"][0]["message"]["content"]
    return strip_thinking(raw_text)


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
    llm = load_model()

    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        # Pull whichever field INPUT_FIELD points to (title or abstract) as the model input.
        input_text = doc[INPUT_FIELD]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {input_text}")

        prompt = build_prompt(input_text, vocabulary)
        raw_answer = ask_model(llm, prompt)

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
        # through the 2500 queries doesn't lose everything done so far.
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
