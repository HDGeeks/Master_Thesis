"""Experiment 1: check output quality of a newer open model (Qwen3-8B) on
the paper's own zero-shot topic labeling task, and compare the paper's
original exact-match evaluation against matching.py.

Linux/CPU version, for a server-grade machine with no GPU. Uses Ollama
(CPU backend, multi-threaded) instead of mlx-lm, which is Mac-only. Same
single-prompt approach as the Mac version (src/experiment_qwen3-8b_1.py),
1 model call per document.

Requires Ollama installed and running (ollama serve), install with:
    curl -fsSL https://ollama.com/install.sh | sh

Downloads the GGUF model file into models/ automatically on first run if
it isn't there yet, and imports it into Ollama automatically too, so this
script can just be run on a fresh machine with no manual setup step.
"""

import json
import os
import random
import re
import subprocess

import ollama
from matching import match_topic, canonicalize

HF_URL = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
GGUF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-8B-Q4_K_M.gguf")
OLLAMA_MODEL_NAME = "qwen3-8b-local"
NUM_SAMPLE_DOCS = 2500
RANDOM_SEED = 42

# Which field of each document to feed the model as input.
# Set to "title" to use only the title, or "abstract" to use only the abstract.
INPUT_FIELD = "abstract"  # "title" or "abstract"

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results",
    f"experiment_qwen3-8b_linux_1_{INPUT_FIELD}_results.json",
)


def ensure_model_downloaded():
    """Download the GGUF into models/ if it isn't there yet."""

    if os.path.exists(GGUF_PATH):
        return

    print(f"Model not found at {GGUF_PATH}, downloading Qwen3-8B-Q4_K_M.gguf (~4.8GB) ...")
    import requests

    os.makedirs(os.path.dirname(GGUF_PATH), exist_ok=True)
    response = requests.get(HF_URL, stream=True)
    response.raise_for_status()
    with open(GGUF_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print("Download complete.")


def ensure_ollama_model_ready():
    """Import the GGUF into Ollama under OLLAMA_MODEL_NAME if it isn't
    already imported."""

    existing_models = [m.model for m in ollama.list().models]
    if any(name.startswith(OLLAMA_MODEL_NAME) for name in existing_models):
        return

    print(f"Importing {GGUF_PATH} into Ollama as {OLLAMA_MODEL_NAME} ...")
    modelfile_path = os.path.join(os.path.dirname(GGUF_PATH), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {os.path.basename(GGUF_PATH)}\n")

    subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", "Modelfile"],
        cwd=os.path.dirname(GGUF_PATH),
        check=True,
    )
    print("Import complete.")


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


def strip_thinking(text):
    """Same safety net as the Mac version: Qwen3 can emit a
    <think>...</think> reasoning block. If it's unclosed (cut off by the
    token limit), there is no real answer to extract, return empty rather
    than leaking the raw reasoning text into matching."""

    if "<think>" in text and "</think>" not in text:
        return ""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(prompt):
    """Run the model once on a single prompt and return the answer."""

    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        keep_alive="30m",  # keep the model loaded in RAM between the 2500 calls instead of reloading it each time
        think=False,  # same intent as enable_thinking=False in the Mac version, skip the reasoning block since we only want the final topic answer
        options={
            "num_predict": 100,  # short cap, the answer is just a topic name (or a few comma-separated candidates), no reasoning block to account for anymore
        },
    )
    return strip_thinking(response["message"]["content"])


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
    ensure_ollama_model_ready()

    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        # Pull whichever field INPUT_FIELD points to (title or abstract) as the model input.
        input_text = doc[INPUT_FIELD]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {input_text}")

        prompt = build_prompt(input_text, vocabulary)
        raw_answer = ask_model(prompt)

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