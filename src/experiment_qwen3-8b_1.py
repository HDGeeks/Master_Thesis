"""Experiment 1: check output quality of a newer open model (Qwen3-8B,
run locally) on the paper's own zero-shot topic labeling task, and compare
the paper's original exact-match evaluation against matching.py.

This talks to the Ollama server running on this machine (localhost:11434),
not a cloud API. The model runs entirely on this laptop's CPU.

Reproduces the exact 3-turn prompt structure from the paper (Figure 2),
run through a local Qwen3-8B model via Ollama instead of the paper's
GPT4All/Llama 3 setup.

Requires:
- Ollama running locally (ollama serve) with the model imported as
  "qwen3-8b-local" (see models/Modelfile)
- The D3 example dataset (reference-repo/data/assets_example/metadata.json
  and targets.json)
"""

import json
import os
import random
import re

import ollama
from matching import match_topic, canonicalize

MODEL_NAME = "qwen3-8b-local"
NUM_SAMPLE_DOCS = 20
RANDOM_SEED = 42

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_qwen3-8b_1_results.json"
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


def build_prompts(title, vocabulary):
    targets_string = ", ".join(vocabulary)

    prompt_1 = "We want to create a list of topics in the following. We call this list targets_list."
    prompt_2 = (
        "Here are some topics that should be added to the targets_list: "
        + targets_string
        + ". Please use the exact spelling that I provide to you."
    )
    prompt_3 = (
        "We now want to annotate a title with the topics provided in the targets_list.\n"
        f"Given the following title: {title}\n"
        "Please assign 1 suitable topic from the targets_list to the title.\n"
        "This topic should be contained in the targets_list we created earlier and use the exact "
        "spelling of the topic in the targets_list.\n"
        "Please respond only with the 1 topic without any further text."
    )
    return [prompt_1, prompt_2, prompt_3]


def strip_thinking(text):
    """Qwen3 can emit a <think>...</think> reasoning block before the real
    answer. Strip it, we only want the final answer text, same as the paper
    only wanted the plain topic string."""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_chat_session(prompts):
    """Run a genuine multi-turn conversation, same as GPT4All's chat_session:
    each prompt gets a real reply, and the reply is kept in history for the
    next turn. Only the final reply is the answer we care about."""

    messages = []
    final_reply = None

    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        raw_reply = response["message"]["content"]
        final_reply = strip_thinking(raw_reply)
        messages.append({"role": "assistant", "content": raw_reply})

    return final_reply


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
    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        title = doc["title"]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {title}")

        prompts = build_prompts(title, vocabulary)
        raw_answer = run_chat_session(prompts)

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
