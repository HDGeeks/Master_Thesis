"""Experiment 1: check output quality of a newer model (Claude) on the
paper's own zero-shot topic labeling task, and compare the paper's original
exact-match evaluation against the improved matching in matching.py.

Reproduces the exact 3-turn prompt structure from the paper (Figure 2),
run through the Anthropic API instead of the local GPT4All/Llama 3 setup.

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
NUM_SAMPLE_DOCS = 20
RANDOM_SEED = 42

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", f"experiment_1_{RUN_TIMESTAMP}_results.json"
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


def run_chat_session(client, prompts):
    """Run a genuine multi-turn conversation, same as GPT4All's chat_session:
    each prompt gets a real reply, and the reply is kept in history for the
    next turn. Only the final reply is the answer we care about."""

    messages = []
    final_reply = None

    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=messages,
        )
        final_reply = response.content[0].text
        messages.append({"role": "assistant", "content": final_reply})

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
    client = anthropic.Anthropic()
    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        title = doc["title"]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {title}")

        prompts = build_prompts(title, vocabulary)
        raw_answer = run_chat_session(client, prompts)

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
