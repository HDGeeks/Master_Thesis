"""Experiment 1: check output quality of Qwen3.5 (any size) on the
paper's own zero-shot topic labeling task, and compare the paper's
original exact-match evaluation against matching.py.

Hugging Face transformers version, for any Qwen3.5 checkpoint stored
locally under /localdata/dagstuhl/ai_models/huggingface-full/. Uses
AutoModelForCausalLM instead of llama-cpp-python
(experiment_qwen3-8b_cpp_linux.py), Ollama
(experiment_qwen3-8b_linux_2.py), or mlx-lm (Mac-only, macos_1.py). Same
single-prompt approach, 1 model call per document, same prompt,
vocabulary, document sample, matching logic, and output schema as the
other three versions, so results are directly comparable.

Model path is a global variable below, edit it directly to point at
whichever local checkpoint you want to run (0.8B, 2B, 4B, 9B, 27B).
"""

import datetime
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from matching import match_topic, canonicalize

# Path to the local Qwen3.5 model folder to load. Change this to switch
# checkpoints, e.g. Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B.
MODEL_PATH = "/localdata/dagstuhl/ai_models/huggingface-full/Qwen/Qwen3.5-9B"

NUM_SAMPLE_DOCS = 2500

# Which field of each document to feed the model as input.
# Set to "title" to use only the title, or "abstract" to use only the abstract.
INPUT_FIELD = "abstract"  # "title" or "abstract"

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reference-repo", "data", "assets_example"
)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model():
    """Load the tokenizer and model once, kept in memory for all 2500
    calls (one long-lived process, same idea as the llama.cpp version).
    Uses the MODEL_PATH global defined at the top of the file."""

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found at {MODEL_PATH}. Update the "
            "MODEL_PATH global at the top of this file to point at one "
            "of the folders under huggingface-full/Qwen/, this script "
            "does not download anything automatically."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",  # match whatever dtype the checkpoint was saved in (bf16 for these Qwen files)
        device_map="auto",   # spread across visible GPUs, fall back to CPU if none
    )
    model.eval()  # inference only, disables dropout and similar training-only layers

    return model, tokenizer


def load_sample_documents(num_docs):
    metadata_path = os.path.join(DATASET_DIR, "metadata.json")
    with open(metadata_path, "r") as f:
        publications = json.load(f)

    return random.sample(publications, num_docs)


def load_vocabulary():
    targets_path = os.path.join(DATASET_DIR, "targets.json")
    with open(targets_path, "r") as f:
        targets = json.load(f)
    return [canonicalize(t) for t in targets]


def build_prompt(input_text, vocabulary):
    """Build the prompt. input_text is whatever field INPUT_FIELD points
    to (title or abstract), the wording adapts to name that field so the
    model gets an accurate description of what it is reading.

    Identical wording to the llama.cpp version, minus the trailing
    "/no_think" text flag, since Qwen3.5's chat template controls
    thinking mode through enable_thinking in apply_chat_template
    instead of an inline text tag. Check chat_template.jinja in the
    model folder if you need to confirm this before a full run."""

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
    """Same safety net as the other versions: Qwen can emit a
    <think>...</think> reasoning block. If it's unclosed (cut off by the
    token limit), there is no real answer to extract, return empty rather
    than leaking the raw reasoning text into matching."""

    if "<think>" in text and "</think>" not in text:
        return ""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_model(model, tokenizer, prompt, max_new_tokens=100):
    """Run the model once on a single prompt and return the answer."""

    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template formats the prompt the way the model was trained
    # on (system/user/assistant tags), instead of feeding raw text.
    # enable_thinking=False turns off Qwen3.5's reasoning mode so the
    # answer comes back directly, matching the short, no-reasoning
    # behavior the "/no_think" flag gave us in the llama.cpp version.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():  # inference only, saves memory by not tracking gradients
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,  # short cap, the answer is just a topic name
            do_sample=False,                # deterministic output, same call every time for a given input
        )

    # slice off the input tokens so we only decode the newly generated part
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

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
    # Build a results filename that encodes which checkpoint produced it,
    # e.g. Qwen3.5-9B, so runs from different model sizes never overwrite
    # each other's output.
    model_name = os.path.basename(MODEL_PATH.rstrip("/"))
    results_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"experiment_{model_name}_hf_{INPUT_FIELD}_{RUN_TIMESTAMP}_results.json",
    )

    model, tokenizer = load_model()

    vocabulary = load_vocabulary()
    documents = load_sample_documents(NUM_SAMPLE_DOCS)

    results = []
    for i, doc in enumerate(documents, start=1):
        # Pull whichever field INPUT_FIELD points to (title or abstract) as the model input.
        input_text = doc[INPUT_FIELD]
        ground_truth = [canonicalize(s) for s in doc["subjects"]]

        print(f"[{i}/{len(documents)}] {input_text}")

        prompt = build_prompt(input_text, vocabulary)
        raw_answer = ask_model(model, tokenizer, prompt)

        old_cat, old_topic, new_cat, new_topic = classify_answer(
            raw_answer, vocabulary, ground_truth
        )

        print(f"  raw answer: {raw_answer!r}")
        print(f"  old matching: {old_cat} ({old_topic})")
        print(f"  new matching: {new_cat} ({new_topic})")

        results.append({
            "D3 ID": doc["D3 ID"],
            "model": model_name,
            "input_field": INPUT_FIELD,
            "input_text": input_text,
            "ground_truth_subjects": ground_truth,
            "raw_answer": raw_answer,
            "old_matching": {"category": old_cat, "topic": old_topic},
            "new_matching": {"category": new_cat, "topic": new_topic},
        })

        # Save after every document, not just at the end, so a crash midway
        # through the 2500 queries doesn't lose everything done so far.
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {results_path}")
    print_summary(results)


def print_summary(results):
    for label in ("old_matching", "new_matching"):
        counts = {"success": 0, "misclassified": 0, "hallucination": 0}
        for r in results:
            counts[r[label]["category"]] += 1
        print(f"{label}: {counts}")


if __name__ == "__main__":
    main()