# Experiment 1: Matching fixes + newer model quality check

## Goal

Two things from docs/brainstorm.md idea 1, plus the first step of the meeting 1 concrete plan (docs/meeting_1_result.md): check whether a newer model gives usable output on the paper's own task, and whether fixing the matching step alone recovers some of the paper's hallucination cases.

## Part 1: Matching fixes

The paper's matching (`reference-repo/modules/data_processing.py`) only lowercases and strips whitespace, then requires an exact string match against the 19 topics. Two of the paper's own documented hallucination cases would survive that:
- The model answering `"'software engineering'"` (with literal quote characters)
- The model answering "computer vision" instead of the actual vocabulary term "computer imaging and vision"

### Design decision: why not plain fuzzy string matching

First attempt was raw character-similarity matching (Python's `difflib`). Checked it against the real 19-topic vocabulary before building anything, and it fails on the exact case it's supposed to fix:

```
"computer vision" vs "computer aided design"       -> similarity 0.722
"computer vision" vs "computer imaging and vision" -> similarity 0.714
```

A naive top-1 similarity match would pick the wrong topic here. The many "computer ___" topics in this vocabulary collide under plain character similarity.

Second problem: "computer science" is the paper's single most common hallucination, and it's the ontology's parent category, not one of the 19 valid topics. It still scores 0.68-0.73 similar to several real topics ("theoretical computer science", "computer security", "computer systems"). Auto-resolving it via fuzzy match would fabricate an answer the model never actually gave.

### What was built (`src/matching.py`)

1. Canonicalize: lowercase, strip whitespace, strip quote characters and trailing punctuation
2. Exact match against the 19 topics (same as the paper, just with better cleanup first)
3. If no exact match: word-containment fuzzy match instead of character similarity. Accept a match only if every word in the candidate appears in exactly one target's word set, e.g. {computer, vision} is a subset of "computer imaging and vision"'s words {computer, imaging, and, vision}, but not a subset of "computer aided design"'s words {computer, aided, design}
4. "computer science" is explicitly excluded from step 3, it always stays a hallucination

Verified against the paper's own examples:

| Input | Old (exact match only) | New (matching.py) |
|---|---|---|
| `"'software engineering'"` | hallucination | software engineering |
| `computer vision` | hallucination | computer imaging and vision |
| `computer science` | hallucination | hallucination (correctly unresolved) |
| `quantum computing` (genuine out-of-vocab) | hallucination | hallucination (correctly unresolved) |

## Part 2: Newer model quality check

Meeting 1's concrete plan starts with manually checking response quality from newer models before deciding the next branch (cost optimization + multi-label + hierarchy, vs few-shot/fine-tuning).

### Setup

- Cloned the paper's official reference-repo (`https://github.com/paNeises/paper-2025-bigds`) for the exact prompt wording, the matching code being improved on, and the reproducible list of the 2500 D3 paper IDs used in the paper
- Downloaded and filtered the full D3 papers dataset down to those exact 2500 documents (`reference-repo/data/assets_example/metadata.json`, `targets.json`), so ground truth CSO subjects are real, not guessed
- Built `src/experiment_1.py`: reproduces the paper's exact 3-turn prompt (Figure 2) as a genuine multi-turn conversation (each prompt gets a real reply, replies stay in context for the next prompt, matching how the paper's GPT4All chat session worked), run against the Claude API, model `claude-haiku-4-5-20251001`. Scores every answer both the old way (exact match only) and the new way (matching.py), so the two fixes can be evaluated on the same run

### Status

Not run yet. Running it as a script needs a separate Anthropic API key (console.anthropic.com, billing separate from the claude.ai chat subscription). Decision pending: pay for a small amount of API credit, or run a handful of documents manually through chat instead (same prompts, no script, no cost, not batchable).
