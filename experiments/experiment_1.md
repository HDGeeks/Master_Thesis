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
- Built `src/experiment_haiku_1.py`: reproduces the paper's exact 3-turn prompt (Figure 2) as a genuine multi-turn conversation (each prompt gets a real reply, replies stay in context for the next prompt, matching how the paper's GPT4All chat session worked), run against the Claude API, model `claude-haiku-4-5-20251001`. Scores every answer both the old way (exact match only) and the new way (matching.py), so the two fixes can be evaluated on the same run


## Part 3: Local inference setup (Qwen3-8B)

Wanted a second, fully local "newer model" for the same quality check, no API cost. Took three attempts to get working inference:

1. **`llama-cpp-python`** (first choice, reuses GGUF, has grammar support for later idea 9): failed to build, this Mac's Xcode Command Line Tools are missing C++ standard headers entirely (`<array>`, `<cstdint>` not found even in a bare `clang++` test), a broken system toolchain, not fixable without a `sudo` CLT reinstall.
2. **Ollama** (prebuilt binary, no compiling): worked, but `ollama ps` revealed it was running 100% on CPU despite this being an Apple M4 with a capable GPU. Homebrew's `ollama` formula bottle doesn't have Metal compiled in (confirmed via `otool -L`, no Metal.framework linkage).
3. **`mlx-lm`** (Apple's own framework, pure pip install, Metal-accelerated, no compilation at all): this is what's actually used. Needed a different model format than GGUF, downloaded `mlx-community/Qwen3-8B-4bit` (4.6GB) into `models/Qwen3-8B-4bit-mlx/` directly via curl after `huggingface_hub`'s built-in `snapshot_download` stalled for an unclear reason.

Built `src/experiment_qwen3-8b_macos_1.py`, mirroring `experiment_haiku_1.py`'s structure, plus `ensure_model_downloaded()` so the script auto-downloads the model on a fresh machine if it's missing, no manual setup step needed.

### Part 3 result: Qwen3-8B local, first version (3-turn prompt, same as Claude run)

Bug found on the first real run: Qwen3's default "thinking" mode produced a `<think>` reasoning block that got cut off by the `max_tokens=500` cap before closing. The original `strip_thinking()` only stripped *closed* `<think>...</think>` pairs, so the entire unclosed reasoning blob became "the answer," got comma-split, and one fragment coincidentally fuzzy-matched a real topic, producing a false "success" that had nothing to do with the model's actual intent. Fixed two ways: disabled thinking mode via `enable_thinking=False` in `apply_chat_template`, and made `strip_thinking()` return an empty string (not the raw blob) when `<think>` appears without a closing tag.

Result after the fix: **40% success, 60% misclassified, 0% hallucination.** Old vs new matching identical (clean output, nothing for the fuzzy fix to catch). Notable pattern: answered "information technology" 8/20 times, wrong every single time, looks like a generic fallback default rather than reasoned answers, similar in spirit to the paper's own "computer science" fallback, just landing on a real vocabulary term instead of an invalid one.

### Part 3 result: Qwen3-8B local, single-prompt version (final)

The 3-turn structure means 3 model calls per document for only 1 scored answer, wasteful at the 2500-document scale this is headed for on a separate machine. Collapsed the 3 turns into a single combined prompt, 1 model call per title instead of 3 (`build_prompt()` + `ask_model()`, replacing `build_prompts()` + `run_chat_session()`).

Result: **55% success, 40% misclassified, 5% hallucination.** First genuine hallucination observed: "Blind Domain Adaptation: An RKHS Approach" answered "machine learning", not one of the 19 topics, correctly caught as hallucination by both matching versions. Old vs new matching still identical. The generic-fallback pattern persisted but shifted label, from "information technology" to "information retrieval" (5/20 uses, right once).

### Part 3 result: Qwen3-8B, full 2500 documents, titles (Linux/Ollama, single-prompt)

Ran on the separate server-grade machine (no GPU, CPU-only via Ollama, `src/experiment_qwen3-8b_linux_1.py`), same single-prompt approach as the Mac version, once, on all 2500 documents, titles as document representation (same as the paper's first experiment).

Result:

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Old matching | 63.3% | 29.5% | 7.2% |
| New matching | 66.6% | 30.3% | 3.1% |

Better than the paper's Llama 3 8B (50/25/25) across every category: higher success, similar misclassification, less than a third the hallucination rate.

The matching fix mattered a lot at this scale, not marginal like the 20-doc runs suggested: **103 of 2500 answers (4.1%) changed category**, and all 103 have the identical raw answer `'computer vision'`, a systematic habit of this model, not a one-off. Under old matching all 103 were hallucinations (not one of the 19 topics verbatim). Under new matching: 83 became success, 20 became misclassified, none stayed unresolved. This is the exact near-miss case the paper documented ("computer vision" vs "computer imaging and vision") showing up naturally at scale and being fully recovered.

Built `src/analyze_results.py` (generic, takes any results file as an argument) to compute these stats and pull a random sample of cases for manual reading, reusable for future experiment result files instead of writing a one-off stats script per run.

### Part 3 result: Qwen3-8B, full 2500 documents, abstracts (Linux/Ollama, single-prompt)

Same setup as the titles run, `src/experiment_qwen3-8b_linux_2.py`, `INPUT_FIELD = "abstract"` instead of title.

Result:

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Old matching | 61.8% | 29.5% | 8.6% |
| New matching | 66.3% | 30.5% | 3.2% |

Same "computer vision" pattern as the titles run: 136 of 2500 answers (5.4%) were the literal string `'computer vision'`, all hallucinations under old matching, recovered to 111 success + 25 misclassified under new matching.

Unlike the paper, abstracts did not produce more hallucinations than titles here (8.6% vs 7.2% old matching, roughly the same; 3.2% vs 3.1% new matching, essentially identical). The paper saw a jump from 25% to 33% hallucination going title to abstract. Success and misclassified rates are also both roughly flat between title and abstract for this model, so Qwen3-8B does not reproduce the paper's title-vs-abstract effect at all.

### Part 4: Full-scale Claude and a second Qwen3-8B run ("cpp" variant)

Two more full 2500-document runs, both title and abstract, from `results/experiment_qwen3-8b_cpp_title_*`, `experiment_qwen3-8b_cpp_abstract_*`, `experiment_1_claude_title_*`, and `experiment_1_claude_abstract_*`. The scripts behind these ("cpp" Qwen3-8B variant, and the full-scale Claude run) weren't built in this session, so implementation details aren't documented here, only the results, read via `src/analyze_results.py`. The abstract run's script attempted prompt caching, but every prompt fell under Anthropic's minimum cacheable size (4096 tokens), so caching silently never activated, this only affected cost/latency, not the answers themselves, and the script has since been retired (didn't test what it was meant to).

Results (new matching):

| Run | Success | Misclassified | Hallucination |
|---|---|---|---|
| Qwen3-8B (cpp), titles | 67.5% | 29.8% | 2.8% |
| Qwen3-8B (cpp), abstracts | 66.5% | 30.3% | 3.2% |
| Claude, titles | 70.9% | 27.6% | 1.5% |
| Claude, abstracts | 74.0% | 23.3% | 2.7% |

Two findings:

1. **Hallucination collapses for both newer models, regardless of title or abstract.** Paper: 25% (titles) / 33% (abstracts). Both Qwen3-8B and Claude stay under 4% in every single run. This looks like a model-generation effect, not a title/abstract effect, newer/better instruction-following models just don't produce as much unparseable garbage in the first place.
2. **Title vs abstract splits in opposite directions per model.** Qwen3-8B is consistent across both this "cpp" run and the earlier `linux_2` run: titles slightly outperform abstracts (67.5% vs 66.5%, matching the earlier 66.6% vs 66.3%), reproducing the paper's own counter-intuitive finding that more context doesn't help this task. Claude does the opposite: abstracts clearly outperform titles (74.0% vs 70.9%). So "does more context help" depends on the model, not a fixed property of the task the way the paper (with a single model) implied.

### Results comparison

| Run | Success | Misclassified | Hallucination |
|---|---|---|---|
| Paper (Llama 3 8B, titles, 2500 docs, 5 runs) | 50% | 25% | 25% |
| Paper (Llama 3 8B, abstracts, 2500 docs, 5 runs) | 50% | 17% | 33% |
| Qwen3-8B local, 3-turn, 20 docs | 40% | 60% | 0% |
| Qwen3-8B local, single-prompt, 20 docs | 55% | 40% | 5% |
| Qwen3-8B local, single-prompt, 2500 docs, titles | 66.6% | 30.3% | 3.1% |
| Qwen3-8B local, single-prompt, 2500 docs, abstracts | 66.3% | 30.5% | 3.2% |
| Qwen3-8B (cpp), 2500 docs, titles | 67.5% | 29.8% | 2.8% |
| Qwen3-8B (cpp), 2500 docs, abstracts | 66.5% | 30.3% | 3.2% |
| Claude, 2500 docs, titles | 70.9% | 27.6% | 1.5% |
| Claude, 2500 docs, abstracts | 74.0% | 23.3% | 2.7% |

### Status

Full 2500-document runs complete for two models (Qwen3-8B, Claude), both title and abstract, both old and new matching. Part 2 (newer model quality check) is done. Follow-up question (does model size matter within a newer generation) spun off into experiments/experiment_2.md. Next: decide the next branch per meeting 1's plan (cost optimization + multi-label + hierarchy, vs few-shot/fine-tuning), informed by these numbers.
