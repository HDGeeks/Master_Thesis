# Summary: "Investigating Zero-shot Topic Labeling of Scientific Papers Using LLMs" (BTW2025-122)

**Authors:** Bruchertseifer, Neises, Hinzmann, Schenkel, Schöch (Trier University / Dagstuhl LZI)

---

## Core Idea
Zero-shot classification of scientific papers into topics from a controlled vocabulary using a local LLM (Llama 3 8B via GPT4All), with no fine-tuning or training examples.

## Setup
- **Dataset:** 2500 papers sampled from DBLP Discovery Dataset (D3 v2.1), pre-labeled by the CSO classifier
- **Vocabulary:** 19 top-level Computer Science Ontology (CSO) topics (e.g., "artificial intelligence", "data mining", ...)
- **Experiments:** Two runs — one using paper **title**, one using paper **abstract** as input
- **Prompting:** 3-turn prompt: (1) define target list, (2) add topics to list, (3) ask for one matching topic
- **Matching:** Exact string match after lowercasing and trimming whitespace

## Results

| | Title | Abstract |
|---|---|---|
| Success | ~50% | ~50% |
| Misclassified | ~25% | ~17% |
| Hallucination | ~25% | ~33% |

- Random baseline would achieve ~24% success (19 topics, avg 4.5 labels/doc)
- Results were **stable across 5 runs**
- Runtime: ~2h52m (titles) / ~3h30m (abstracts) on RTX 4090

## Key Surprising Findings
1. **Abstracts gave no improvement in success rate** over titles (both ~50%)
2. **Abstracts produced significantly more hallucinations** (33% vs 25%) — counter-intuitive since abstracts carry more semantic content
3. The shift was from *misclassified* → *hallucination*, not to *success* as expected

## Why Hallucinations Happen (observed causes)
- LLM returns `"computer science"` (parent of all topics, not in the controlled vocab)
- LLM returns `"computer vision"` instead of `"computer imaging and vision"` (close but not exact)
- LLM wraps answer in quotes: `"'software engineering'"` — breaks exact matching

## Shortcomings / Limitations
1. **Exact string matching is fragile** — minor formatting breaks valid answers
2. **Single-label output only** — docs have avg 4.5 ground truth labels; the LLM is asked for just 1
3. **Only 19 coarse-grained top-level topics** — no depth/hierarchy tested
4. **One LLM only** (Llama 3 8B) — no model comparison
5. **No few-shot examples** — purely zero-shot
6. **No self-correction loop** — detected hallucinations are just discarded
7. **Context/prompt length limits** not fully understood (only visible in debug logs)
8. **Evaluation metric is binary** — misclassified is penalized even when the LLM's answer is plausible
9. **Ground truth is noisy** — D3 labels come from the CSO classifier (also automatic), not human annotation
10. **Humanities dataset (STTCL) failed badly (~10%)** — vocabulary too large/raw, data needs preprocessing
11. **No ablation of prompt design** — the 3-turn prompt structure is untested against alternatives

---

## Improvement Ideas (brainstorm — to be vetted)

### Matching / Output Parsing
- Fuzzy string matching (e.g., edit distance, token overlap) to catch near-misses like "computer vision" → "computer imaging and vision"
- Structured output / constrained decoding (e.g., JSON schema, grammar-constrained generation) to force the LLM to only output valid vocabulary terms
- Post-hoc normalization: strip punctuation, quotes, extra whitespace before matching

### Prompting Strategy
- Few-shot prompting: add 1-3 examples of correct title→topic mappings in the prompt
- Chain-of-thought prompting: ask the model to reason before giving the answer, then extract the label from the reasoning
- Self-correction loop: if hallucination is detected, send a follow-up prompt pointing to the error and asking for a valid topic from the list
- Ask for top-3 topics instead of 1, then match against the multi-label ground truth (better recall)
- Enumerate topics as numbered list in prompt to reduce hallucination of parent categories

### Model & Infrastructure
- Test multiple models (Mistral, Phi-3, Gemma, etc.) and compare; larger quantizations vs smaller models
- Use instruction-tuned models with stricter system prompts that enforce vocabulary constraints
- Explore embedding-based retrieval: embed vocabulary labels and paper text, use cosine similarity to pick labels (no generation needed)

### Evaluation
- Treat it as multi-label classification: evaluate Precision, Recall, F1 against the full subjects list (not just top-1 match)
- Separate evaluation on "easy" vs "hard" papers (papers with 1 label vs 12 labels)
- Human annotation gold standard for a subset to measure how noisy the D3 labels actually are
- Compare against a fine-tuned BERT baseline to put the zero-shot numbers in context

### Hierarchy & Scalability
- Exploit the CSO hierarchy: classify at multiple levels (coarse → fine), using the coarse prediction to narrow down the fine-grained label set
- Test scalability: how does performance degrade as vocabulary grows from 19 → 100 → 1000 topics?
- Dynamic vocabulary truncation: only present the LLM with a shortlisted subset of labels (pre-filtered by embedding similarity) to reduce confusion

### Data & Domain
- Apply to cross-domain papers (e.g., NLP papers that span AI + linguistics)
- Test on humanities datasets with better preprocessing of the vocabulary (deduplicate, normalize DCN labels)
- Augment titles with keywords extracted by a lightweight model before LLM classification
