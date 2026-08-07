# Experiment 2: Model size vs accuracy (Qwen3.5 family)

## Goal

Experiment 1 found that newer models (Qwen3-8B, Claude) dramatically beat the paper's Llama 3 8B baseline on hallucination and success rate. This experiment asks a follow-up question: within a single newer-generation model family, does size itself matter? Ties back to the original brainstorm idea 8 (docs/brainstorm.md): "test multiple models and compare; larger quantizations vs smaller models."

Uses the Qwen3.5 family (0.8B, 2B, 4B, 9B, 27B), same task, same prompt, same matching logic, same 2500-document dataset as Experiment 1, so results are directly comparable across both experiments.

## Setup

- Runs on a Helmholtz cluster node: NVIDIA RTX 4090 (24GB VRAM), AMD Ryzen 9 7900X (12 cores / 24 threads), 125GB RAM
- Model checkpoints stored locally under `/localdata/dagstuhl/ai_models/huggingface-full/Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`
- Built `src/experiment_qwen3.5_hf_1.py`: Hugging Face `transformers` backend (`AutoModelForCausalLM`), same single-prompt approach as the other Experiment 1 scripts. Model size and input field are CLI args (`--model`, `--field`), not hardcoded globals, so each size runs as its own process, a crash or OOM on the largest checkpoint doesn't take the others down with it
- Built `src/metrics.py`: reusable runtime/throughput/memory instrumentation (model load time, total runtime, avg/median per-call latency, tokens/sec, truncation rate, peak RAM/GPU memory). Saved to `results/metrics/`, separate from the results files, linked by shared filename stem plus an explicit `results_file` field inside the metrics JSON (robust to either file being renamed later)
- Bug found and fixed: `tokenizer.apply_chat_template(..., return_tensors="pt")` returned a `BatchEncoding`, not a bare tensor, on this `transformers` version. Passing it straight into `model.generate()` caused `AttributeError` deep inside `generate()`'s internals (`inputs_tensor.shape[0]` on a dict-like object). Fixed with `return_dict=True` and explicit unpacking (`model.generate(**model_inputs, ...)`)

### Known constraint: 27B likely won't fit

Rough bf16 memory math (2 bytes/parameter): 27B needs ~54GB just for weights, more than double the 24GB on this single GPU. `device_map="auto"` can't span multiple GPUs here (only one is available), so it will either OOM or silently CPU-offload (much slower, not a fair comparison point). Decision pending: load 27B (and possibly 9B) in 4-bit via `bitsandbytes`, or accept and document the CPU-offload slowdown.

## Results

### Qwen3.5-4B

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Titles (new matching) | 75.4% | 24.0% | 0.6% |
| Abstracts (new matching) | 76.7% | 22.3% | 1.0% |

Metrics: title run 657.7s (~11 min) wall-clock, abstract run 576.8s (~9.6 min), ~30 tokens/sec both, near-zero truncation (0/2500 titles, 2/2500 abstracts), peak GPU memory ~8.2-8.6GB (matches the ~8GB bf16 estimate almost exactly, comfortable headroom on the 24GB card).

**Headline finding**: Qwen3.5-4B, less than half the parameter count of Qwen3-8B, beats both Qwen3-8B (67.5%/66.5%) and Claude (70.9%/74.0%) on success rate. Newer generation matters more than raw size, at least at this one data point. Also matches Claude's title-vs-abstract direction (abstracts slightly better) rather than Qwen3-8B's (titles slightly better).

### Qwen3.5-0.8B

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Titles (new matching) | 52.1% | 47.3% | 0.6% |
| Abstracts (new matching) | 53.7% | 43.9% | 2.4% |

Metrics: title run 547.6s (~9.1 min), abstract run 446.9s (~7.4 min), ~40 tokens/sec both (faster than 4B's ~30, expected for a smaller model), near-zero truncation, peak GPU memory ~1.5-1.7GB (vs ~8.2-8.6GB for 4B, scales with size as expected).

**Finding**: a real drop from 4B, about 23 points lower on success. So size does matter within the family, going from 4B to 0.8B costs a lot, even though 4B itself already beat the larger, older-gen Qwen3-8B. The interesting nuance is where the drop shows up: hallucination stays low even at 0.8B (0.6%/2.4%, close to 4B's 0.6%/1.0%), what collapses is judgment, not format-following, misclassified jumps to 47.3%/43.9% (vs 24.0%/22.3% at 4B). Reads as: staying inside the controlled vocabulary is a low bar even a small model clears, picking the *correct* topic is what actually needs the extra capacity. Also notable: 0.8B (52.1% titles) still barely edges out the original paper's Llama 3 8B (50%), a model 10x its size from an older generation.

### Qwen3.5-9B

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Titles (new matching) | 78.6% | 20.8% | 0.6% |
| Abstracts (new matching) | 80.8% | 18.7% | 0.5% |

Metrics: title run 656.6s (~10.9 min, almost identical to 4B's 657.7s despite being a bigger model, still memory-bandwidth-bound rather than compute-bound at this size on the 4090), abstract run 931.2s (~15.5 min, clearly slower than title here, unlike 4B where abstract was oddly faster). ~28.2/26.5 tokens/sec (title/abstract), zero truncation. Peak GPU memory 17.2-17.6GB, matches the earlier "~18GB, tight but should fit" estimate almost exactly, ~6-7GB headroom left on the 24GB card.

**Best result of the project so far**, and the scaling picture across the three Qwen3.5 sizes run to date is clean and monotonic:

| Size | Titles | Abstracts |
|---|---|---|
| 0.8B | 52.1% | 53.7% |
| 4B | 75.4% | 76.7% |
| 9B | 78.6% | 80.8% |

Bigger keeps winning, but with clear diminishing returns: 0.8B→4B is a +23 point jump, 4B→9B is only +3-4 points. Most of the achievable gain from scale is already captured by 4B, worth keeping in mind when weighing whether 27B is worth the trouble (VRAM risk, quantization, slower runtime) for a comparatively small further gain.

### Qwen3.5-27B, titles, bf16 (2 runs, CPU-offloaded)

27B does not fit in bf16 on the 24GB GPU (~54GB needed), so `device_map="auto"` split it across GPU and CPU RAM automatically instead of failing. Ran title twice before switching to 4-bit (see below):

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Run 1 (new matching) | 75.4% | 24.2% | 0.4% |
| Run 2 (new matching) | 75.4% | 24.2% | 0.4% |

Identical counts both times, expected, `do_sample=False` is fully deterministic, this confirms pipeline reproducibility rather than giving two independent accuracy data points.

**Runtime confirms the CPU-offload penalty is severe, not marginal**: ~25,263s and ~24,792s (~7 hours per run), 0.38/0.37 tokens/sec, versus 9B's ~28 tokens/sec, a ~73x throughput collapse and ~30x wall-clock slowdown. Peak GPU memory ~21GB (most of the model did fit on GPU), peak RAM ~53GB (the rest lived in system RAM, shuffled in per layer during each forward pass).

**Accuracy finding, genuinely surprising**: 27B (75.4%) does not beat 9B (78.6%), and lands exactly on 4B's number (75.4%). The clean monotonic scaling seen through 0.8B→4B→9B breaks here. Bigger is not automatically better past 9B, at least for this checkpoint on titles. Worth treating as a real result, not noise, since both runs landed on the identical number.

Switched to `--quantize 4bit` (bitsandbytes) for further 27B runs, all layers fit on GPU at 4-bit (~13.5GB), avoiding the CPU-offload penalty entirely. These 2 bf16 runs are kept as a separate quantization-effect comparison point once the 4-bit numbers are in, rather than folded into the same "3 runs for consistency" set as the other sizes.

### Qwen3.5-27B, 4-bit (title x1, abstract x3, all GPU-only)

| | Success | Misclassified | Hallucination |
|---|---|---|---|
| Titles (new matching) | 76.0% | 24.0% | 0.04% |
| Abstracts (new matching, 3 runs, identical) | 77.0% | 22.9% | 0.1% |

All 3 abstract runs landed on the exact same counts (1926/572/2 every time), same determinism as everywhere else, confirms reproducibility again. Timing tightly consistent across the three: 1316.5s / 1258.8s / 1262.8s (~21-22 min), ~8.0 tokens/sec, ~18.9GB peak GPU memory each time.

**Speed**: title run 1003s (~16.7 min, 9.6 tokens/sec), a ~25x speedup over the bf16/CPU-offload version (~7 hours, 0.38 tokens/sec). Still about 1.5x slower than 9B (~11 min, ~28 tokens/sec) despite 4-bit needing less raw memory bandwidth per token than 9B's bf16, the gap is dequantization compute overhead (unpacking 4-bit weights before each matmul), not bandwidth. A useful correction to the earlier "should be comparable to 9B" guess, real number came in close to the original, more conservative estimate instead.

**Accuracy holds up well under quantization, but not identically per document**: aggregate success barely moved (75.4% bf16 → 76.0% 4-bit on titles), but a direct per-document diff between the bf16 and 4-bit title runs showed only 87.8% identical raw answers, 12.2% (304/2500) of documents got a genuinely different answer. Breakdown of those 304: 99 success→success (different but both valid), 73 misclassified→success, 62 misclassified→misclassified, 59 success→misclassified, 7 hallucination→misclassified, 3 hallucination→success, 1 success→hallucination. Net +23 (83 improved vs 60 worsened), consistent with the aggregate +16 success delta once the wash cases are netted out. Lesson: aggregate accuracy can look stable while hiding real answer-level churn, worth flagging as a methodological caveat rather than assuming a stable top-line number means nothing changed underneath.

**The headline finding is now well-replicated, not a single-run fluke**: 27B (76.0%/77.0%) still does not beat 9B (78.6%/80.8%) at either field, and this holds at both precisions tested (bf16 title 75.4%, 4-bit title 76.0%, both below 9B's 78.6%). Full Qwen3.5 scaling picture:

| Size | Titles | Abstracts |
|---|---|---|
| 0.8B | 52.1% | 53.7% |
| 4B | 75.4% | 76.7% |
| 9B | 78.6% | 80.8% |
| 27B | 76.0% | 77.0% |

**9B is the accuracy sweet spot** in this family for this task, not the largest checkpoint. Scaling helps sharply from 0.8B to 4B, keeps helping a little through 9B, then reverses at 27B. Worth leading with this framing in the thesis over a flat "bigger is better" narrative, since the data directly contradicts that story at the top end.

### Full comparison across both experiments

| Run | Success | Misclassified | Hallucination |
|---|---|---|---|
| Paper (Llama 3 8B), titles | 50% | 25% | 25% |
| Paper (Llama 3 8B), abstracts | 50% | 17% | 33% |
| Qwen3.5-0.8B, titles | 52.1% | 47.3% | 0.6% |
| Qwen3.5-0.8B, abstracts | 53.7% | 43.9% | 2.4% |
| Qwen3-8B, titles | 67.5% | 29.8% | 2.8% |
| Qwen3-8B, abstracts | 66.5% | 30.3% | 3.2% |
| Claude, titles | 70.9% | 27.6% | 1.5% |
| Claude, abstracts | 74.0% | 23.3% | 2.7% |
| Qwen3.5-4B, titles | 75.4% | 24.0% | 0.6% |
| Qwen3.5-4B, abstracts | 76.7% | 22.3% | 1.0% |
| Qwen3.5-9B, titles | 78.6% | 20.8% | 0.6% |
| Qwen3.5-9B, abstracts | 80.8% | 18.7% | 0.5% |
| Qwen3.5-27B, titles, bf16/CPU-offload (x2, identical) | 75.4% | 24.2% | 0.4% |
| Qwen3.5-27B, titles, 4-bit | 76.0% | 24.0% | 0.04% |
| Qwen3.5-27B, abstracts, 4-bit (x3, identical) | 77.0% | 22.9% | 0.1% |

## Quantization sensitivity scales inversely with model size

`run.sh`'s consistency-check pass accidentally applied `--quantize 4bit` to every model, not just 27B (a script bug, `--quantize 4bit` sat outside the 27B-specific block). Since generation is fully deterministic (`do_sample=False`, every repeat at fixed settings has come back bit-identical across the whole project so far), the intended "3 runs to check variance" wasn't actually testable, there is no run-to-run randomness in this pipeline. What the accident produced instead was more useful: 1 native-precision run plus 2 confirmed-stable 4-bit runs per model, a real bf16-vs-4bit comparison across the whole size range.

| Size | Field | bf16 | 4-bit | Δ |
|---|---|---|---|---|
| 0.8B | titles | 52.1% | 39.2% | **-12.9 pts** |
| 0.8B | abstracts | 53.7% | 27.6% | **-26.1 pts** |
| 4B | titles | 75.4% | 74.6% | -0.8 pts |
| 4B | abstracts | 76.7% | 74.5% | -2.2 pts |
| 9B | titles | 78.6% | 80.0% | +1.4 pts |
| 9B | abstracts | 80.8% | 81.5% | +0.7 pts |
| 27B | titles | 75.4% (bf16, CPU-offloaded) | 76.0% | +0.6 pts |
| 27B | abstracts | n/a (only run at 4-bit) | 77.0% | n/a |

**Clean, monotonic pattern**: quantization sensitivity drops sharply as model size increases. 0.8B is fragile, losing 13-26 points, over half its edge over random guessing gone on abstracts. 4B already mostly absorbs it (under 2.5 points lost). 9B and 27B are effectively indifferent to 4-bit, gains and losses both stay under 1.5 points, likely noise-level. Reads as: larger models have more redundant capacity to tolerate the precision loss, smaller ones don't have slack to spare.

Secondary supporting data point: **Qwen3-8B (non-3.5, bitsandbytes 4-bit)** scored 64.0%/63.1% (titles/abstracts) with notably higher hallucination (7.8%/7.0%), meaningfully worse than the same base model's earlier GGUF Q4_K_M run via `experiment_qwen3-8b_cpp.py` (67.5%/66.5%, 2.8%/3.2% hallucination). Different quantization method, same underlying checkpoint, real difference in outcome, consistent with older/weaker-trained checkpoints (Qwen3-8B is an older generation than the Qwen3.5 line) being less robust to quantization noise in general, not just a pure model-size effect.

## Per-model raw run tables

Every individual run, new matching. All duplicate runs at the same field+precision came back bit-identical (deterministic, `do_sample=False`), confirmed by direct comparison, not assumed.

### Qwen3.5-0.8B

| Run | Field | Precision | Success | Misclassified | Hallucination |
|---|---|---|---|---|---|
| 1 | title | bf16 | 52.1% | 47.3% | 0.6% |
| 2 | title | 4-bit | 39.2% | 58.8% | 2.0% |
| 3 | title | 4-bit | 39.2% | 58.8% | 2.0% |
| 1 | abstract | bf16 | 53.7% | 43.9% | 2.4% |
| 2 | abstract | 4-bit | 27.6% | 69.6% | 2.8% |
| 3 | abstract | 4-bit | 27.6% | 69.6% | 2.8% |

### Qwen3.5-4B

| Run | Field | Precision | Success | Misclassified | Hallucination |
|---|---|---|---|---|---|
| 1 | title | bf16 | 75.4% | 24.0% | 0.6% |
| 2 | title | 4-bit | 74.6% | 24.8% | 0.5% |
| 3 | title | 4-bit | 74.6% | 24.8% | 0.5% |
| 1 | abstract | bf16 | 76.7% | 22.3% | 1.0% |
| 2 | abstract | 4-bit | 74.5% | 24.3% | 1.2% |
| 3 | abstract | 4-bit | 74.5% | 24.3% | 1.2% |

### Qwen3.5-9B

| Run | Field | Precision | Success | Misclassified | Hallucination |
|---|---|---|---|---|---|
| 1 | title | bf16 | 78.6% | 20.8% | 0.6% |
| 2 | title | 4-bit | 80.0% | 19.7% | 0.2% |
| 3 | title | 4-bit | 80.0% | 19.7% | 0.2% |
| 1 | abstract | bf16 | 80.8% | 18.7% | 0.5% |
| 2 | abstract | 4-bit | 81.5% | 17.6% | 0.9% |
| 3 | abstract | 4-bit | 81.5% | 17.6% | 0.9% |

### Qwen3.5-27B

| Run | Field | Precision | Success | Misclassified | Hallucination |
|---|---|---|---|---|---|
| 1 | title | bf16 (CPU-offload, ~7hr) | 75.4% | 24.2% | 0.4% |
| 2 | title | bf16 (CPU-offload, ~7hr) | 75.4% | 24.2% | 0.4% |
| 3 | title | 4-bit | 76.0% | 24.0% | 0.04% |
| 1 | abstract | 4-bit | 77.0% | 22.9% | 0.1% |
| 2 | abstract | 4-bit | 77.0% | 22.9% | 0.1% |
| 3 | abstract | 4-bit | 77.0% | 22.9% | 0.1% |

### Qwen3-8B (non-3.5)

| Run | Field | Precision | Success | Misclassified | Hallucination |
|---|---|---|---|---|---|
| 1 | title | GGUF Q4_K_M (llama.cpp) | 67.5% | 29.8% | 2.8% |
| 2 | title | bitsandbytes 4-bit | 64.0% | 28.2% | 7.8% |
| 3 | title | bitsandbytes 4-bit | 64.0% | 28.2% | 7.8% |
| 1 | abstract | GGUF Q4_K_M (llama.cpp) | 66.5% | 30.3% | 3.2% |
| 2 | abstract | bitsandbytes 4-bit | 63.1% | 29.8% | 7.0% |
| 3 | abstract | bitsandbytes 4-bit | 63.1% | 29.8% | 7.0% |

## Status

All planned sizes done: Qwen3.5-0.8B, 4B, 9B, 27B (title + abstract, both bf16 and 4-bit where applicable), plus Qwen3-8B (non-3.5) at 4-bit and GGUF Q4_K_M. Remaining: Qwen3.5-2B, if pursued further. Each runs as `python3 experiment_qwen3.5_hf_1.py --model <name> --field <title|abstract> [--quantize 4bit]`.
