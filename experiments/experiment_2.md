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

## Status

Qwen3.5-0.8B and Qwen3.5-4B done (title + abstract each). Remaining: 2B, 9B, and 27B (pending the quantization decision above). Each runs as `python3 experiment_qwen3.5_hf_1.py --model Qwen3.5-<size> --field <title|abstract>`.
