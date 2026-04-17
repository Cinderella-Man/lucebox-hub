<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>The first GGUF port of DFlash speculative decoding.</strong><br/>
  Qwen3.5-27B at 135 tok/s on a single RTX 3090. 128K context on 24 GB.<br/>
  3.5x faster than chain speculative decoding, 2.9x faster than SGLang AWQ.<br/><br/>
  <a href="https://lucebox.com/blog/dflash">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                   AR (tok/s)   DFlash (tok/s)   Speedup
HumanEval              37.4         130.2          3.48x
Math500                37.4         111.2          2.97x
GSM8K                  37.6          97.0          2.58x
```

> Consumer GPUs can run 27B models at chat-grade speed without multi-GPU, without batching, without quantization compromises. The bottleneck was never hardware. It was the decoding algorithm.

## Why this exists

Single-user local inference on consumer hardware hits a wall at 27B parameters. Autoregressive decode is memory-bandwidth-bound: every token reads the full model weights from VRAM. On a 24 GB RTX 3090 with Q4_K_M weights, that ceiling is ~38 tok/s no matter what framework you use.

Speculative decoding breaks that ceiling by proposing multiple tokens per step with a tiny draft model, then verifying them in a single target forward pass. The [DFlash paper (arxiv:2502.20762)](https://arxiv.org/abs/2502.20762) introduced **block-diffusion speculative decoding**: the draft is a 5-layer non-causal diffusion model conditioned on captured target hidden states. It accepts ~8 tokens per step vs ~3 for chain EAGLE, and comes with [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) as the official draft.

The catch: the reference implementation targets BF16 on H100. There's no working GGUF path, no consumer-GPU implementation, and no port of the [DDTree](https://arxiv.org/abs/2604.12989) tree-structured verify that extracts the last 30% of the speedup.

This repo is that port. Hand-written C++/CUDA on top of ggml, hardcoded for one model pair, designed to fit Qwen3.5-27B Q4_K_M target + BF16 draft into 24 GB and decode at 135 tok/s.

## Results

### LLM benchmarks (Qwen3.5-27B Q4_K_M, concurrency=1, n_gen=256, 10 prompts/dataset)

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 37.4     | **130.2**    | 8.31 | **3.48×** |
| Math500   | 37.4     | **111.2**    | 7.04 | **2.97×** |
| GSM8K     | 37.6     | **97.0**     | 6.14 | **2.58×** |

AR = Autoregressive (`test_generate`). DFlash = DDTree budget 22 + fast rollback (`test_dflash`).
AL = Acceptance Length, average committed tokens per draft/verify step.
Reproduce with `python3 scripts/bench_llm.py` (loads HF datasets).

### 128K context on 24 GB

Q4_0 KV cache + sliding `target_feat` ring (4096 slots) keep the full 131072-token context inside 24 GB VRAM. ~3% AL hit vs F16 KV, 8x memory saving.

| Prompt length | Prefill time | Decode tok/s |
|:-------------:|:------------:|:------------:|
| 520 (HE)      | 0.06 s       | 135          |
| 13K           | 15 s         | 99           |
| 32K           | 106 s        | 35           |
| 128K          | ~10 min      | ~15-20 (est) |

Set `DFLASH27B_KV_Q4=1` to enable. See [RESULTS.md](RESULTS.md) for the full sweep.

## How it works

**Block-diffusion draft.** Each step, the draft sees `[last_target_token, MASK×15]` plus the last 5 captured target hidden states. It denoises the masks in a single forward pass, producing 16 candidate tokens conditioned on real target features. This is structurally stronger than chain EAGLE because every draft position conditions on the same captured context, not on its own noisy predictions.

**DDTree tree verify.** Instead of a single chain of 16 candidates, DDTree builds a best-first tree of up to 22 nodes spanning the top-K branches at each position. One target forward verifies the whole tree via a custom causal mask derived from parent pointers. Sweet spot is budget=22 where draft accuracy plateaus.

**Per-step rollback.** The target's recurrent state (SSM intermediate, conv state, KV cache) is checkpoint-snapshotted before verify and restored after accept. We wrote three custom CUDA kernels to keep rollback free:

| Kernel | Purpose |
|--------|---------|
| `ggml_gated_delta_net_tree_persist` | Direct-writes SSM intermediates into a persistent buffer, skipping a 9 ms `ggml_cpy` per step |
| `ggml_ssm_conv_tree` | Tree-aware conv state gather: each sibling node reads its K-1 conv window along the DDTree parent chain, not DFS order |
| Sliding `target_feat` ring | 4096-slot ring buffer via `(pos % cap)` write/read, enables 128K without holding 6.6 GB of captured features |

**Single-kernel decode path.** Prefill and decode share the same graph builder; only the `n_tokens` dimension and ubatch size differ. No separate codegen paths for chain vs tree verify: both use the `DDTree` structure, chain mode is just `budget=n_spec+1` with no branching.

## Architecture (important)

Qwen3.5-27B is **not** a dense transformer. llama.cpp calls this arch `qwen35`:

- 64 layers total. Every 4th is full softmax attention, the rest are **Gated DeltaNet** (linear attention with learned recurrence).
- M-RoPE with dimension sections `[11, 11, 10, 0]` instead of plain rotary.
- 32 Q heads, 8 KV heads, head dim 128, V head dim 128.
- SSM state cache alongside the KV cache.

The DeltaNet primitive is already a first-class ggml op (`ggml_gated_delta_net`, CUDA kernel at `deps/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`). We port only the graph-construction glue around it plus our three tree-aware kernels, ~2000 lines total.

## Why no llama.cpp, vLLM, or SGLang?

- **llama.cpp** has Qwen3.5-27B via GGUF but no DFlash integration path. Chain EAGLE-style spec is not enough; block diffusion + DDTree requires a custom decode loop that bypasses `llama_decode`.
- **vLLM / SGLang** can run the draft (published FP8 / INT4 variants exist) but require multi-GPU for BF16 target, and their GGUF path for Qwen3.5-27B is broken as of 2026-04. SGLang's AWQ baseline on one 3090 is 47 tok/s — this repo is **2.9x faster** on the same hardware.
- **z-lab reference** uses HuggingFace transformers, targets H100, and needs 60+ GB VRAM.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash

# Build (requires CUDA 12+, CMake 3.18+, sm_86-compatible GPU)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j

# Fetch models
# Target (14.9 GB Q4_K_M): unsloth/Qwen3.5-27B-GGUF
# Draft  (3.46 GB BF16): z-lab/Qwen3.5-27B-DFlash (safetensors)
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/

# Streaming one-shot generate
python3 scripts/run.py --prompt "def fibonacci(n):"
echo "Write a haiku about GPUs" | python3 scripts/run.py

# Multi-turn chat REPL (streaming)
python3 examples/chat.py

# OpenAI-compatible HTTP server (drop-in for Open WebUI / LM Studio / Cline)
pip install fastapi uvicorn
python3 scripts/server.py --port 8000
# then point any OpenAI-compatible client at http://localhost:8000/v1
# with any API key (not checked) and model name "luce-dflash"

# Reproduce the paper numbers
python3 scripts/bench_llm.py       # HumanEval + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # minimal HE bench

# Run a single prompt
python3 scripts/tokenize_prompt.py "def sum_product(numbers):" --out /tmp/prompt.bin
build/test_dflash models/Qwen3.5-27B-Q4_K_M.gguf \
    models/draft/model.safetensors /tmp/prompt.bin 256 /tmp/out.bin \
    --fast-rollback --ddtree --ddtree-budget=22
python3 scripts/detokenize.py /tmp/out.bin
```

### 128K context mode

```bash
DFLASH27B_KV_Q4=1 DFLASH27B_PREFILL_UBATCH=16 \
build/test_dflash models/Qwen3.5-27B-Q4_K_M.gguf \
    models/draft/model.safetensors /tmp/long_prompt.bin 64 /tmp/out.bin \
    --fast-rollback --ddtree --ddtree-budget=16
```

**Requirements:**
- NVIDIA GPU, sm_86+ (RTX 3090, A10, A40, 4090, etc). Tested on RTX 3090.
- CUDA 12+
- 24 GB VRAM for Q4_K_M target + BF16 draft
- ~80 GB disk for model weights

## Files

| File | Description |
|------|-------------|
| `include/dflash27b.h`                 | Model constants + `dflash27b_last_error()` |
| `src/errors.cpp`                      | Thread-safe last-error string |
| `src/gguf_target_loader.cpp`          | Load Q4_K_M qwen35 target from GGUF into ggml tensors |
| `src/safetensors_draft.cpp`           | Load BF16 DFlash draft from safetensors |
| `src/qwen35_target_graph.cpp`         | Qwen3.5-27B hybrid forward with layer-feature capture and Q4 KV |
| `src/qwen3_dflash_graph.cpp`          | 5-layer non-causal block-diffusion draft graph |
| `src/kv_cache.cpp`                    | Sliding `target_feat` ring (4096 slots) |
| `src/delta_net_chunked.cpp`           | Chunked DeltaNet path for long prefills |
| `src/f16_convert.cu`                  | F32 ↔ F16 conversion helpers |
| `test/test_dflash.cpp`                | Main driver — chain / DDTree spec decode with `--stream-fd` |
| `test/test_vs_oracle.cpp`             | Draft graph vs PyTorch reference (cos sim 0.9999) |
| `test/test_generate.cpp`              | Autoregressive baseline for A/B |
| `scripts/run.py`                      | Streaming one-shot generate (auto-applies chat template) |
| `scripts/server.py`                   | OpenAI-compatible HTTP server (`/v1/chat/completions` streaming SSE) |
| `scripts/bench_llm.py`                | HumanEval + GSM8K + Math500 bench (loads HF datasets) |
| `scripts/bench_he.py`                 | Minimal 10-prompt HumanEval bench runner |
| `examples/chat.py`                    | Multi-turn chat REPL with streaming output |
| `scripts/tokenize_prompt.py`          | HF tokenizer → `int32` binary for `test_dflash` |
| `scripts/detokenize.py`               | Inverse |
| `scripts/gen_oracle.py`               | Generate PyTorch oracle inputs for `test_vs_oracle` |
| `RESULTS.md`                          | Full benchmark results and config sweep |

## Scope and limitations

This is a **research proof-of-concept**, not a production inference server.

- **Batch size 1 only.** Targets single-user local inference (Ollama / LM Studio use case), not multi-tenant serving.
- **One model pair only.** Hardcoded for Qwen3.5-27B Q4_K_M target + z-lab DFlash BF16 draft. Does not generalize to other models without rewriting the graph builders. The public draft checkpoint is exclusively for Qwen3.5-27B.
- **Greedy decoding.** No temperature sampling, no top-k/top-p.
- **Single GPU, CUDA only.** sm_86+ tested. No Metal, no ROCm, no multi-GPU.
- **Greedy only at the moment.** `temperature` / `top_p` on the OpenAI server are accepted but ignored (greedy-only result returned). Adding proper rejection sampling in `test_dflash.cpp`'s verify path is a weekend-sized contributor task (see [CONTRIBUTING.md](CONTRIBUTING.md) #3).
- **Model reload per turn.** Chat + server respawn `test_dflash` per request (~10 s first-token latency). Streaming arrives live after that. A persistent daemon that keeps the model resident is the next usability win ([CONTRIBUTING.md](CONTRIBUTING.md) #2).
- **Correctness.** `test_vs_oracle` validates the draft graph at cos sim 0.9999 vs the PyTorch DFlash reference. The target graph matches llama.cpp's `models/qwen35.cpp` semantically (Q/K/V norms, M-RoPE, V-head reorder, ssm_out ordering) and produces bit-identical output to `test_generate` in autoregressive mode.

The goal is to show that single-user inference on consumer hardware has way more headroom than the status quo suggests, and to publish the port so others can reproduce, critique, and extend it.

## Lessons from building this

**The draft is brittle to target quantisation noise.** The z-lab draft was trained against BF16 target features. On Q4_K_M target, per-position accept drops ~30 points vs the paper. Q5_K_M or Q6_K_M would likely recover most of it. We shipped Q4_K_M because it's the largest that fits on 24 GB with headroom for the DDTree verify tree.

**DDTree without chain pre-seed halves your AL.** The paper describes pure best-first tree construction. With greedy verify on a quantised target, the top-1 path needs to be guaranteed present at full chain depth, or the tree rescues an inferior rejected suffix instead of the real continuation. A single extra flag in `build_ddtree` (`chain_seed=true`) recovered AL from ~4 to ~9.

**`ggml_view_*` offsets silently overflow at tree verify.** Our original rollback path did per-layer `ggml_cpy` via `ggml_view_1d` into the KV cache's slot range. At budget=22 the offset + span exceeded `nbytes(cache)` and hit a `GGML_ASSERT` mid-run. Fixing it meant replacing the generic copy path with a direct kernel (`ggml_gated_delta_net_tree_persist`) that writes SSM intermediates in-place.

## Future work

See [CONTRIBUTING.md](CONTRIBUTING.md) for a sorted task list with file pointers. Headline items:

- **Daemon mode** to keep the 15 GB target resident across chat turns (drops first-token latency from ~10 s to ms).
- **Temperature / top-k / top-p sampling** via rejection sampling in the verify path.
- **Q5_K_M / Q6_K target** support to close the remaining AL gap to the paper.
- **Full llama.cpp integration** (new arch, `llama-speculative-dflash.cpp`, `llama-cli` / `llama-server` wiring).

## Citation

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: Block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Luce},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/dflash},
  year   = {2026}
}

@article{dflash2025,
  title  = {DFlash: Block-Diffusion Speculative Decoding},
  author = {z-lab},
  journal= {arXiv:2502.20762},
  year   = {2025}
}
```

## Contributing

This is a research codebase but deliberately structured so contributors can push both usability and performance without touching the CUDA kernels. See [CONTRIBUTING.md](CONTRIBUTING.md) for the sorted task list — everything from "add streaming output" (a few hours) to "port the megakernel verify path" (weeks).

Good first picks:
- **Streaming tokens** from `test_dflash` stdout so chat shows tokens live.
- **Daemon mode** that keeps the model resident across turns (drops chat latency from 10 s to ms).
- **Temperature / top-k** sampling in `spec_loop.cpp`.
- **OpenAI-compatible HTTP server** wrapping daemon mode (drop-in for Open WebUI / LM Studio / Cline).

## Community

Questions, ideas, or want to see what others are building on consumer GPUs? Join the [Luce Discord](https://discord.gg/yHfswqZmJQ).

---

MIT · [Lucebox](https://lucebox.com)

Built with [Claude](https://claude.ai)

Inspired by [z-lab/DFlash](https://arxiv.org/abs/2502.20762), [liranringel/ddtree](https://github.com/liranringel/ddtree), [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
