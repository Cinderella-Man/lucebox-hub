<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  One CUDA kernel. All 24 layers. Zero round-trips to the CPU.<br/>
  <a href="https://lucebox.com/blog/megakernel">Blog post</a> · <a href="RESULTS.md">Full benchmarks</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

We fused the entire forward pass of [Qwen 3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) into a single persistent CUDA kernel: 18 DeltaNet layers + 6 full attention layers, all in one dispatch.

```
413 tok/s decode on RTX 3090   (1.55x over llama.cpp)
37,800 tok/s prefill            (3.4x over llama.cpp)
1.87 tok/J at 220W              (matching Apple M5 Max)
```

## Why

Qwen 3.5-0.8B isn't a standard transformer. It alternates DeltaNet (linear attention with recurrence) and full attention in a 3:1 ratio. No framework optimizes for this. llama.cpp runs it generically and gets 267 tok/s on the same GPU.

The megakernel keeps everything on the SMs: weights stay in registers, state stays in shared memory, layers execute back-to-back with cooperative grid sync instead of kernel launches.

## Quick start

```bash
git clone https://github.com/Luce-Org/luce-megakernel.git
cd luce-megakernel
pip install -e .
python bench_pp_tg.py
```

**Requires:** RTX 3090 or newer (Ampere+), CUDA 12+, PyTorch 2.0+

## How it works

```
Token in
  │
  ├─ Layer 0-2:  DeltaNet  (recurrence in F32 registers, conv1d, gated output)
  ├─ Layer 3:    Attention  (QKV + RoPE + causal softmax + KV cache)
  ├─ Layer 4-6:  DeltaNet
  ├─ Layer 7:    Attention
  │   ...repeat 6x...
  ├─ Layer 23:   Attention
  │
  └─ All inside one kernel launch (82 blocks × 512 threads)
```

- **BF16 weights, BF16 activations, FP32 accumulation**
- DeltaNet state (128×128 per head) lives in registers across layers
- KV cache in BF16, updated in-kernel
- Cooperative grid sync between layers (no CPU involvement)
- Prefill uses cuBLAS GEMMs + standalone DeltaNet/attention kernels

## Benchmarks

RTX 3090, Qwen 3.5-0.8B BF16, pp520 tg128:

| | Prefill | Decode |
|---|---|---|
| **Megakernel** | **37,800** | **413 tok/s** |
| llama.cpp | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |
| Apple M5 Max | - | 229 |

Power sweep ([full data](RESULTS.md)):

| 420W (stock) | 300W | **220W** | 150W |
|---|---|---|---|
| 433 tok/s | 432 tok/s | **411 tok/s** | 194 tok/s |
| 1.38 tok/J | 1.44 tok/J | **1.87 tok/J** | 1.29 tok/J |

## Files

```
kernel.cu            Decode megakernel (the whole thing)
prefill.cu           Prefill kernels + cuBLAS orchestration
torch_bindings.cpp   PyTorch C++ interface
model.py             HuggingFace weight loader + Decoder
setup.py             Build
bench_pp_tg.py       Prefill + decode benchmark
final_bench.py       Benchmark with power measurement
```

## License

MIT
