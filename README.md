<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  The first megakernel for hybrid DeltaNet/Attention LLMs.<br/>
  All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch.<br/>
  <a href="https://lucebox.com/blog/megakernel">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/NjweHTtTVj">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

```
                        Prefill      Decode      tok/J
Megakernel (RTX 3090)   37,800       413         1.87  @220W
llama.cpp  (RTX 3090)   11,247       267         0.76
Apple M5 Max               -         229         1.76
```

## What this is

A persistent CUDA kernel that processes the entire Qwen 3.5-0.8B forward pass in one dispatch. 18 DeltaNet layers + 6 attention layers, no CPU round-trips between them.

Qwen 3.5-0.8B uses a hybrid DeltaNet + Attention architecture. It's new, and no one has built a fused kernel for it yet. This is the first.

## Run

```bash
pip install -e .
python bench_pp_tg.py
```

Requires NVIDIA Ampere+ GPU, CUDA 12+, PyTorch 2.0+. Tested on RTX 3090.

## How it works

Each token goes through all 24 layers inside one kernel (82 blocks x 512 threads). DeltaNet recurrence stays in F32 registers. KV cache updates in-kernel. Layers sync via cooperative grid instead of separate launches.

BF16 weights, BF16 activations, FP32 accumulation. Weights loaded directly from HuggingFace.

## Files

```
kernel.cu            Decode megakernel
prefill.cu           Prefill (cuBLAS + standalone kernels)
torch_bindings.cpp   PyTorch C++ bindings
model.py             Weight loading + Decoder
setup.py             Build
bench_pp_tg.py       Benchmark
```

## Community

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/NjweHTtTVj).

---

MIT · [Lucebox](https://lucebox.com)

Built with [Claude](https://claude.ai)
