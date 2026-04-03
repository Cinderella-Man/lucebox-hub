# Benchmark Results

## Hardware

| Machine | GPU/Chip | Memory |
|---------|----------|--------|
| Lucebox | NVIDIA RTX 3090 (24GB VRAM) | 32GB RAM |
| MacBook Pro | Apple M5 Max | 36GB Unified |

## RTX 3090: pp520 tg128

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | **37,800** | **413** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.55x** | **3.8x** |

## Apple M5 Max

| Method | tok/s |
|--------|:---:|
| LM Studio (llama.cpp) BF16 | 229 |

## Power Efficiency (DVFS)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|---|---|---|---|---|---|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot: 220W, 1.87 tok/J.
