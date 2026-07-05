# pure-c-llm

A complete language-model stack written **from scratch in pure C** — and a native **CUDA** trainer — with **zero machine-learning frameworks**. No PyTorch, no TensorFlow, no NumPy in the training path. Every layer is hand-written: the BPE tokenizer, the tensor math, the transformer, the analytic backpropagation, the AdamW optimizer, the HPC kernels, a function-preserving model-growth engine, and the text generator.

Built and benchmarked on a single 15 W laptop CPU (Intel i7-1165G7), then scaled to GPU on free Kaggle T4s.

---

## What it does

```
raw text ──▶ BPE tokenizer ──▶ [ transformer: embed → N×(attention + FFN) → tied head ]
                                     │                    │
                              train (C, CPU)        train (CUDA, GPU)
                                     │
                              Net2Net growth  ──▶  bigger model, same function
                                     │
                                 generate ──▶ streamed text completion
```

It **trains to convergence** (verified), it **grows** a trained small model into a larger one without losing what it learned, and it **generates text** — all locally, all in C.

---

## Highlights (measured, reproducible)

| Result | Number |
|---|---|
| Pure-C transformer: forward + **hand-written backprop** + AdamW | trains to loss ≈ 0.01 on a learnable task |
| HPC GEMM ladder: naive → hand-optimized, vs. OpenBLAS | **90–96 % of OpenBLAS** single-thread |
| INT8 inference via AVX-512 VNNI | **~4.5×** FP32 throughput, **0.56 %** error |
| Net2Net function-preserving growth (6.9M → 52M params) | grown model starts at the **small model's loss (6.85), not random (7.62)** |
| CUDA 343M trainer on Kaggle T4 (cuBLAS + batched attention) | 27 s/step → **~3 s/step** after optimization; auto-resume across 9 h sessions |
| Local generation | Shakespeare-flavored text at **~60 tok/s** |

The optimization work is validated with a **roofline model** built from measured compute/memory ceilings on the target hardware.

---

## Repository layout

**Core engine (pure C, CPU)**
| File | Role |
|---|---|
| `tokenizer.c` / `.h` | Byte-level BPE: train a vocab, encode, decode |
| `tensor.c` / `.h` | Flat-array tensor math: matmul, add, layernorm, softmax, gelu (OpenMP) |
| `transformer.c` / `.h` | One pre-norm transformer block; KV-cached decode path |
| `model.c` / `.h` | Full GPT: embeddings → blocks → final norm → **tied** LM head |
| `main.c` | Interactive KV-cached generation loop |
| `train.c` | The training engine: analytic backprop for every op + AdamW + checkpointing |
| `generate.c` | Standalone generator; reads any checkpoint, dims from the header |
| `grow_model.c` | **Net2Net** function-preserving model growth (widen + deepen) |

**HPC study**
| File | Role |
|---|---|
| `gemm_bench.c` | The optimization ladder: naive → register-blocked → cache-blocked/packed → multithreaded → INT8 VNNI → vs OpenBLAS, with measured roofline anchors |

**GPU trainer (native CUDA)**
| File | Role |
|---|---|
| `gpu_kernels.cu` / `.h` | CUDA kernels: tiled + cuBLAS GEMMs, batched causal attention, LayerNorm, GELU, fused cross-entropy, AdamW |
| `train_kaggle.cu` | Orchestrator: flat weight arena, `mmap` token streaming, full fwd/bwd on GPU, atomic checkpoint + auto-resume + session-timeout guard |
| `download_data.py` | The **only** Python — permitted role: download raw text and dump it to the binary token format |

---

## Build & run

Everything compiles with a stock GCC (`-march=native` is the single biggest free win — it unlocks AVX2/AVX-512). CUDA needs `nvcc`.

**Tokenizer** — train a 2048-token BPE vocab and tokenize a corpus:
```bash
gcc -O2 tokenizer.c -o tokenizer
./tokenizer train shakespeare.txt 2048 vocab.bin train.bin
```

**Train the small model** (CPU):
```bash
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp train.c -o train -lm
./train 500 3e-4 - ckpt_small.bin
```

**Grow it** (function-preserving 2× widen + 2× deepen):
```bash
gcc -O2 grow_model.c -o grow_model
./grow_model ckpt_small.bin ckpt_grown.bin
```

**Fine-tune the grown model** — note the compile-time dim overrides:
```bash
gcc -O3 -march=native -ffast-math -fopenmp -DD_MODEL=512 -DN_HEADS=16 -DD_FF=2048 -DN_LAYERS=16 train.c -o train_grown -lm
./train_grown 600 3e-4 ckpt_grown.bin ckpt_grown_ft.bin
```

**Generate**:
```bash
gcc -O3 -march=native -fopenmp -DTOKENIZER_LIB generate.c tokenizer.c -o generate -lm
./generate ckpt_grown_ft.bin vocab_shk.bin "ROMEO:" 100 0.8 40
```

**HPC benchmark** (single-thread kernel study):
```bash
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp gemm_bench.c -o gemm_bench -lm
./gemm_bench
```

**GPU training** (Kaggle T4):
```bash
nvcc -O3 -arch=sm_75 --use_fast_math -o train_gpu train_kaggle.cu gpu_kernels.cu -lcublas
./train_gpu
```

Trained checkpoints are published under **[Releases](../../releases)**. The multi-gigabyte GPU training-state checkpoint lives on Kaggle (too large for GitHub).

---

## The optimization ladder (CPU GEMM)

Every rung is measured against a compute roof and a memory roof on the target chip:

| Rung | Technique | GFLOP/s | of peak |
|---|---|---|---|
| v0 | naive `i-j-k` | 0.8 | 1 % |
| v1 | `i-k-j` (cache-friendly inner loop) | 20 | 18 % |
| v2 | register-blocked AVX-512 microkernel | 86 | 79 % |
| v3 | + packing / cache-blocking (Goto/BLIS) | holds 68 % where v2 collapses on >L3 shapes |  |
| v3-MT | + OpenMP over 4 cores | ~215 | — |
| v4 | INT8 via VNNI `vpdpbusd` | 358 GOP/s | ~4.5× FP32 |
| v5 | reference: **OpenBLAS** | v3 reaches **90–96 %** of it |  |

---

## Status

- ✅ Pure-C tokenizer → tensor → transformer → model → training → growth → generation, all working and verified
- ✅ HPC ladder complete with roofline; INT8 VNNI inference
- ✅ CUDA 343M trainer running on Kaggle with checkpoint/resume
- 🔜 FP16 tensor-core kernels (the next ~3–4× on GPU); bridging trained GPU weights back to the local C generator

---

## References

This project is an assembly of well-established ideas. Key sources:

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer.
- Radford et al., *Language Models are Unsupervised Multitask Learners* / GPT-2 (2019) — the architecture template (pre-norm, learned positions, tied embeddings).
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016); Gage (1994) — byte-pair encoding.
- Ba, Kiros, Hinton, *Layer Normalization* (2016); Hendrycks & Gimpel, *GELU* (2016).
- Kingma & Ba, *Adam* (2014); Loshchilov & Hutter, *Decoupled Weight Decay Regularization* / AdamW (2019).
- **Chen, Goodfellow, Shlens, *Net2Net: Accelerating Learning via Knowledge Transfer* (2015)** — the function-preserving model growth implemented in `grow_model.c`.
- Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication* (2008) — the packing/blocking behind the fast GEMM.
- Williams, Waterman, Patterson, *Roofline* (2009) — the performance model used throughout the HPC study.
- Dao et al., *FlashAttention* (2022) — online-softmax attention (the path for scaling).
- Karpathy, *llm.c* — the closest reference codebase; used to cross-check the backward math.

## License

MIT — see `LICENSE`.
