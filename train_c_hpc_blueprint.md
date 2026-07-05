# Project Blueprint: Optimizing `train.c` for CPU-Native Transformer Training

**Target hardware:** Intel Core i7-11700 (Rocket Lake / Cypress Cove), 8 physical cores / 16 threads, AVX-512, 16 MB L3, 512 KB L2/core, 48 KB L1D/core, dual-channel DDR4.
**Current model:** ~6.9M params (8 layers, d_model 256, d_ff 1024). **Stated target:** ~343M params (GPT-2 Medium: d_model 1024, 24 layers, 16 heads, d_ff 4096, vocab ~50k, ctx 1024).

This document is organized so the highest-leverage decisions come first. Sections 3–7 answer your five technical questions in depth; Section 1 reframes the goal; Section 8 is the concrete staged plan for the code.

---

## 1. The reality check that changes the plan

The optimizations you asked about are all real, and I've written them up below. But the framing "turn 343M-from-scratch into a professional-grade *training* run on one i7" collides with three hard numbers, and a good HPC engineer surfaces those before writing kernels.

### 1a. Compute budget (roofline / time-to-train)

Training cost is well-approximated by `6 × N_params × N_tokens` FLOPs (2 for the forward matmul-add, 4 for backward). Peak FP32 on your chip:

- Rocket Lake **client** implements AVX-512 with a *single fused 512-bit FMA* (it fuses the two 256-bit FMA ports). So peak FP32 is ~`8 cores × 16 lanes × 2 (FMA) × ~4 GHz ≈ 1.0 TFLOP/s`, and — importantly — **AVX-512 does *not* double FP32 throughput over AVX2 on this specific CPU**. Its wins are encoding, masking, and VNNI (see §5), not 2× FLOPs.
- A hand-written, well-blocked kernel realistically sustains 30–60% of peak: call it **300–600 GFLOP/s**.

Now the time. Even a *deliberately small* token budget for a 350M model — say 10B tokens, which is already on the low end for coherence — costs:

```
6 × 350e6 × 10e9  =  2.1e19 FLOP
2.1e19 / 5e11 GFLOP/s  ≈  4.2e7 s  ≈  486 days  ≈  1.3 years
```

At 300 GFLOP/s it's ~2.2 years. A genuinely good 350M model wants far more than 10B tokens (Chinchilla-optimal is ~7B; strong models use 50–300B), pushing this to **many years**. Every optimization in this document *combined* might buy you 3–8× over a naive `-O3` build — which turns a 2-year job into a ~4-month job that still produces a weak model. The compute wall is the binding constraint, and no amount of AVX-512 moves it into "feasible" for from-scratch 343M.

### 1b. Memory footprint (the current architecture won't fit)

At GPT-2-Medium shape with batch 4, seq 1024, the model *state* alone is ~5.6 GB (params 1.4 + Adam m/v 2.8 + grads 1.4). That's fine. The problem is your activation strategy: you store every intermediate for backward, **including the full attention probability matrix** `A_probs`, which is `B × H × T × T` per layer:

| Buffer (GPT-2 Medium, B=4, T=1024) | Size |
|---|---|
| Params (FP32) | 1.4 GB |
| Adam moments (m, v) | 2.8 GB |
| Gradients | 1.4 GB |
| **Stored attention probs, all 24 layers** | **~26 GB** |
| FFN hidden (`A_h` + `A_hg`), all layers | ~3.2 GB |
| Residual stream + other saved acts | several GB |

The stored `T×T` attention alone (~26 GB) blows past a 16–32 GB desktop. Scaling the current code is not a matter of bigger `malloc`s — it requires **online-softmax attention** (FlashAttention-style: recompute the probs in the backward pass instead of storing them) and **activation checkpointing** (Chen et al. 2016: keep only layer boundaries, recompute the rest). That's a real re-architecture, not a tuning pass.

### 1c. The mismatch is in the *goal*, not the effort

Given your portfolio context, the win condition you actually want — an impressive, defensible systems project — is served *better* by these optimizations applied to a right-sized model than by a stalled multi-year run. Two framings that keep everything you're excited about:

1. **Right-size the model to converge.** A 10–50M model on this engine, fully optimized, trains to real, demonstrable convergence in hours-to-days. You get loss curves, samples, ablations — a complete story. This is the path I'd pick for a portfolio piece.
2. **Make the engine itself the artifact.** Ship a benchmarked CPU GEMM/attention library with a roofline analysis: "naive → +`-march=native` → +register-blocking → +cache-blocking → +VNNI INT8 inference," each stage with measured GFLOP/s and % of peak. That's a genuinely strong HPC portfolio project, and 343M can appear as an *inference* target (which *is* feasible on CPU) rather than a training one.

Everything below makes both of those excellent. Keep building — just point it at a target the hardware can reach.

---

## 2. The zeroth optimization (do this before any intrinsics)

Your build line is:

```
gcc -O3 -fopenmp train.c -o train.exe -lm
```

There is **no `-march`**. Plain `-O3` targets the baseline x86-64 ISA (SSE2) — so the compiler is emitting *scalar/SSE* code, not AVX2 or AVX-512, for every loop you asked about vectorizing. You're leaving the single biggest, free speedup on the table. Fix the flags first, measure, *then* decide whether hand-written intrinsics are even needed:

```
gcc -O3 -march=native -mtune=native -ffast-math -funroll-loops \
    -fopenmp train.c -o train -lm
```

- `-march=native` unlocks AVX2/AVX-512/FMA/VNNI on this CPU.
- `-ffast-math` lets the vectorizer reorder FP reductions (your dot-products and softmax sums) and contract into FMAs. It changes FP semantics slightly — harmless for training, but be deliberate about it. If you want to be conservative, `-fno-math-errno -ffp-contract=fast -fassociative-math` gets most of the benefit.
- Verify what actually vectorized:

```
gcc ... -O3 -march=native -fopt-info-vec-optimized 2>&1 | grep -Ei "train.c.*loop"
```

On many kernels, GCC 11+/Clang 14+ at `-O3 -march=native` will *already* produce near-microkernel-quality AVX-512 for the simple inner loops. **Measure this build before writing a single intrinsic.** Hand-rolling only pays where the compiler demonstrably falls short.

---

## 3. SIMD vectorization of `matmul` and `gelu`

### 3a. What your `matmul_forward` already does right

Your inner loop is:

```c
for (int k = 0; k < K; k++) {
    float a = x[k];
    const float *wr = Wt + (size_t)k * N;
    for (int n = 0; n < N; n++) o[n] += a * wr[n];   // contiguous FMA over n
}
```

That inner loop is a SAXPY over contiguous memory — the *ideal* auto-vectorization case (unit-stride load `wr[n]`, unit-stride load/store `o[n]`, fused multiply-add). With `-march=native` the compiler turns it into `_mm512_fmadd_ps` with no help from you. So "how do I saturate the registers in matmul" mostly answers itself once the flags are right. The weakness of this loop is **not** vectorization — it's cache reuse (§4): it re-reads and re-writes the whole `o` row from memory on every `k`, and streams `Wt` from DRAM once per output row.

### 3b. When to hand-write a microkernel

The reason to reach for intrinsics is **latency hiding via register blocking**, not the SIMD width itself. An FMA on this core has ~4–5 cycle latency and ~2/cycle throughput, so you need ~8–10 *independent* FMA chains in flight to saturate the unit. A single accumulator row can't do that. The fix is to compute an `MR × 16` output tile with `MR` accumulators living in ZMM registers (there are 32 of them under AVX-512), each an independent chain:

```c
#include <immintrin.h>
/* C[0:MR,0:16] += A[0:MR,0:K] @ Bpan[0:K,0:16]
 * Bpan is a 16-wide column panel of W, row-major [K][16] (contiguous per k).
 * Matches your Wt layout [K,N]: Bpan[k] is a contiguous 16-float slice. */
#define MR 8
static void micro_8x16(const float *A, int lda,
                       const float *Bpan,
                       float *C, int ldc, int K) {
    __m512 c0=_mm512_loadu_ps(C+0*ldc), c1=_mm512_loadu_ps(C+1*ldc),
           c2=_mm512_loadu_ps(C+2*ldc), c3=_mm512_loadu_ps(C+3*ldc),
           c4=_mm512_loadu_ps(C+4*ldc), c5=_mm512_loadu_ps(C+5*ldc),
           c6=_mm512_loadu_ps(C+6*ldc), c7=_mm512_loadu_ps(C+7*ldc);
    for (int k = 0; k < K; k++) {
        __m512 b = _mm512_loadu_ps(Bpan + (size_t)k*16);   /* 1 load feeds 8 FMAs */
        c0=_mm512_fmadd_ps(_mm512_set1_ps(A[0*lda+k]), b, c0);
        c1=_mm512_fmadd_ps(_mm512_set1_ps(A[1*lda+k]), b, c1);
        c2=_mm512_fmadd_ps(_mm512_set1_ps(A[2*lda+k]), b, c2);
        c3=_mm512_fmadd_ps(_mm512_set1_ps(A[3*lda+k]), b, c3);
        c4=_mm512_fmadd_ps(_mm512_set1_ps(A[4*lda+k]), b, c4);
        c5=_mm512_fmadd_ps(_mm512_set1_ps(A[5*lda+k]), b, c5);
        c6=_mm512_fmadd_ps(_mm512_set1_ps(A[6*lda+k]), b, c6);
        c7=_mm512_fmadd_ps(_mm512_set1_ps(A[7*lda+k]), b, c7);
    }
    _mm512_storeu_ps(C+0*ldc,c0); _mm512_storeu_ps(C+1*ldc,c1);
    _mm512_storeu_ps(C+2*ldc,c2); _mm512_storeu_ps(C+3*ldc,c3);
    _mm512_storeu_ps(C+4*ldc,c4); _mm512_storeu_ps(C+5*ldc,c5);
    _mm512_storeu_ps(C+6*ldc,c6); _mm512_storeu_ps(C+7*ldc,c7);
}
```

Notes that make this fast rather than merely vectorized:
- **One `b` load + one broadcast feed 8 FMAs** — the arithmetic intensity inside the kernel is high, so you're compute-bound on the FMA unit, which is the goal.
- `MR=8` gives 8 in-flight chains; bumping to `MR=12` or `14` (still leaving registers for `b`) hides FMA latency more completely. Tune empirically.
- Align allocations to 64 bytes (`posix_memalign(&p, 64, bytes)` / `_mm_malloc`) and switch to `_mm512_load_ps` — on Rocket Lake the aligned/unaligned gap is small, but it removes split-load penalties at cache-line boundaries.
- The `N` dimension is tiled by 16 (one ZMM). Handle the `N % 16` remainder with a masked tail (`_mm512_maskz_loadu_ps` + `__mmask16`) — this is exactly the AVX-512 feature that AVX2 lacks and where AVX-512 earns its keep here.
- Under AVX-512 all-core load the clock drops (frequency offset); don't be surprised if per-core GFLOP/s is below the nominal-boost estimate. This is expected, not a bug.

Again: build the `-march=native` version first and profile. If GCC already emits this shape (it often does), skip the hand-written kernel and spend your effort on §4, which matters more.

### 3c. `gelu` vectorization — real, but low-leverage

Your GELU calls scalar `tanhf()` per element. `tanhf` is a libm function that will *not* auto-vectorize (the compiler can't vectorize a libm call without a vector-math library), so this loop stays scalar even with `-march=native`. Two fixes:

- **Link a vector math library:** [SLEEF](https://sleef.org) gives you `Sleef_tanhf16_u10(__m512)` (and `expf16`) — drop-in vectorized transcendentals at controlled ULP error.
- **Switch to the sigmoid-approx GELU** `x·σ(1.702x)`, which needs one `exp` instead of a `tanh`, and hand-vectorize `exp` with the standard Cephes-style range-reduction + polynomial in AVX-512. This changes the numerics marginally from the tanh-approx but is a common, accepted substitution.

**But keep perspective:** matmuls are >90–95% of your FLOPs. Vectorizing GELU might shave a few percent off total step time. Do it after the GEMMs, not before.

---

## 4. Cache blocking for the feed-forward GEMMs

First, a terminology correction worth making in your writeup: what you want here is **cache-*aware* (blocked/tiled) GEMM**, not "cache-oblivious." Cache-oblivious is a specific recursive technique that avoids naming cache sizes on purpose; high-performance BLAS does the opposite — it *hard-codes* block sizes tuned to L1/L2/L3 (the GotoBLAS / BLIS approach). For a fixed target CPU, cache-aware wins.

### 4a. Why the current FFN thrashes (you diagnosed this correctly)

At GPT-2-Medium shape, `W1` is `[1024, 4096]` = 16 MB, essentially the whole 16 MB L3. Your `matmul_forward` streams `Wt` from memory once per output row. With `M = B·T = 4096` rows, you read all 16 MB of `W1` **4096 times** = ~64 GB of DRAM traffic for a *single* FFN up-projection in *one* layer. At ~40 GB/s practical bandwidth that's ~1.6 s of pure memory stalls per layer — this is the L3 thrashing you're worried about, and it's the dominant cost, not the arithmetic.

### 4b. The fix: block + pack (Goto/BLIS)

The standard structure has three nested blocking levels plus packing:

- **`Kc` (K-block, ~256–512):** slice the contraction dimension so a `Kc × Nr` panel of `W` fits in L1/L2 and gets reused across many rows of the activation matrix.
- **`Mc` (M-block, ~72–144):** slice the activation rows so an `Mc × Kc` panel of `A` stays resident in L2.
- **`Nc` (N-block, up to a few thousand):** slice output columns so the `Kc × Nc` panel of `W` fits in L3.
- **Packing:** copy each `A`-panel and `W`-panel into small contiguous scratch buffers *once* before the microkernel runs over them. Packing is the technique that actually delivers the speedup — it gives the microkernel unit-stride streaming with no TLB thrash and no strided cache-line waste. Un-packed blocked GEMM leaves most of the win on the table.

Order the loops so the reused panel stays hot: for each `Nc` block, for each `Kc` block, pack `W[Kc,Nc]` into L3; for each `Mc` block, pack `A[Mc,Kc]` into L2; run the `MR×16` microkernel over the tile. The 16 MB weight matrix now gets read from DRAM roughly *once* instead of 4096 times.

A reasonable starting point for this cache hierarchy: `MR×NR = 8×16` (or `14×16`), `Kc ≈ 384`, `Mc ≈ 96`, `Nc ≈ 3072`. These are starting values — sweep them and keep what profiles best.

### 4c. The honest recommendation

Writing a fully-packed, blocked, multi-level GEMM that beats the compiler is a multi-week effort, and a production engineer would not do it when **OpenBLAS, Intel MKL/oneMKL, or oneDNN already hit 85–95% of peak** on exactly this problem. For a professional-grade engine, the pragmatic move is:

```c
// link -lopenblas (or MKL), replace matmul_forward's core with:
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0f, in, K, Wt, N, 0.0f, out, N);
// then add bias in a vectorized pass
```

This one change typically dwarfs everything in §3. **Choose based on your goal:** if the point is *learning* GEMM optimization (great portfolio material), hand-roll the blocked kernel and benchmark it against MKL as your baseline — that comparison *is* the deliverable. If the point is a fast engine, call MKL and move on.

---

## 5. "Quantized training" — correcting the premise

This is the section where the plan needs the most adjustment, because the request as stated ("weight updates happen in smaller data types") describes something that doesn't produce stable training. Here's the accurate picture for *this CPU*.

### 5a. Mixed precision (BF16/FP16) is *not* a compute win on Rocket Lake

GPU mixed precision (BF16/FP16 matmul + FP32 master weights + loss scaling) relies on hardware that does BF16/FP16 *matmul acceleration*. Rocket Lake has **neither AVX512-BF16 nor AVX512-FP16** (those arrived on client with Sapphire Rapids). There is no hardware FMA for half precision here — converting to BF16/FP16 for compute would run through software conversion and be *slower*, not faster. BF16 could serve as a *storage* format to halve memory traffic in a bandwidth-bound kernel, but with no fast conversion instruction the benefit is marginal and usually not worth the complexity. **Skip half-precision compute on this chip.**

### 5b. INT8 via AVX-512 VNNI is the one that's real — for *inference/compute*, not weight updates

Rocket Lake *does* have **AVX-512 VNNI**. `_mm512_dpbusd_epi32` computes, per 512-bit register, 16 INT32 accumulators, each summing four `uint8 × int8` products — **64 INT8 MACs in one instruction**, up to ~4× the FP32 MAC rate. That's a genuine throughput lever. But two things must be clear:

- **You cannot do the optimizer in INT8.** AdamW's `m`, `v`, and the master weights must stay FP32 (BF16 with care, but see 5a). INT8 weight *updates* would destroy training within a few steps — the update magnitudes are far below INT8 resolution. Every credible quantized-training method (LLM.int8, SwitchBack, Jetfire, etc.) keeps **FP32 master weights and FP32 optimizer state**, and quantizes *only the matmul operands* for the forward (and sometimes backward) compute, using a straight-through estimator (STE) for gradients through the quantizer.
- **QAT is stacking a second research problem on top of the first.** Doing correct INT8 GEMM (per-tensor or per-channel scales, requantization of the INT32 accumulator back to the next layer's INT8 domain, STE, keeping accuracy) *while also* training a large model from scratch in hand-written C is a lot. It's a great standalone project for *inference* (quantize a trained model, benchmark VNNI throughput vs FP32) — which, notably, is the path that makes 343M *runnable* on this CPU.

### 5c. What actually relieves your bandwidth bottleneck

Your instinct that 343M FP32 is bandwidth-limited is right — but the fix is mostly §4, not quantization:

1. **Cache-block the GEMMs (§4).** This converts the hot matmuls from bandwidth-bound to compute-bound. It's the highest-leverage bandwidth fix and it's precision-neutral.
2. **Store tokens as `uint16`, not `int32`.** GPT-2 vocab (50257) fits in 16 bits — halves the dataset file *and* the bandwidth of every embedding gather.
3. **Fuse elementwise ops** (bias-add, residual-add, GELU) into the producing loop so intermediate tensors aren't written to and re-read from DRAM.
4. **Only then**, if you've done all that and profiling still shows the GEMMs memory-bound, reach for VNNI INT8 on the forward pass.

---

## 6. Threading and affinity (there is no NUMA here)

Correction up front: **a single-socket i7 is not a NUMA system.** It has one memory controller and one memory domain (UMA). `libnuma`, `numa_alloc_onnode`, `numactl --membind`, cross-node first-touch — none of it applies. Don't spend time there. What *does* matter on this chip is thread pinning and the physical-vs-logical decision.

### 6a. Use 8 threads (physical), not 16 (logical), for FP-bound work

Both hyperthreads on a core share the *same* FMA units. For compute-bound matmul that already saturates the FMA pipeline, a second thread on the same core doesn't add throughput — it adds contention and cache pressure, and often *lowers* GFLOP/s. Your code prints `omp_get_max_threads()` (defaults to 16); for the matmul-heavy path, 8 pinned threads is usually faster. Measure both, but expect 8 to win.

### 6b. Pin threads to prevent migration

Portable OpenMP way (set before launch):

```bash
export OMP_NUM_THREADS=8
export OMP_PLACES=cores        # one place per physical core
export OMP_PROC_BIND=close     # keep threads on their assigned cores
```

`OMP_PLACES=cores` binds one thread per physical core (ignoring the sibling hyperthread); `close` prevents the runtime from migrating threads across cores mid-run, which is the migration overhead you want gone. Try `spread` too — for memory-bound phases it can help by distributing across the ring.

Manual `pthread` pinning (Linux), if you ever drop OpenMP for a custom pool:

```c
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
cpu_set_t set; CPU_ZERO(&set); CPU_SET(physical_core_id, &set);
pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
```

Your build comments (`.\train.exe`) suggest you may be on Windows/PowerShell. The `OMP_*` environment variables work identically there; the native pinning API is `SetThreadAffinityMask` instead of `pthread_setaffinity_np`. If you're on WSL2, note it virtualizes CPU topology — pin from the Linux side and verify with `lscpu`/`htop` that threads actually land where you asked.

### 6c. Parallelize the right loops

You already have the right idea (OpenMP on batch/row loops). Two refinements at scale: (1) the tied-head and cross-entropy loops over `VOCAB_SIZE` become the second-hottest region once vocab is ~50k — make sure those are parallelized (cross-entropy already is; the tied-head backward's `dwte` accumulation is parallel over `vv`, good). (2) Consider `schedule(static)` (which you use) vs `guided` for the ragged causal-attention loop, where row `i` does `O(i)` work — static gives thread 0 the cheap early rows and thread N the expensive late rows, causing imbalance. `schedule(dynamic, 8)` or `guided` balances the triangular workload better.

---

## 7. Memory-mapped streaming with `mmap`

`mmap` for the token dataset is a reasonable, standard choice — but one premise needs correcting.

### 7a. There are no page faults in the backward pass

The token file is read **only during batch sampling**, at the top of each step, to look up embeddings. The backward pass operates entirely on activations, weights, gradients, and Adam state — all ordinary heap RAM, none of it mmap'd. So mapping the dataset has *zero* interaction with backprop; there are no dataset page faults during the backward pass to avoid. The page faults you'd see are only when sampling touches a not-yet-resident region of the token file, which happens during forward-pass embedding lookup at worst.

### 7b. Do it like this

```c
#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int fd = open("train.bin", O_RDONLY);
struct stat st; fstat(fd, &st);
size_t nbytes = st.st_size;
const uint16_t *toks = mmap(NULL, nbytes, PROT_READ, MAP_PRIVATE, fd, 0);
madvise((void *)toks, nbytes, MADV_RANDOM);   // you sample random windows
long ntok = nbytes / sizeof(uint16_t);
```

- `MADV_RANDOM` is the correct hint because you draw random start offsets — it *disables* the kernel's sequential readahead, which would otherwise fetch pages you won't use. (If you switched to sequential streaming, `MADV_SEQUENTIAL` + `MADV_WILLNEED` would be right instead.)
- Store tokens as `uint16` (vocab fits) to halve the mapping and the fault traffic.
- For very large files, back the mapping with **huge pages** (transparent huge pages, or `MAP_HUGETLB`) to cut TLB misses on the big mapping.
- To hide fault latency entirely, prefetch the *next* batch's windows in a background thread while the current step computes — a simple double-buffer.

### 7c. Keep the scope honest

`mmap` solves "I don't want to load a 20–40 GB token file into RAM." It does **nothing** for the real memory pressure at 343M, which is the model state + activations (§1b), all of which live in RAM regardless. Don't expect mmap to be what makes a large model fit — it isn't.

---

## 8. Staged plan for `train.c`

Ordered by return on effort. Measure `tok/s` and estimated GFLOP/s after every stage — the measurements are the point, and they're your portfolio narrative.

**Stage 0 — Flags + baseline (minutes).** Add `-march=native -ffast-math -funroll-loops`. Re-run. Record the speedup; confirm with `-fopt-info-vec-optimized` what vectorized. This alone is often 2–4×.

**Stage 1 — Thread tuning (minutes).** `OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`. Compare 8 vs 16 threads. Switch the causal-attention loop to `schedule(guided)`.

**Stage 2 — GEMM (hours–weeks, pick one).** Either link OpenBLAS/MKL and route the three hot projections (QKV/O, FFN-up, FFN-down) and the tied head through `cblas_sgemm` — fastest path to a fast engine — *or* hand-write the blocked+packed microkernel from §3b/§4b and benchmark it against MKL as your baseline (the better portfolio story). This is where the real speed is.

**Stage 3 — Vectorized GELU (hours).** SLEEF `tanhf16`, or sigmoid-GELU with hand-vectorized `exp`. Small but easy once GEMMs are done.

**Stage 4 — Memory re-architecture (only if scaling up).** Online-softmax attention (stop storing `A_probs`; recompute in backward) + activation checkpointing at layer boundaries. Required before any model where §1b's table exceeds your RAM. `uint16` tokens + `mmap` per §7.

**Stage 5 — INT8 inference via VNNI (project in its own right).** Quantize a *trained* model, implement `vpdpbusd`-based INT8 GEMM with per-channel scales + requant, benchmark throughput vs FP32. This is the realistic way to make 343M *run* on this CPU. Leave INT8 *training* alone unless it becomes its own research effort with FP32 master weights and STE.

**Stage 6 — Pick a convergeable target.** For an end-to-end trained result with real loss curves and samples, size the model so `6 × N × tokens` fits in hours-to-days at your measured GFLOP/s (§1a). A 10–50M model gets you there; 343M-from-scratch does not.

---

## 9. Standard references (the "research techniques" you asked for)

- **Goto & van de Geijn**, "Anatomy of High-Performance Matrix Multiplication" (2008) — the packing/blocking method behind every fast GEMM. Pair with the **BLIS** papers and source.
- **Williams, Waterman, Patterson**, "Roofline: An Insightful Visual Performance Model" (2009) — the framework for the §1a compute-vs-bandwidth analysis; build a roofline for your kernels.
- **Karpathy, `llm.c`** — a from-scratch C/CUDA GPT-2 training implementation; the closest reference codebase to what you're building, and excellent for cross-checking your backward math and layout choices.
- **Intel oneDNN, oneMKL/MKL, OpenBLAS** — the optimized GEMM/primitive libraries to benchmark against (and probably use).
- **SLEEF** — vectorized transcendentals (`exp`, `tanh`) for the GELU kernel.
- **Dao et al., FlashAttention** (2022) — the online-softmax attention you need for Stage 4 (adapt the algorithm; the CPU version is the memory-saving math, not the CUDA kernel).
- **Chen et al.**, "Training Deep Nets with Sublinear Memory Cost" (2016) — activation checkpointing.
- **Hoffmann et al., "Chinchilla"** (2022) — the token-budget scaling law behind the §1a time estimates.
- **Intel Intrinsics Guide** and **Agner Fog's instruction tables** — per-instruction latency/throughput for Rocket Lake, essential for tuning the microkernel and choosing `MR`.
