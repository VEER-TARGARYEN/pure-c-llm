/*
 * transformer.c  --  Implementation of the pre-norm Transformer block.
 *
 * Build the benchmark:
 *   gcc -O3 -fopenmp -DTEST_TRANSFORMER transformer.c tensor.c -o transformer_test.exe -lm
 * Run:
 *   .\transformer_test.exe
 *
 * The big linear layers (QKV, output proj, FFN) go through Piece 2's matmul,
 * which already uses cache-friendly i-k-j ordering and parallelizes over rows.
 * The attention core is written by hand here so we can (a) parallelize over
 * heads and (b) apply the causal mask by construction.
 */

#include "transformer.h"
#include "tensor.h"
#include <math.h>     /* sqrtf, expf */
#include <stddef.h>   /* size_t      */

/* ==========================================================================
 * add_bias:  Y[r, :] += bias[:]   (broadcast a length-`cols` bias over rows)
 * A linear layer is matmul followed by this. Parallel over rows.
 * ========================================================================== */
static void add_bias(float *Y, const float *bias, int rows, int cols)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float *yr = Y + (size_t)r * cols;
        for (int c = 0; c < cols; c++)
            yr[c] += bias[c];
    }
}

/* ==========================================================================
 * multihead_attention: causal, self-attention, parallel over heads.
 * --------------------------------------------------------------------------
 * For each head h and each query position i we:
 *   1) score q_i against every key k_j with j <= i, scaled by 1/sqrt(d_head);
 *   2) softmax those scores (numerically stable: subtract the row max);
 *   3) take the probability-weighted sum of value vectors v_j.
 *
 * THE CAUSAL MASK.  A lower-triangular mask means position i may only attend to
 * positions j <= i (itself and the past), never j > i (the future). We enforce
 * it the cheap way: the inner loops simply STOP at j == i. Never visiting the
 * upper triangle is mathematically identical to setting those scores to -inf
 * before softmax (exp(-inf) = 0), but it also skips ~half the work.
 *
 * PARALLELISM.  The 16 heads are fully independent, so `#pragma omp parallel
 * for` over h spreads them across the 8 CPU threads (~2 heads each). All the
 * per-head buffers below are indexed by h, so threads never collide. Note we do
 * NOT call the tensor.h kernels in here: those parallelize internally, and we
 * want the parallelism at the head level, not nested inside it.
 *
 * CACHE BEHAVIOUR.  d_head (96) contiguous floats are the innermost dimension
 * of every dot product and every value accumulation, so each inner loop streams
 * sequential memory -- friendly to the L1/L2 and auto-vectorizable.
 * ========================================================================== */
void multihead_attention(const float *Q, const float *K, const float *V,
                         float *out, float *scores, int T)
{
    const float scale = 1.0f / sqrtf((float)D_HEAD);

#pragma omp parallel for schedule(static)
    for (int h = 0; h < N_HEADS; h++) {
        const int head_off = h * D_HEAD;                 /* this head's column slice */
        float *hscores = scores + (size_t)h * T * T;     /* this head's (T x T) scratch */

        for (int i = 0; i < T; i++) {
            const float *qi   = Q + (size_t)i * D_MODEL + head_off;   /* (D_HEAD) */
            float       *srow = hscores + (size_t)i * T;              /* row of scores */

            /* --- 1) scaled dot-product scores for keys j = 0..i (causal) --- */
            float maxv = -1e30f;
            for (int j = 0; j <= i; j++) {
                const float *kj = K + (size_t)j * D_MODEL + head_off; /* (D_HEAD) */
                float dot = 0.0f;
                for (int d = 0; d < D_HEAD; d++)
                    dot += qi[d] * kj[d];
                dot *= scale;
                srow[j] = dot;
                if (dot > maxv) maxv = dot;
            }

            /* --- 2) numerically-stable softmax over j = 0..i --- */
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                float e = expf(srow[j] - maxv);
                srow[j] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j <= i; j++)
                srow[j] *= inv_sum;

            /* --- 3) weighted sum of value vectors --> out[i, head slice] --- */
            float *oi = out + (size_t)i * D_MODEL + head_off;         /* (D_HEAD) */
            for (int d = 0; d < D_HEAD; d++)
                oi[d] = 0.0f;
            for (int j = 0; j <= i; j++) {
                const float  p  = srow[j];
                const float *vj = V + (size_t)j * D_MODEL + head_off;
                for (int d = 0; d < D_HEAD; d++)
                    oi[d] += p * vj[d];
            }
        }
    }
}

/* ==========================================================================
 * forward_layer: one pre-norm transformer block, residual stream updated
 *                in place.
 *
 *   a  = LN1(x)
 *   x  = x + Attention(a)          <- residual 1
 *   b  = LN2(x)
 *   x  = x + FFN(b)                <- residual 2
 *
 * FFN is: (T x 1536) --W_ff1--> (T x 6144) --GELU--> --W_ff2--> (T x 1536).
 *
 * Note s->ln is reused for both layer-norm outputs: by the time we compute LN2
 * we have finished reading LN1's result (it was consumed by the QKV matmuls).
 * ========================================================================== */
void forward_layer(float *x, const LayerWeights *w, LayerScratch *s, int T)
{
    const float eps = 1e-5f;
    const int   N   = T * D_MODEL;         /* elements in one (T x D_MODEL) tensor */

    /* ---- 1. LayerNorm 1 ---- */
    layernorm(x, w->ln1_gamma, w->ln1_beta, s->ln, T, D_MODEL, eps);

    /* ---- 2. Q, K, V projections (each: matmul then add bias) ---- */
    matmul(s->ln, w->Wq, s->Q, T, D_MODEL, D_MODEL);  add_bias(s->Q, w->bq, T, D_MODEL);
    matmul(s->ln, w->Wk, s->K, T, D_MODEL, D_MODEL);  add_bias(s->K, w->bk, T, D_MODEL);
    matmul(s->ln, w->Wv, s->V, T, D_MODEL, D_MODEL);  add_bias(s->V, w->bv, T, D_MODEL);

    /* ---- 3. Causal multi-head self-attention ---- */
    multihead_attention(s->Q, s->K, s->V, s->attn, s->scores, T);

    /* ---- 4. Output projection ---- */
    matmul(s->attn, w->Wo, s->proj, T, D_MODEL, D_MODEL);
    add_bias(s->proj, w->bo, T, D_MODEL);

    /* ---- 5. Residual add 1:  x = x + attention_out ---- */
    add(x, s->proj, x, N);

    /* ---- 6. LayerNorm 2 ---- */
    layernorm(x, w->ln2_gamma, w->ln2_beta, s->ln, T, D_MODEL, eps);

    /* ---- 7. Feed-forward network: 1536 -> 6144 -> GELU -> 1536 ---- */
    matmul(s->ln, w->W_ff1, s->ff_hidden, T, D_MODEL, D_FF);
    add_bias(s->ff_hidden, w->b_ff1, T, D_FF);
    gelu(s->ff_hidden, s->ff_hidden, T * D_FF);                 /* in-place activation */
    matmul(s->ff_hidden, w->W_ff2, s->ff_out, T, D_FF, D_MODEL);
    add_bias(s->ff_out, w->b_ff2, T, D_MODEL);

    /* ---- 8. Residual add 2:  x = x + ffn_out ---- */
    add(x, s->ff_out, x, N);
}

/* ==========================================================================
 * DECODE PATH  --  single token + KV cache (used by generation).
 * ========================================================================== */

/*
 * proj_token: out(N) = in(K) @ W(K x N) + bias(N), for a SINGLE input vector.
 * --------------------------------------------------------------------------
 * The batch matmul parallelizes over output ROWS, but in decode we only have
 * one row, so that would leave 7 cores idle. Here we instead parallelize over
 * the output COLUMNS: we cut the N outputs into contiguous tiles and give each
 * tile to a thread. Inside a tile the k-loop streams one row of W (W[k, n0..n1])
 * and the out slice contiguously -- cache-friendly and vectorizable. This keeps
 * the memory-bound weight reads spread across all 8 threads.
 */
static void proj_token(float *out, const float *in, const float *W,
                       const float *bias, int K, int N)
{
    const int TILE   = 64;
    const int ntiles = (N + TILE - 1) / TILE;

#pragma omp parallel for schedule(static)
    for (int t = 0; t < ntiles; t++) {
        int n0 = t * TILE;
        int n1 = n0 + TILE; if (n1 > N) n1 = N;

        for (int n = n0; n < n1; n++)
            out[n] = bias ? bias[n] : 0.0f;

        for (int k = 0; k < K; k++) {
            const float  a  = in[k];
            const float *wr = W + (size_t)k * N;
            for (int n = n0; n < n1; n++)
                out[n] += a * wr[n];
        }
    }
}

/*
 * cached_attention: one query attends to cached keys/values 0..pos.
 * --------------------------------------------------------------------------
 * The causal mask is automatic: we only ever look at cached rows 0..pos, i.e.
 * this token and the past, never the future. Parallel over the 16 heads.
 */
static void cached_attention(const float *q, const float *kcache, const float *vcache,
                             float *out, float *scores, int pos)
{
    const float scale = 1.0f / sqrtf((float)D_HEAD);

#pragma omp parallel for schedule(static)
    for (int h = 0; h < N_HEADS; h++) {
        const int    off  = h * D_HEAD;
        const float *qh   = q + off;
        float       *srow = scores + (size_t)h * T_MAX;   /* this head's score row */

        /* 1) scaled dot-product against every cached key 0..pos */
        float maxv = -1e30f;
        for (int j = 0; j <= pos; j++) {
            const float *kj = kcache + (size_t)j * D_MODEL + off;
            float dot = 0.0f;
            for (int d = 0; d < D_HEAD; d++)
                dot += qh[d] * kj[d];
            dot *= scale;
            srow[j] = dot;
            if (dot > maxv) maxv = dot;
        }

        /* 2) stable softmax over 0..pos */
        float sum = 0.0f;
        for (int j = 0; j <= pos; j++) {
            float e = expf(srow[j] - maxv);
            srow[j] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int j = 0; j <= pos; j++)
            srow[j] *= inv;

        /* 3) weighted sum of cached values -> this head's output slice */
        float *oh = out + off;
        for (int d = 0; d < D_HEAD; d++)
            oh[d] = 0.0f;
        for (int j = 0; j <= pos; j++) {
            const float  p  = srow[j];
            const float *vj = vcache + (size_t)j * D_MODEL + off;
            for (int d = 0; d < D_HEAD; d++)
                oh[d] += p * vj[d];
        }
    }
}

/* One pre-norm block for a single token, KV-cached (see transformer.h). */
void forward_block(float *x, const LayerWeights *w, DecodeScratch *s,
                   float *kcache, float *vcache, int pos)
{
    const float eps = 1e-5f;

    /* 1. LN1 (single row) */
    layernorm(x, w->ln1_gamma, w->ln1_beta, s->ln, 1, D_MODEL, eps);

    /* 2. project Q into scratch; write K and V straight into the cache at pos */
    float *krow = kcache + (size_t)pos * D_MODEL;
    float *vrow = vcache + (size_t)pos * D_MODEL;
    proj_token(s->q, s->ln, w->Wq, w->bq, D_MODEL, D_MODEL);
    proj_token(krow, s->ln, w->Wk, w->bk, D_MODEL, D_MODEL);
    proj_token(vrow, s->ln, w->Wv, w->bv, D_MODEL, D_MODEL);

    /* 3. attention over cached K/V [0..pos] */
    cached_attention(s->q, kcache, vcache, s->attn, s->scores, pos);

    /* 4. output projection */
    proj_token(s->proj, s->attn, w->Wo, w->bo, D_MODEL, D_MODEL);

    /* 5. residual add 1 */
    add(x, s->proj, x, D_MODEL);

    /* 6. LN2 */
    layernorm(x, w->ln2_gamma, w->ln2_beta, s->ln, 1, D_MODEL, eps);

    /* 7. FFN: 1536 -> 6144 -> GELU -> 1536 */
    proj_token(s->ff_hidden, s->ln, w->W_ff1, w->b_ff1, D_MODEL, D_FF);
    gelu(s->ff_hidden, s->ff_hidden, D_FF);
    proj_token(s->ff_out, s->ff_hidden, w->W_ff2, w->b_ff2, D_FF, D_MODEL);

    /* 8. residual add 2 */
    add(x, s->ff_out, x, D_MODEL);
}

/* ==========================================================================
 * BENCHMARK  --  compile with -DTEST_TRANSFORMER for a standalone executable.
 * ========================================================================== */
#ifdef TEST_TRANSFORMER

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Small helper: malloc `n` floats or die. */
static float *falloc(size_t n)
{
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "out of memory (%zu floats)\n", n); exit(1); }
    return p;
}

/* Uniform random in [-scale, scale]. Weight init ~ GPT-2 style small values. */
static void randfill(float *p, size_t n, float scale)
{
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

static void alloc_weights(LayerWeights *w)
{
    size_t dd = (size_t)D_MODEL * D_MODEL;
    w->Wq = falloc(dd); w->Wk = falloc(dd); w->Wv = falloc(dd); w->Wo = falloc(dd);
    w->bq = falloc(D_MODEL); w->bk = falloc(D_MODEL);
    w->bv = falloc(D_MODEL); w->bo = falloc(D_MODEL);
    w->ln1_gamma = falloc(D_MODEL); w->ln1_beta = falloc(D_MODEL);
    w->ln2_gamma = falloc(D_MODEL); w->ln2_beta = falloc(D_MODEL);
    w->W_ff1 = falloc((size_t)D_MODEL * D_FF); w->b_ff1 = falloc(D_FF);
    w->W_ff2 = falloc((size_t)D_FF * D_MODEL); w->b_ff2 = falloc(D_MODEL);
}

static void init_weights(LayerWeights *w)
{
    size_t dd = (size_t)D_MODEL * D_MODEL;
    randfill(w->Wq, dd, 0.02f); randfill(w->Wk, dd, 0.02f);
    randfill(w->Wv, dd, 0.02f); randfill(w->Wo, dd, 0.02f);
    randfill(w->W_ff1, (size_t)D_MODEL * D_FF, 0.02f);
    randfill(w->W_ff2, (size_t)D_FF * D_MODEL, 0.02f);

    /* biases start at 0, layer-norm scale (gamma) at 1, shift (beta) at 0 */
    for (int i = 0; i < D_MODEL; i++) {
        w->bq[i] = w->bk[i] = w->bv[i] = w->bo[i] = 0.0f;
        w->ln1_gamma[i] = w->ln2_gamma[i] = 1.0f;
        w->ln1_beta[i]  = w->ln2_beta[i]  = 0.0f;
        w->b_ff2[i] = 0.0f;
    }
    for (int i = 0; i < D_FF; i++) w->b_ff1[i] = 0.0f;
}

static void alloc_scratch(LayerScratch *s, int T)
{
    s->ln    = falloc((size_t)T * D_MODEL);
    s->Q     = falloc((size_t)T * D_MODEL);
    s->K     = falloc((size_t)T * D_MODEL);
    s->V     = falloc((size_t)T * D_MODEL);
    s->attn  = falloc((size_t)T * D_MODEL);
    s->proj  = falloc((size_t)T * D_MODEL);
    s->scores    = falloc((size_t)N_HEADS * T * T);
    s->ff_hidden = falloc((size_t)T * D_FF);
    s->ff_out    = falloc((size_t)T * D_MODEL);
}

int main(void)
{
    const int T = T_MAX;   /* benchmark the full 512-token context */
    srand(1234);

    printf("=== Transformer block benchmark ===\n");
    printf("T=%d  d_model=%d  heads=%d  d_head=%d  d_ff=%d\n",
           T, D_MODEL, N_HEADS, D_HEAD, D_FF);
    printf("OpenMP threads = %d\n\n", omp_get_max_threads());

    LayerWeights w;  alloc_weights(&w);  init_weights(&w);
    LayerScratch s;  alloc_scratch(&s, T);

    /* residual stream input: (T x D_MODEL) small random activations */
    float *x = falloc((size_t)T * D_MODEL);
    randfill(x, (size_t)T * D_MODEL, 1.0f);

    /* Warm-up pass: fault in all the pages / warm the caches (not timed). */
    forward_layer(x, &w, &s, T);

    /* Re-init the residual stream so the timed run starts from clean data. */
    randfill(x, (size_t)T * D_MODEL, 1.0f);

    /* ---- the single timed forward pass ---- */
    double t0 = omp_get_wtime();
    forward_layer(x, &w, &s, T);
    double t1 = omp_get_wtime();

    double ms = (t1 - t0) * 1000.0;

    /* Approximate multiply-accumulates, to report throughput. */
    double macs =
          3.0 * T * D_MODEL * D_MODEL          /* Q, K, V projections   */
        + 1.0 * T * D_MODEL * D_MODEL          /* output projection     */
        + 2.0 * T * D_MODEL * D_FF             /* FFN two matmuls       */
        + 2.0 * N_HEADS * (0.5 * T * (T + 1)) * D_HEAD; /* attn (causal) */
    double gflops = (2.0 * macs / 1e9) / (ms / 1000.0);

    printf("Forward pass time : %.3f ms\n", ms);
    printf("Compute           : %.2f GFLOP\n", 2.0 * macs / 1e9);
    printf("Throughput        : %.1f GFLOP/s\n", gflops);
    /* print a value so the optimizer can't delete the whole computation */
    printf("x[0]=%.5f  x[last]=%.5f\n", x[0], x[(size_t)T * D_MODEL - 1]);

    return 0;
}

#endif /* TEST_TRANSFORMER */
