/*
 * model.c  --  Full GPT-2 forward pass, KV-cached, with a weight-tied LM head.
 *
 * Compiled as part of the generation program (see main.c):
 *   gcc -O3 -fopenmp -DTOKENIZER_LIB -Wno-unused-function \
 *       main.c model.c transformer.c tensor.c tokenizer.c -o llm.exe -lm
 */

#include "model.h"
#include "tensor.h"        /* layernorm */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>        /* size_t */

/* malloc-or-die, to keep call sites readable. */
static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(1); }
    return p;
}

/* ==========================================================================
 * KV cache lifecycle: one (BLOCK_SIZE x D_MODEL) K slab and V slab per layer.
 * ~75 MB total at these dims -- allocated once and reused across all tokens.
 * ========================================================================== */
void alloc_kv_cache(KVCache *c)
{
    for (int l = 0; l < N_LAYERS; l++) {
        c->k[l] = (float *)xmalloc((size_t)BLOCK_SIZE * D_MODEL * sizeof(float));
        c->v[l] = (float *)xmalloc((size_t)BLOCK_SIZE * D_MODEL * sizeof(float));
    }
}

void free_kv_cache(KVCache *c)
{
    for (int l = 0; l < N_LAYERS; l++) { free(c->k[l]); free(c->v[l]); }
}

/* ==========================================================================
 * Per-call decode scratch (one token's worth of temporaries). Tiny (~100 KB).
 * ========================================================================== */
static void alloc_decode_scratch(DecodeScratch *s)
{
    s->ln        = (float *)xmalloc((size_t)D_MODEL * sizeof(float));
    s->q         = (float *)xmalloc((size_t)D_MODEL * sizeof(float));
    s->attn      = (float *)xmalloc((size_t)D_MODEL * sizeof(float));
    s->proj      = (float *)xmalloc((size_t)D_MODEL * sizeof(float));
    s->scores    = (float *)xmalloc((size_t)N_HEADS * T_MAX * sizeof(float));
    s->ff_hidden = (float *)xmalloc((size_t)D_FF * sizeof(float));
    s->ff_out    = (float *)xmalloc((size_t)D_MODEL * sizeof(float));
}

static void free_decode_scratch(DecodeScratch *s)
{
    free(s->ln); free(s->q); free(s->attn); free(s->proj);
    free(s->scores); free(s->ff_hidden); free(s->ff_out);
}

/* ==========================================================================
 * lm_head: tied output projection, parallel over the VOCAB_SIZE rows of wte.
 *   logits[t, v] = dot( x[t, :] , wte[v, :] )
 * ========================================================================== */
void lm_head(float *logits, const float *x, const float *wte, int num_tokens)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float *wv = wte + (size_t)v * D_MODEL;
        for (int t = 0; t < num_tokens; t++) {
            const float *xt = x + (size_t)t * D_MODEL;
            float dot = 0.0f;
            for (int d = 0; d < D_MODEL; d++)
                dot += xt[d] * wv[d];
            logits[(size_t)t * VOCAB_SIZE + v] = dot;
        }
    }
}

/* ==========================================================================
 * forward_gpt2: KV-cached forward for a single token at `current_pos`.
 * ========================================================================== */
void forward_gpt2(float *logits, int token, GPT2Weights *weights,
                  KVCache *cache, int current_pos)
{
    const float eps = 1e-5f;

    float *x = (float *)xmalloc((size_t)D_MODEL * sizeof(float));  /* residual vector */
    DecodeScratch s;
    alloc_decode_scratch(&s);

    /* ---- 1. Embedding: x = token_embedding[token] + positional_embedding[pos] ---- */
    const float *tok = weights->wte + (size_t)token       * D_MODEL;
    const float *pos = weights->wpe + (size_t)current_pos * D_MODEL;
    for (int d = 0; d < D_MODEL; d++)
        x[d] = tok[d] + pos[d];

    /* ---- 2. Run through all 12 blocks, each reading/writing its cache slab ---- */
    for (int l = 0; l < N_LAYERS; l++)
        forward_block(x, &weights->layers[l], &s, cache->k[l], cache->v[l], current_pos);

    /* ---- 3. Final LayerNorm (single row, in place) ---- */
    layernorm(x, weights->lnf_gamma, weights->lnf_beta, x, 1, D_MODEL, eps);

    /* ---- 4. Tied LM head -> logits for this one position ---- */
    lm_head(logits, x, weights->wte, 1);

    free_decode_scratch(&s);
    free(x);
}

/* ==========================================================================
 * DUMMY WEIGHTS  --  allocate + randomly initialize the whole model.
 * (Real trained weights arrive in a later piece; for now generation runs on
 * random weights, so the output is well-formed but meaningless.)
 * ========================================================================== */

/* Fast xorshift32 -- init'ing ~344M weights with rand() would be needlessly slow. */
static unsigned int g_rng = 1234u;
static float frand(float scale)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    float u = (float)(g_rng & 0x00FFFFFFu) / (float)0x01000000; /* [0,1) */
    return (u * 2.0f - 1.0f) * scale;
}

static void randfill(float *p, size_t n, float scale)
{
    for (size_t i = 0; i < n; i++) p[i] = frand(scale);
}

static void alloc_layer(LayerWeights *w)
{
    size_t dd = (size_t)D_MODEL * D_MODEL;
    w->Wq = xmalloc(dd*sizeof(float)); w->Wk = xmalloc(dd*sizeof(float));
    w->Wv = xmalloc(dd*sizeof(float)); w->Wo = xmalloc(dd*sizeof(float));
    w->bq = xmalloc(D_MODEL*sizeof(float)); w->bk = xmalloc(D_MODEL*sizeof(float));
    w->bv = xmalloc(D_MODEL*sizeof(float)); w->bo = xmalloc(D_MODEL*sizeof(float));
    w->ln1_gamma = xmalloc(D_MODEL*sizeof(float)); w->ln1_beta = xmalloc(D_MODEL*sizeof(float));
    w->ln2_gamma = xmalloc(D_MODEL*sizeof(float)); w->ln2_beta = xmalloc(D_MODEL*sizeof(float));
    w->W_ff1 = xmalloc((size_t)D_MODEL*D_FF*sizeof(float)); w->b_ff1 = xmalloc(D_FF*sizeof(float));
    w->W_ff2 = xmalloc((size_t)D_FF*D_MODEL*sizeof(float)); w->b_ff2 = xmalloc(D_MODEL*sizeof(float));
}

static void init_layer(LayerWeights *w)
{
    size_t dd = (size_t)D_MODEL * D_MODEL;
    randfill(w->Wq, dd, 0.02f); randfill(w->Wk, dd, 0.02f);
    randfill(w->Wv, dd, 0.02f); randfill(w->Wo, dd, 0.02f);
    randfill(w->W_ff1, (size_t)D_MODEL*D_FF, 0.02f);
    randfill(w->W_ff2, (size_t)D_FF*D_MODEL, 0.02f);
    for (int i = 0; i < D_MODEL; i++) {
        w->bq[i]=w->bk[i]=w->bv[i]=w->bo[i]=0.0f;
        w->ln1_gamma[i]=w->ln2_gamma[i]=1.0f;
        w->ln1_beta[i]=w->ln2_beta[i]=0.0f;
        w->b_ff2[i]=0.0f;
    }
    for (int i = 0; i < D_FF; i++) w->b_ff1[i] = 0.0f;
}

static void free_layer(LayerWeights *w)
{
    free(w->Wq); free(w->Wk); free(w->Wv); free(w->Wo);
    free(w->bq); free(w->bk); free(w->bv); free(w->bo);
    free(w->ln1_gamma); free(w->ln1_beta); free(w->ln2_gamma); free(w->ln2_beta);
    free(w->W_ff1); free(w->b_ff1); free(w->W_ff2); free(w->b_ff2);
}

void build_dummy_model(GPT2Weights *m)
{
    m->wte = xmalloc((size_t)VOCAB_SIZE * D_MODEL * sizeof(float));
    m->wpe = xmalloc((size_t)BLOCK_SIZE * D_MODEL * sizeof(float));
    m->lnf_gamma = xmalloc(D_MODEL*sizeof(float));
    m->lnf_beta  = xmalloc(D_MODEL*sizeof(float));

    randfill(m->wte, (size_t)VOCAB_SIZE * D_MODEL, 0.02f);
    randfill(m->wpe, (size_t)BLOCK_SIZE * D_MODEL, 0.02f);
    for (int i = 0; i < D_MODEL; i++) { m->lnf_gamma[i] = 1.0f; m->lnf_beta[i] = 0.0f; }

    for (int l = 0; l < N_LAYERS; l++) { alloc_layer(&m->layers[l]); init_layer(&m->layers[l]); }
}

void free_model(GPT2Weights *m)
{
    free(m->wte); free(m->wpe); free(m->lnf_gamma); free(m->lnf_beta);
    for (int l = 0; l < N_LAYERS; l++) free_layer(&m->layers[l]);
}
