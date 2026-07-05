/*
 * model.h  --  Full GPT-2-style model: embeddings, a stack of transformer
 *              blocks, a final LayerNorm, and a weight-tied causal LM head.
 *
 * Piece 4 + Piece 5a. The forward pass is now KV-CACHED and processes ONE
 * token per call, which is what an autoregressive generation loop needs.
 *
 * --------------------------------------------------------------------------
 * WEIGHT TYING
 *   The token-embedding matrix wte (vocab_size x d_model) doubles as the LM
 *   head: logit(v) = dot(hidden, wte[v]). No separate output matrix.
 * --------------------------------------------------------------------------
 */
#ifndef MODEL_H
#define MODEL_H

#include "transformer.h"   /* LayerWeights, DecodeScratch, D_MODEL, N_HEADS, T_MAX ... */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- model-level dimensions ---- */
#define VOCAB_SIZE 2048       /* size of the token vocabulary          */
#define N_LAYERS   12         /* number of stacked transformer blocks  */
#define BLOCK_SIZE T_MAX      /* max context length (512)              */

/* All learned parameters of the model. */
typedef struct {
    float *wte;                     /* (VOCAB_SIZE x D_MODEL) token embeddings + tied head */
    float *wpe;                     /* (BLOCK_SIZE x D_MODEL) positional embeddings        */
    LayerWeights layers[N_LAYERS];  /* the 12 transformer blocks                           */
    float *lnf_gamma, *lnf_beta;    /* (D_MODEL) final LayerNorm scale / shift             */
} GPT2Weights;

/*
 * The Key/Value cache: for every layer, the K and V vectors of every position
 * seen so far. Pre-allocated for the maximum context (BLOCK_SIZE). This is the
 * state that makes generation linear instead of quadratic -- keys and values
 * are computed once per token and then reused on all subsequent steps.
 */
typedef struct {
    float *k[N_LAYERS];   /* each (BLOCK_SIZE x D_MODEL) */
    float *v[N_LAYERS];   /* each (BLOCK_SIZE x D_MODEL) */
} KVCache;

/* ---- KV cache lifecycle ---- */
void alloc_kv_cache(KVCache *c);
void free_kv_cache(KVCache *c);

/* ---- model lifecycle (dummy/random weights until we have trained ones) ---- */
void build_dummy_model(GPT2Weights *m);   /* allocate + random-initialize */
void free_model(GPT2Weights *m);

/* Tied causal LM head: logits(num_tokens x VOCAB_SIZE), parallel over vocab. */
void lm_head(float *logits, const float *x, const float *wte, int num_tokens);

/*
 * KV-cached forward pass for a SINGLE token.
 *   logits      : output (VOCAB_SIZE) next-token scores for this position
 *   token       : input token id, in [0, VOCAB_SIZE)
 *   weights     : model parameters
 *   cache       : KV cache (read for 0..current_pos-1, written at current_pos)
 *   current_pos : this token's position in the sequence (0-based, < BLOCK_SIZE)
 */
void forward_gpt2(float *logits, int token, GPT2Weights *weights,
                  KVCache *cache, int current_pos);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
