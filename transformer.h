/*
 * transformer.h  --  A single pre-norm Transformer block for the C-LLM.
 *
 * Piece 3. Depends on Piece 2 (tensor.h) for the heavy math.
 *
 * --------------------------------------------------------------------------
 * ARCHITECTURE (locked-in maximum parameters)
 * --------------------------------------------------------------------------
 *   Context length      T        = 512   (runtime arg, must be <= T_MAX)
 *   Embedding dim       d_model  = 1536
 *   Heads               n_heads  = 16
 *   Dim per head        d_head   = 96    (16 * 96 = 1536)
 *   FFN hidden dim      d_ff     = 6144
 *
 * DATA LAYOUT
 *   Every activation is a flat, row-major float array. A (T x d_model) tensor
 *   stores token t's vector contiguously at offset t*d_model. Heads live side
 *   by side inside that vector: head h occupies columns [h*d_head, (h+1)*d_head).
 *
 * MEMORY CONTRACT (same as Piece 2)
 *   The caller owns ALL memory. The two structs below hold nothing but raw
 *   float* pointers -- they organize the ~18 weight/scratch arrays of a layer
 *   so the function signatures stay sane. Dimensions are still passed to every
 *   math kernel explicitly; nothing is hidden inside a struct.
 * --------------------------------------------------------------------------
 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- fixed architecture dimensions ---- */
#define T_MAX    512   /* maximum context length            */
#define D_MODEL  1536  /* embedding dimension               */
#define N_HEADS  16    /* number of attention heads         */
#define D_HEAD   96    /* dimension per head (D_MODEL/N_HEADS) */
#define D_FF     6144  /* feed-forward hidden dimension      */

/*
 * All the learned parameters of ONE transformer block. Every field is a flat
 * row-major array; the comment gives its logical shape.
 */
typedef struct {
    /* --- attention --- */
    float *Wq, *Wk, *Wv;        /* each (D_MODEL x D_MODEL): Q/K/V projections */
    float *bq, *bk, *bv;        /* each (D_MODEL): projection biases           */
    float *Wo;                  /* (D_MODEL x D_MODEL): output projection       */
    float *bo;                  /* (D_MODEL): output-projection bias            */

    /* --- layer norms --- */
    float *ln1_gamma, *ln1_beta; /* each (D_MODEL): pre-attention LN scale/shift */
    float *ln2_gamma, *ln2_beta; /* each (D_MODEL): pre-FFN LN scale/shift       */

    /* --- feed-forward network --- */
    float *W_ff1, *b_ff1;       /* (D_MODEL x D_FF), (D_FF): expand 1536 -> 6144 */
    float *W_ff2, *b_ff2;       /* (D_FF x D_MODEL), (D_MODEL): project 6144 -> 1536 */
} LayerWeights;

/*
 * Pre-allocated scratch space for one forward pass. Sizes assume the runtime
 * sequence length T (<= T_MAX). All are flat row-major float arrays.
 */
typedef struct {
    float *ln;         /* (T x D_MODEL)         : reused for LN1 and LN2 output */
    float *Q, *K, *V;  /* each (T x D_MODEL)    : projected queries/keys/values */
    float *attn;       /* (T x D_MODEL)         : concatenated per-head outputs  */
    float *proj;       /* (T x D_MODEL)         : attention after output proj    */
    float *scores;     /* (N_HEADS x T x T)     : per-head attention scores      */
    float *ff_hidden;  /* (T x D_FF)            : FFN hidden activations         */
    float *ff_out;     /* (T x D_MODEL)         : FFN output                     */
} LayerScratch;

/*
 * Causal multi-head self-attention.
 *   Q, K, V : (T x D_MODEL) inputs (already projected)
 *   out     : (T x D_MODEL) concatenated head outputs
 *   scores  : (N_HEADS x T x T) scratch, one (T x T) slab per head
 * The loop over the 16 heads is parallelized with OpenMP.
 */
void multihead_attention(const float *Q, const float *K, const float *V,
                         float *out, float *scores, int T);

/* --------------------------------------------------------------------------
 * DECODE PATH (KV-cache, one token at a time)
 * --------------------------------------------------------------------------
 * During generation we process a SINGLE new token per step. Its Q is tiny, and
 * its K/V are written once into a persistent cache and reused on every later
 * step -- that is what turns O(n^2) re-processing into O(n) per token.
 *
 * DecodeScratch holds the per-step temporaries for one token (all one row,
 * except `scores` which must span up to T_MAX past positions per head).
 */
typedef struct {
    float *ln;         /* (D_MODEL)          layer-norm output          */
    float *q;          /* (D_MODEL)          this token's query         */
    float *attn;       /* (D_MODEL)          attention output           */
    float *proj;       /* (D_MODEL)          attention output projection */
    float *scores;     /* (N_HEADS x T_MAX)  attention scores per head  */
    float *ff_hidden;  /* (D_FF)             FFN hidden activations      */
    float *ff_out;     /* (D_MODEL)          FFN output                 */
} DecodeScratch;

/*
 * One transformer block for a single token at position `pos`, using a KV cache.
 *   x       : (D_MODEL) residual vector for the current token, updated in place
 *   w       : this layer's weights
 *   s       : decode scratch
 *   kcache  : (T_MAX x D_MODEL) this layer's key cache
 *   vcache  : (T_MAX x D_MODEL) this layer's value cache
 *   pos     : current sequence position (0-based); K/V written at row `pos`,
 *             attention runs over cached rows 0..pos.
 *
 * (The block is handed its own layer's K/V slabs rather than the whole cache,
 * which keeps the cross-layer KVCache struct a model-level concern -- see
 * model.h -- and avoids a circular header dependency.)
 */
void forward_block(float *x, const LayerWeights *w, DecodeScratch *s,
                   float *kcache, float *vcache, int pos);

/*
 * One full pre-norm transformer block, updating the residual stream `x`
 * (T x D_MODEL) IN PLACE:
 *
 *   x += Attention(LN1(x))     (residual 1)
 *   x += FFN(LN2(x))           (residual 2)
 */
void forward_layer(float *x, const LayerWeights *w, LayerScratch *s, int T);

#ifdef __cplusplus
}
#endif

#endif /* TRANSFORMER_H */
