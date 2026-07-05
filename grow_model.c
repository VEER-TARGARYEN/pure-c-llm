/*
 * grow_model.c  --  Net2Net-style FUNCTION-PRESERVING model growth, pure C.
 *
 * Reads a checkpoint saved by train.c (small config) and writes a checkpoint
 * for a model with  d_model x2,  n_heads x2,  d_ff x2,  n_layers x2  that
 * computes EXACTLY the same function as the small model -- so fine-tuning
 * starts from the small model's loss instead of from random (~ln vocab).
 *
 * Reference: Chen, Goodfellow, Shlens -- "Net2Net: Accelerating Learning via
 * Knowledge Transfer" (2015); adapted here to a pre-norm, weight-tied GPT.
 *
 * --------------------------------------------------------------------------
 * THE RECIPE (and why it is exact)
 * --------------------------------------------------------------------------
 * Widening. The grown residual stream carries the small stream DUPLICATED and
 * HALVED:  x' = [x, x] / 2.
 *   - LayerNorm is invariant to both tricks (duplication keeps mean/variance,
 *     scaling cancels in x/std), so with duplicated gamma/beta every LN emits
 *     [LN(x), LN(x)] -- the un-scaled duplicated signal.
 *   - A linear layer y = aW reading a duplicated input [a,a] doubles every
 *     dot product, so each grown weight matrix is the original TILED 2x2 and
 *     SCALED: quadrant sums per output column must equal the original column.
 *       Wq, Wk, Wv, W_ff1 (read LN output, write duplicated outputs): tile W/2.
 *       Wo, W_ff2 (write residual deltas, which must ALSO be halved to match
 *       the stream convention x' = [.,.]/2): tile W/4, biases duplicated /2.
 *   - Heads: doubling head count with d_head unchanged makes the new heads
 *     exact copies (attention is computed per head, copies attend identically).
 *   - Tied head: logits = [f,f] . wte'[v]. Duplicating wte doubles the dot,
 *     so wte' = [wte, wte]/2 makes logits EXACTLY equal (and the same /2 is
 *     precisely the embedding scale the stream convention needs). wpe same.
 *   - Symmetry breaking: each tile gets +e (top) / -e (bottom) noise. The sum
 *     per column is unchanged -- the function stays EXACT -- but duplicated
 *     units are no longer identical, so gradients can pull them apart during
 *     fine-tuning (with exact duplicates they would move in lockstep forever).
 *
 * Deepening. A pre-norm block is x = x + F(LN(x)): forcing F's final matrix to
 * zero (Wo = 0 for attention, W_ff2 = 0 for the FFN) makes the block an exact
 * identity. We interleave: [grow(L0), identity, grow(L1), identity, ...]. The
 * identity blocks inherit the grown Q/K/V/W_ff1 weights as feature extractors;
 * their zeroed output matrices get non-zero gradients immediately, so the new
 * layers wake up during fine-tuning.
 *
 * Usage:  grow_model <in_ckpt> <out_ckpt>
 * Build:  gcc -O2 -Wall -Wextra grow_model.c -o grow_model.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CKPT_MAGIC 0x434B5031   /* must match train.c */
#define NOISE      0.002f       /* symmetry-breaking amplitude (cancels exactly) */

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }

/* xorshift32 noise in [-s, s] */
static unsigned g_rng = 20260701u;
static float nz(float s) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xFFFFFFu) / (float)0x1000000 * 2.0f - 1.0f) * s;
}

static float *rdmat(FILE *f, long n) {
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) die("out of memory");
    if (fread(p, sizeof(float), (size_t)n, f) != (size_t)n) die("short read");
    return p;
}

/* ---- emitters (all stream row-major to the output file) ------------------ */

/* vector v (len D)  ->  [v, v] * s   (len 2D) */
static void emit_vec_dup(FILE *fo, const float *v, int D, float s) {
    float *buf = (float *)malloc((size_t)2 * D * sizeof(float));
    for (int i = 0; i < D; i++) { buf[i] = v[i] * s; buf[D + i] = v[i] * s; }
    fwrite(buf, sizeof(float), (size_t)2 * D, fo);
    free(buf);
}

/* embedding rows: each row (len D) -> [row, row] * s; row count unchanged */
static void emit_rows_dupcols(FILE *fo, const float *M, int rows, int D, float s) {
    float *buf = (float *)malloc((size_t)2 * D * sizeof(float));
    for (int r = 0; r < rows; r++) {
        const float *row = M + (long)r * D;
        for (int i = 0; i < D; i++) { buf[i] = row[i] * s; buf[D + i] = row[i] * s; }
        fwrite(buf, sizeof(float), (size_t)2 * D, fo);
    }
    free(buf);
}

/* matrix W (in x out) -> (2in x 2out): every quadrant is W*s, with +e on the
 * top row-block and -e on the bottom so per-column sums stay exact. */
static void emit_mat_grow(FILE *fo, const float *W, int in, int out, float s) {
    long ncols = 2L * out;
    float *E   = (float *)malloc((size_t)in * ncols * sizeof(float));
    float *buf = (float *)malloc((size_t)ncols * sizeof(float));
    if (!E || !buf) die("out of memory");
    for (long i = 0; i < (long)in * ncols; i++) E[i] = nz(NOISE);
    for (int half = 0; half < 2; half++) {
        float sign = (half == 0) ? 1.0f : -1.0f;
        for (int r = 0; r < in; r++) {
            const float *wr = W + (long)r * out;
            const float *er = E + (long)r * ncols;
            for (long c = 0; c < ncols; c++)
                buf[c] = wr[c % out] * s + sign * er[c];
            fwrite(buf, sizeof(float), (size_t)ncols, fo);
        }
    }
    free(E); free(buf);
}

static void emit_zeros(FILE *fo, long n) {
    float *buf = (float *)calloc(4096, sizeof(float));
    while (n > 0) { long k = n > 4096 ? 4096 : n; fwrite(buf, sizeof(float), (size_t)k, fo); n -= k; }
    free(buf);
}

/* ---- one transformer layer's tensors, in train.c registry order ---------- */
typedef struct {
    float *g1,*b1,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*g2,*b2,*W1,*bff1,*W2,*bff2;
} Lay;

static Lay read_layer(FILE *f, int D, int DF) {
    Lay L;
    L.g1 = rdmat(f, D);            L.b1   = rdmat(f, D);
    L.Wq = rdmat(f, (long)D*D);    L.bq   = rdmat(f, D);
    L.Wk = rdmat(f, (long)D*D);    L.bk   = rdmat(f, D);
    L.Wv = rdmat(f, (long)D*D);    L.bv   = rdmat(f, D);
    L.Wo = rdmat(f, (long)D*D);    L.bo   = rdmat(f, D);
    L.g2 = rdmat(f, D);            L.b2   = rdmat(f, D);
    L.W1 = rdmat(f, (long)D*DF);   L.bff1 = rdmat(f, DF);
    L.W2 = rdmat(f, (long)DF*D);   L.bff2 = rdmat(f, D);
    return L;
}
static void free_layer(Lay *L) {
    free(L->g1); free(L->b1); free(L->Wq); free(L->bq); free(L->Wk); free(L->bk);
    free(L->Wv); free(L->bv); free(L->Wo); free(L->bo); free(L->g2); free(L->b2);
    free(L->W1); free(L->bff1); free(L->W2); free(L->bff2);
}

/* emit one grown layer; identity=1 zeroes the residual-writing matrices */
static void emit_layer(FILE *fo, const Lay *L, int D, int DF, int identity) {
    emit_vec_dup(fo, L->g1, D, 1.0f);          emit_vec_dup(fo, L->b1, D, 1.0f);
    emit_mat_grow(fo, L->Wq, D, D, 0.5f);      emit_vec_dup(fo, L->bq, D, 1.0f);
    emit_mat_grow(fo, L->Wk, D, D, 0.5f);      emit_vec_dup(fo, L->bk, D, 1.0f);
    emit_mat_grow(fo, L->Wv, D, D, 0.5f);      emit_vec_dup(fo, L->bv, D, 1.0f);
    if (identity) { emit_zeros(fo, 4L*D*D);    emit_zeros(fo, 2L*D); }
    else { emit_mat_grow(fo, L->Wo, D, D, 0.25f); emit_vec_dup(fo, L->bo, D, 0.5f); }
    emit_vec_dup(fo, L->g2, D, 1.0f);          emit_vec_dup(fo, L->b2, D, 1.0f);
    emit_mat_grow(fo, L->W1, D, DF, 0.5f);     emit_vec_dup(fo, L->bff1, DF, 1.0f);
    if (identity) { emit_zeros(fo, 4L*DF*D);   emit_zeros(fo, 2L*D); }
    else { emit_mat_grow(fo, L->W2, DF, D, 0.25f); emit_vec_dup(fo, L->bff2, D, 0.5f); }
}

static long params_of(int V, int BLK, int D, int DF, int L) {
    return (long)V*D + (long)BLK*D + 2L*D
         + (long)L * (4L*D*D + 2L*D*DF + 8L*D + DF + D);
}

int main(int argc, char **argv) {
    const char *in_path  = (argc > 1) ? argv[1] : "ckpt_small.bin";
    const char *out_path = (argc > 2) ? argv[2] : "ckpt_grown.bin";

    FILE *fi = fopen(in_path, "rb");
    if (!fi) die("cannot open input checkpoint");
    int h[8];
    if (fread(h, sizeof(int), 8, fi) != 8 || h[0] != CKPT_MAGIC) die("bad checkpoint");
    int BLK = h[1], D = h[2], H = h[3], DF = h[4], L = h[5], V = h[6];

    printf("source: d_model=%d heads=%d d_ff=%d layers=%d vocab=%d  (%.2fM params, step %d)\n",
           D, H, DF, L, V, params_of(V,BLK,D,DF,L)/1e6, h[7]);
    printf("target: d_model=%d heads=%d d_ff=%d layers=%d vocab=%d  (%.2fM params)\n",
           2*D, 2*H, 2*DF, 2*L, V, params_of(V,BLK,2*D,2*DF,2*L)/1e6);

    FILE *fo = fopen(out_path, "wb");
    if (!fo) die("cannot open output checkpoint");
    int ho[8] = { CKPT_MAGIC, BLK, 2*D, 2*H, 2*DF, 2*L, V, 0 };
    fwrite(ho, sizeof(int), 8, fo);

    /* embeddings: duplicate columns, halve (tied head then reproduces exact logits) */
    float *wte = rdmat(fi, (long)V * D);
    emit_rows_dupcols(fo, wte, V, D, 0.5f);  free(wte);
    float *wpe = rdmat(fi, (long)BLK * D);
    emit_rows_dupcols(fo, wpe, BLK, D, 0.5f); free(wpe);

    /* layers: grown original + identity block, interleaved */
    for (int l = 0; l < L; l++) {
        Lay Ly = read_layer(fi, D, DF);
        emit_layer(fo, &Ly, D, DF, 0);
        emit_layer(fo, &Ly, D, DF, 1);
        free_layer(&Ly);
        printf("  layer %d -> grown layer %d + identity layer %d\n", l, 2*l, 2*l+1);
    }

    /* final LayerNorm */
    float *gf = rdmat(fi, D), *bf = rdmat(fi, D);
    emit_vec_dup(fo, gf, D, 1.0f);
    emit_vec_dup(fo, bf, D, 1.0f);
    free(gf); free(bf);

    fclose(fi); fclose(fo);
    printf("wrote %s (function-preserving: grown model's loss == source model's loss)\n", out_path);
    return 0;
}
