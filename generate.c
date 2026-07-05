/*
 * generate.c  --  Text generation from a train.c checkpoint. Pure C.
 *
 * Loads ANY CKP1 checkpoint (dims are read from the header at runtime, so the
 * same binary serves the 6.9M, the grown 52M, or any future model), plus a
 * Piece-1 BPE vocab for encode/decode, and streams a completion.
 *
 * Build:
 *   gcc -O3 -march=native -ffast-math -fopenmp -DTOKENIZER_LIB \
 *       generate.c tokenizer.c -o generate.exe -lm
 * Run:
 *   .\generate.exe <ckpt.bin> <vocab.bin> "<prompt>" [n_tokens] [temp] [top_k]
 *   .\generate.exe ckpt_grown_ft2.bin vocab_shk.bin "ROMEO:" 100 0.8 40
 *
 * KV-cached single-token decode (linear in sequence length), OpenMP on the
 * matrix-vector products and the attention heads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "tokenizer.h"

#define CKPT_MAGIC 0x434B5031

/* ---- model dims, read from the checkpoint header at runtime ---- */
static int BLK, D, H, DF, L, V, DH;

typedef struct {
    float *g1,*b1,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*g2,*b2,*W1,*bff1,*W2,*bff2;
} Lay;

static float *g_wte, *g_wpe, *g_gf, *g_bf;
static Lay   *g_lay;

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }

static float *rd(FILE *f, long n) {
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) die("out of memory");
    if (fread(p, sizeof(float), (size_t)n, f) != (size_t)n) die("short read in checkpoint");
    return p;
}

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open checkpoint");
    int h[8];
    if (fread(h, sizeof(int), 8, f) != 8 || h[0] != CKPT_MAGIC) die("not a CKP1 checkpoint");
    BLK = h[1]; D = h[2]; H = h[3]; DF = h[4]; L = h[5]; V = h[6];
    DH = D / H;
    fprintf(stderr, "model: d_model=%d heads=%d d_ff=%d layers=%d vocab=%d block=%d (step %d)\n",
            D, H, DF, L, V, BLK, h[7]);

    g_wte = rd(f, (long)V * D);
    g_wpe = rd(f, (long)BLK * D);
    g_lay = (Lay *)malloc((size_t)L * sizeof(Lay));
    for (int l = 0; l < L; l++) {
        Lay *y = &g_lay[l];
        y->g1 = rd(f, D);           y->b1   = rd(f, D);
        y->Wq = rd(f, (long)D*D);   y->bq   = rd(f, D);
        y->Wk = rd(f, (long)D*D);   y->bk   = rd(f, D);
        y->Wv = rd(f, (long)D*D);   y->bv   = rd(f, D);
        y->Wo = rd(f, (long)D*D);   y->bo   = rd(f, D);
        y->g2 = rd(f, D);           y->b2   = rd(f, D);
        y->W1 = rd(f, (long)D*DF);  y->bff1 = rd(f, DF);
        y->W2 = rd(f, (long)DF*D);  y->bff2 = rd(f, D);
    }
    g_gf = rd(f, D);
    g_bf = rd(f, D);
    fclose(f);
}

/* ---- KV cache + per-token scratch ---- */
static float **g_kc, **g_vc;                      /* per layer: (BLK x D) */
static float *s_x, *s_ln, *s_q, *s_att, *s_proj, *s_h, *s_scores, *s_logits;

static void alloc_state(void) {
    g_kc = (float **)malloc((size_t)L * sizeof(float *));
    g_vc = (float **)malloc((size_t)L * sizeof(float *));
    for (int l = 0; l < L; l++) {
        g_kc[l] = (float *)malloc((size_t)BLK * D * sizeof(float));
        g_vc[l] = (float *)malloc((size_t)BLK * D * sizeof(float));
    }
    s_x = malloc((size_t)D*4);  s_ln = malloc((size_t)D*4);  s_q = malloc((size_t)D*4);
    s_att = malloc((size_t)D*4); s_proj = malloc((size_t)D*4); s_h = malloc((size_t)DF*4);
    s_scores = malloc((size_t)H*BLK*4); s_logits = malloc((size_t)V*4);
}

/* out(N) = x(K) @ W(K,N) + b(N), parallel over column tiles */
static void matvec(float *out, const float *x, const float *W, const float *b, int K, int N) {
    const int TILE = 64, nt = (N + TILE - 1) / TILE;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int n0 = t*TILE, n1 = n0+TILE > N ? N : n0+TILE;
        for (int n = n0; n < n1; n++) out[n] = b ? b[n] : 0.0f;
        for (int k = 0; k < K; k++) {
            float a = x[k];
            const float *wr = W + (size_t)k * N;
            for (int n = n0; n < n1; n++) out[n] += a * wr[n];
        }
    }
}

static void ln_row(const float *x, const float *g, const float *b, float *y, int n) {
    float mu = 0, var = 0;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; i++) { float d = x[i]-mu; var += d*d; }
    var /= n;
    float rs = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) y[i] = (x[i]-mu)*rs*g[i] + b[i];
}

/* one pre-norm block for the token at position `pos`, KV-cached */
static void block(int l, int pos) {
    Lay *w = &g_lay[l];
    float *krow = g_kc[l] + (size_t)pos * D;
    float *vrow = g_vc[l] + (size_t)pos * D;

    ln_row(s_x, w->g1, w->b1, s_ln, D);
    matvec(s_q, s_ln, w->Wq, w->bq, D, D);
    matvec(krow, s_ln, w->Wk, w->bk, D, D);
    matvec(vrow, s_ln, w->Wv, w->bv, D, D);

    const float scale = 1.0f / sqrtf((float)DH);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        const int off = h * DH;
        float *sr = s_scores + (size_t)h * BLK;
        float maxv = -1e30f;
        for (int j = 0; j <= pos; j++) {
            const float *kj = g_kc[l] + (size_t)j*D + off;
            float dot = 0;
            for (int d = 0; d < DH; d++) dot += s_q[off+d] * kj[d];
            dot *= scale; sr[j] = dot; if (dot > maxv) maxv = dot;
        }
        float sum = 0;
        for (int j = 0; j <= pos; j++) { float e = expf(sr[j]-maxv); sr[j] = e; sum += e; }
        float inv = 1.0f/sum;
        float *oh = s_att + off;
        for (int d = 0; d < DH; d++) oh[d] = 0;
        for (int j = 0; j <= pos; j++) {
            float p = sr[j]*inv;
            const float *vj = g_vc[l] + (size_t)j*D + off;
            for (int d = 0; d < DH; d++) oh[d] += p * vj[d];
        }
    }

    matvec(s_proj, s_att, w->Wo, w->bo, D, D);
    for (int i = 0; i < D; i++) s_x[i] += s_proj[i];

    ln_row(s_x, w->g2, w->b2, s_ln, D);
    matvec(s_h, s_ln, w->W1, w->bff1, D, DF);
    const float kg = 0.7978845608028654f;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < DF; i++) {
        float v = s_h[i];
        s_h[i] = 0.5f*v*(1.0f + tanhf(kg*(v + 0.044715f*v*v*v)));
    }
    matvec(s_proj, s_h, w->W2, w->bff2, DF, D);
    for (int i = 0; i < D; i++) s_x[i] += s_proj[i];
}

/* full forward for one token; fills s_logits */
static void forward(int tok, int pos) {
    const float *te = g_wte + (size_t)tok * D;
    const float *pe = g_wpe + (size_t)pos * D;
    for (int i = 0; i < D; i++) s_x[i] = te[i] + pe[i];
    for (int l = 0; l < L; l++) block(l, pos);
    ln_row(s_x, g_gf, g_bf, s_ln, D);
#pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        const float *wv = g_wte + (size_t)v * D;
        float dot = 0;
        for (int d = 0; d < D; d++) dot += s_ln[d] * wv[d];
        s_logits[v] = dot;
    }
}

/* temperature + top-k sampling (temp <= 0 -> greedy argmax) */
static int sample(const float *logits, int n, float temp, int topk) {
    if (temp <= 1e-6f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float *p = (float *)malloc((size_t)n * sizeof(float));
    float maxl = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxl) maxl = logits[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { p[i] = expf((logits[i]-maxl)/temp); sum += p[i]; }
    for (int i = 0; i < n; i++) p[i] = (float)(p[i]/sum);

    if (topk > 0 && topk < n) {
        /* zero everything below the k-th largest prob (k is small: O(kN)) */
        float kth = 0.0f;
        float *tmp = (float *)malloc((size_t)n * sizeof(float));
        memcpy(tmp, p, (size_t)n * sizeof(float));
        for (int r = 0; r < topk; r++) {
            int bi = 0;
            for (int i = 1; i < n; i++) if (tmp[i] > tmp[bi]) bi = i;
            kth = tmp[bi]; tmp[bi] = -1.0f;
        }
        free(tmp);
        double ksum = 0;
        for (int i = 0; i < n; i++) { if (p[i] < kth) p[i] = 0; ksum += p[i]; }
        for (int i = 0; i < n; i++) p[i] = (float)(p[i]/ksum);
    }

    double r = (double)rand() / ((double)RAND_MAX + 1.0), c = 0;
    int pick = n - 1;
    for (int i = 0; i < n; i++) { c += p[i]; if (r < c) { pick = i; break; } }
    free(p);
    return pick;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: generate <ckpt.bin> <vocab.bin> \"<prompt>\" [n_tokens] [temp] [top_k]\n");
        return 1;
    }
    const char *ckpt_path  = argv[1];
    const char *vocab_path = argv[2];
    const char *prompt     = argv[3];
    int   n_new = (argc > 4) ? atoi(argv[4])        : 100;
    float temp  = (argc > 5) ? (float)atof(argv[5]) : 0.8f;
    int   topk  = (argc > 6) ? atoi(argv[6])        : 40;

    load_model(ckpt_path);
    alloc_state();

    int tok_vocab, num_merges;
    Merge *merges = load_vocab(vocab_path, &tok_vocab, &num_merges);
    unsigned char **tb; int *tl;
    build_token_bytes(tok_vocab, merges, num_merges, &tb, &tl);
    int sample_v = tok_vocab < V ? tok_vocab : V;

    srand((unsigned)time(NULL));

    /* encode + feed the prompt (filling the KV cache) */
    int n_ids;
    int *ids = encode((const unsigned char *)prompt, (long)strlen(prompt), merges, num_merges, &n_ids);
    if (n_ids == 0) die("empty prompt");
    fprintf(stderr, "prompt: %d tokens\n\n", n_ids);

    printf("%s", prompt);
    fflush(stdout);

    int pos = 0;
    for (int i = 0; i < n_ids && pos < BLK; i++, pos++)
        forward(ids[i] < V ? ids[i] : 0, pos);
    free(ids);

    /* autoregressive generation, streamed */
    double t0 = omp_get_wtime();
    int made = 0;
    for (int s = 0; s < n_new && pos < BLK; s++, pos++, made++) {
        int next = sample(s_logits, sample_v, temp, topk);
        fwrite(tb[next], 1, (size_t)tl[next], stdout);
        fflush(stdout);
        forward(next, pos);
    }
    double dt = omp_get_wtime() - t0;
    printf("\n");
    fprintf(stderr, "\n[%d tokens in %.1f s, %.1f tok/s]\n", made, dt, made/dt);

    free_token_bytes(tb, tl, tok_vocab);
    free(merges);
    return 0;
}
