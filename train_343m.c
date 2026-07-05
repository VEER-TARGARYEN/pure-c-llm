/*
 * train.c  --  Piece 5b: the training engine (manual backprop + AdamW).
 *
 * A small "deep and narrow" verification model, trainable on a CPU in real time.
 * No autograd: every gradient is a hand-written chain-rule loop over raw flat
 * float* arrays, with OpenMP on the outer (batch / sequence / feature) loops.
 *
 * Build:  gcc -O3 -fopenmp train.c -o train.exe -lm
 * Run:    .\train.exe [num_steps]      (loads train.bin, or synthesizes data)
 *
 * --------------------------------------------------------------------------
 * FORWARD (pre-norm transformer), per layer:
 *     ln1 = LN1(x)
 *     q,k,v = ln1 @ Wq,Wk,Wv
 *     att = CausalAttention(q,k,v)
 *     x   = x + att @ Wo                (residual 1)
 *     ln2 = LN2(x)
 *     x   = x + (GELU(ln2 @ W1)) @ W2   (residual 2)
 * Then: lnf = LNf(x); logits = lnf @ wte^T   (weight-tied head).
 * --------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>   /* AVX-512 microkernel for the forward GEMMs */

/* ---- FULL model: the locked-in 343.9M-parameter configuration ---- */
#define BLOCK_SIZE 512
#define D_MODEL    1536
#define N_HEADS    16
#define D_HEAD     96     /* D_MODEL / N_HEADS */
#define D_FF       6144
#define N_LAYERS   12
#define VOCAB_SIZE 2048

/* ---- training shape (B=2 keeps params+Adam+activations ~7 GB in 16 GB RAM) ---- */
#define B_SIZE 2                       /* batch size                    */
#define T_SIZE BLOCK_SIZE              /* sequence length per sample    */
#define M_SIZE (B_SIZE * T_SIZE)       /* rows = batch*time, flattened  */

#define LN_EPS   1e-5f
#define ADAM_EPS 1e-8f

/* ==========================================================================
 * TINY HELPERS
 * ========================================================================== */
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM (%zu bytes)\n", n); exit(1); }
    return p;
}
static float *fzeros(int n) { float *p = (float *)xmalloc((size_t)n * sizeof(float)); memset(p, 0, (size_t)n * sizeof(float)); return p; }

/* xorshift32 for fast weight init */
static unsigned g_rng = 12345u;
static float frand(float scale) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    float u = (float)(g_rng & 0x00FFFFFFu) / (float)0x01000000;
    return (u * 2.0f - 1.0f) * scale;
}
static float *fmat(int n, float scale)  { float *p = (float *)xmalloc((size_t)n*sizeof(float)); for (int i=0;i<n;i++) p[i]=frand(scale); return p; }
static float *fones(int n)              { float *p = (float *)xmalloc((size_t)n*sizeof(float)); for (int i=0;i<n;i++) p[i]=1.0f; return p; }

static void add_vec(float *out, const float *a, const float *b, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* ==========================================================================
 * PARAMETERS + GRADIENTS + ADAM STATE
 * --------------------------------------------------------------------------
 * Tensors holds a full set of named pointers. We keep two: W (weights) and
 * G (gradients), with identical layout. A flat registry pairs each (w, dw)
 * with freshly-allocated Adam moment buffers (m, v) so the optimizer can loop
 * over every parameter uniformly.
 * ========================================================================== */
typedef struct {
    float *wte, *wpe;                                  /* embeddings          */
    float *g1[N_LAYERS], *b1[N_LAYERS];                /* LN1 gamma/beta      */
    float *Wq[N_LAYERS], *bq[N_LAYERS];
    float *Wk[N_LAYERS], *bk[N_LAYERS];
    float *Wv[N_LAYERS], *bv[N_LAYERS];
    float *Wo[N_LAYERS], *bo[N_LAYERS];
    float *g2[N_LAYERS], *b2[N_LAYERS];                /* LN2 gamma/beta      */
    float *W1[N_LAYERS], *bff1[N_LAYERS];              /* FFN up   (D->D_FF)  */
    float *W2[N_LAYERS], *bff2[N_LAYERS];              /* FFN down (D_FF->D)  */
    float *gf, *bf;                                    /* final LN gamma/beta */
} Tensors;

typedef struct { float *w, *dw, *m, *v; int n; } Param;

static Tensors W, G;
static Param   g_params[300];
static int     g_nparams = 0;

/* Register a weight `w` of length n: allocate its dw/m/v (zeroed) and return dw. */
static float *reg(float *w, int n) {
    float *dw = fzeros(n);
    g_params[g_nparams].w  = w;
    g_params[g_nparams].dw = dw;
    g_params[g_nparams].m  = fzeros(n);
    g_params[g_nparams].v  = fzeros(n);
    g_params[g_nparams].n  = n;
    g_nparams++;
    return dw;
}

static void build_params(void) {
    const int D = D_MODEL, DF = D_FF, V = VOCAB_SIZE, BLK = BLOCK_SIZE;

    W.wte = fmat(V*D, 0.02f);   G.wte = reg(W.wte, V*D);
    W.wpe = fmat(BLK*D, 0.02f); G.wpe = reg(W.wpe, BLK*D);

    for (int l = 0; l < N_LAYERS; l++) {
        W.g1[l]=fones(D);       G.g1[l]=reg(W.g1[l],D);
        W.b1[l]=fzeros(D);      G.b1[l]=reg(W.b1[l],D);
        W.Wq[l]=fmat(D*D,0.02f);G.Wq[l]=reg(W.Wq[l],D*D);
        W.bq[l]=fzeros(D);      G.bq[l]=reg(W.bq[l],D);
        W.Wk[l]=fmat(D*D,0.02f);G.Wk[l]=reg(W.Wk[l],D*D);
        W.bk[l]=fzeros(D);      G.bk[l]=reg(W.bk[l],D);
        W.Wv[l]=fmat(D*D,0.02f);G.Wv[l]=reg(W.Wv[l],D*D);
        W.bv[l]=fzeros(D);      G.bv[l]=reg(W.bv[l],D);
        W.Wo[l]=fmat(D*D,0.02f);G.Wo[l]=reg(W.Wo[l],D*D);
        W.bo[l]=fzeros(D);      G.bo[l]=reg(W.bo[l],D);
        W.g2[l]=fones(D);       G.g2[l]=reg(W.g2[l],D);
        W.b2[l]=fzeros(D);      G.b2[l]=reg(W.b2[l],D);
        W.W1[l]=fmat(D*DF,0.02f);G.W1[l]=reg(W.W1[l],D*DF);
        W.bff1[l]=fzeros(DF);   G.bff1[l]=reg(W.bff1[l],DF);
        W.W2[l]=fmat(DF*D,0.02f);G.W2[l]=reg(W.W2[l],DF*D);
        W.bff2[l]=fzeros(D);    G.bff2[l]=reg(W.bff2[l],D);
    }
    W.gf=fones(D);  G.gf=reg(W.gf,D);
    W.bf=fzeros(D); G.bf=reg(W.bf,D);
}

/* ==========================================================================
 * CORE FORWARD MATH (each also saves what its backward needs)
 * ========================================================================== */

#if defined(__AVX512F__)
/* register-blocked 8x16 microkernel: C[0:8, j0:j0+16] = A[0:8,0:K] @ B, in ZMM
 * regs (8 independent FMA chains, one B-load feeds 8). This is gemm_bench's v2
 * kernel, ~4x the plain i-k-j on cache-resident shapes. */
static inline void mm_micro8x16(const float *A, const float *B, float *C, int j0, int K, int N) {
    __m512 c0=_mm512_setzero_ps(),c1=_mm512_setzero_ps(),c2=_mm512_setzero_ps(),c3=_mm512_setzero_ps();
    __m512 c4=_mm512_setzero_ps(),c5=_mm512_setzero_ps(),c6=_mm512_setzero_ps(),c7=_mm512_setzero_ps();
    for (int k=0;k<K;k++){
        __m512 b=_mm512_loadu_ps(B+(size_t)k*N+j0);
        const float *a=A+k;
        c0=_mm512_fmadd_ps(_mm512_set1_ps(a[0*K]),b,c0); c1=_mm512_fmadd_ps(_mm512_set1_ps(a[1*K]),b,c1);
        c2=_mm512_fmadd_ps(_mm512_set1_ps(a[2*K]),b,c2); c3=_mm512_fmadd_ps(_mm512_set1_ps(a[3*K]),b,c3);
        c4=_mm512_fmadd_ps(_mm512_set1_ps(a[4*K]),b,c4); c5=_mm512_fmadd_ps(_mm512_set1_ps(a[5*K]),b,c5);
        c6=_mm512_fmadd_ps(_mm512_set1_ps(a[6*K]),b,c6); c7=_mm512_fmadd_ps(_mm512_set1_ps(a[7*K]),b,c7);
    }
    _mm512_storeu_ps(C+0*N+j0,c0);_mm512_storeu_ps(C+1*N+j0,c1);_mm512_storeu_ps(C+2*N+j0,c2);_mm512_storeu_ps(C+3*N+j0,c3);
    _mm512_storeu_ps(C+4*N+j0,c4);_mm512_storeu_ps(C+5*N+j0,c5);_mm512_storeu_ps(C+6*N+j0,c6);_mm512_storeu_ps(C+7*N+j0,c7);
}
#endif

/* out(M,N) = in(M,K) @ W(K,N) + bias(N).  Uses the AVX-512 microkernel on
 * tile-friendly dims (all of this model's forward GEMMs qualify: M=1024, N in
 * {256,1024}); falls back to i-k-j otherwise or when AVX-512 is unavailable. */
static void matmul_forward(float *out, const float *in, const float *Wt,
                           const float *bias, int M, int K, int N) {
#if defined(__AVX512F__)
    if ((M % 8 == 0) && (N % 16 == 0)) {
#pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < M; i0 += 8)
            for (int j0 = 0; j0 < N; j0 += 16)
                mm_micro8x16(in + (size_t)i0*K, Wt, out + (size_t)i0*N, j0, K, N);
    } else
#endif
    {
#pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float *o = out + (size_t)m * N;
            const float *x = in + (size_t)m * K;
            for (int n = 0; n < N; n++) o[n] = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = x[k];
                const float *wr = Wt + (size_t)k * N;
                for (int n = 0; n < N; n++) o[n] += a * wr[n];
            }
        }
    }
    if (bias) {
#pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float *o = out + (size_t)m * N;
            for (int n = 0; n < N; n++) o[n] += bias[n];
        }
    }
}

/*
 * matmul_backward for Y = X @ W (+b).  All three grads ACCUMULATE (+=):
 *   dX[m,k] += sum_n dY[m,n] * W[k,n]          (parallel over rows m)
 *   dW[k,n] += sum_m X[m,k] * dY[m,n]          (parallel over k, contiguous n)
 *   db[n]   += sum_m dY[m,n]
 */
static void matmul_backward(const float *dY, const float *X, const float *Wt,
                            float *dX, float *dW, float *db, int M, int K, int N) {
    /* dX */
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *dy = dY + (size_t)m * N;
        float *dx = dX + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            const float *wr = Wt + (size_t)k * N;
            float s = 0.0f;
            for (int n = 0; n < N; n++) s += dy[n] * wr[n];
            dx[k] += s;
        }
    }
    /* dW */
#pragma omp parallel for schedule(static)
    for (int k = 0; k < K; k++) {
        float *dwr = dW + (size_t)k * N;
        for (int m = 0; m < M; m++) {
            float a = X[(size_t)m * K + k];
            const float *dy = dY + (size_t)m * N;
            for (int n = 0; n < N; n++) dwr[n] += a * dy[n];
        }
    }
    /* db */
    if (db) {
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int m = 0; m < M; m++) s += dY[(size_t)m * N + n];
            db[n] += s;
        }
    }
}

/* LayerNorm over the last dim (D). Saves mean & rstd per row for the backward. */
static void layernorm_forward(float *out, const float *in, const float *gamma,
                              const float *beta, float *mean, float *rstd, int M, int D) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *x = in + (size_t)m * D;
        float *o = out + (size_t)m * D;
        float mu = 0.0f; for (int d = 0; d < D; d++) mu += x[d]; mu /= D;
        float var = 0.0f; for (int d = 0; d < D; d++) { float t = x[d]-mu; var += t*t; } var /= D;
        float rs = 1.0f / sqrtf(var + LN_EPS);
        mean[m] = mu; rstd[m] = rs;
        for (int d = 0; d < D; d++) o[d] = (x[d]-mu)*rs*gamma[d] + beta[d];
    }
}

/*
 * layernorm_backward. din ACCUMULATES (so the residual path can already be in
 * it); dgamma/dbeta accumulate. Uses the standard result:
 *   dxhat = dy*gamma
 *   dx = rstd*(dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
 */
static void layernorm_backward(float *din, float *dgamma, float *dbeta,
                               const float *dout, const float *in, const float *gamma,
                               const float *mean, const float *rstd, int M, int D) {
    /* din: independent per row -> parallel over m */
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *dy = dout + (size_t)m * D;
        const float *x  = in   + (size_t)m * D;
        float *dx = din + (size_t)m * D;
        float mu = mean[m], rs = rstd[m];
        float c1 = 0.0f, c2 = 0.0f;
        for (int d = 0; d < D; d++) {
            float xhat  = (x[d]-mu)*rs;
            float dxhat = dy[d]*gamma[d];
            c1 += dxhat;
            c2 += dxhat*xhat;
        }
        c1 /= D; c2 /= D;
        for (int d = 0; d < D; d++) {
            float xhat  = (x[d]-mu)*rs;
            float dxhat = dy[d]*gamma[d];
            dx[d] += rs*(dxhat - c1 - xhat*c2);
        }
    }
    /* dgamma/dbeta: reduce over rows -> parallel over feature d (race-free) */
#pragma omp parallel for schedule(static)
    for (int d = 0; d < D; d++) {
        float gg = 0.0f, bb = 0.0f;
        for (int m = 0; m < M; m++) {
            float xhat = (in[(size_t)m*D+d]-mean[m])*rstd[m];
            float dy   = dout[(size_t)m*D+d];
            gg += dy*xhat;
            bb += dy;
        }
        dgamma[d] += gg;
        dbeta[d]  += bb;
    }
}

/* GELU (tanh approx) forward and backward. */
static const float GELU_K = 0.7978845608028654f; /* sqrt(2/pi) */
static void gelu_forward(float *out, const float *in, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float x = in[i];
        float t = tanhf(GELU_K*(x + 0.044715f*x*x*x));
        out[i] = 0.5f*x*(1.0f+t);
    }
}
static void gelu_backward(float *din, const float *dout, const float *in, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float x = in[i];
        float u = GELU_K*(x + 0.044715f*x*x*x);
        float t = tanhf(u);
        float dudx = GELU_K*(1.0f + 3.0f*0.044715f*x*x);
        float dgelu = 0.5f*(1.0f+t) + 0.5f*x*(1.0f-t*t)*dudx;
        din[i] = dout[i]*dgelu;          /* SET */
    }
}

/* ==========================================================================
 * CAUSAL MULTI-HEAD ATTENTION (forward + backward), parallel over (batch,head)
 * ========================================================================== */
static void attention_forward(const float *q, const float *k, const float *v,
                              float *att, float *probs, int B, int T) {
    const float scale = 1.0f / sqrtf((float)D_HEAD);
#pragma omp parallel for schedule(static)
    for (int bh = 0; bh < B*N_HEADS; bh++) {
        int b = bh / N_HEADS, h = bh % N_HEADS, off = h*D_HEAD;
        for (int i = 0; i < T; i++) {
            const float *qi = q + ((size_t)(b*T+i)*D_MODEL) + off;
            float *p = probs + (((size_t)bh*T + i)*T);
            /* scores over keys j = 0..i (causal), track max */
            float maxv = -1e30f;
            for (int j = 0; j <= i; j++) {
                const float *kj = k + ((size_t)(b*T+j)*D_MODEL) + off;
                float dot = 0.0f; for (int d = 0; d < D_HEAD; d++) dot += qi[d]*kj[d];
                dot *= scale; p[j] = dot; if (dot > maxv) maxv = dot;
            }
            /* softmax */
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) { float e = expf(p[j]-maxv); p[j] = e; sum += e; }
            float inv = 1.0f/sum;
            for (int j = 0; j <= i; j++) p[j] *= inv;
            /* weighted sum of values */
            float *oi = att + ((size_t)(b*T+i)*D_MODEL) + off;
            for (int d = 0; d < D_HEAD; d++) oi[d] = 0.0f;
            for (int j = 0; j <= i; j++) {
                float pj = p[j];
                const float *vj = v + ((size_t)(b*T+j)*D_MODEL) + off;
                for (int d = 0; d < D_HEAD; d++) oi[d] += pj*vj[d];
            }
        }
    }
}

/* Given d_att, compute dq,dk,dv (all ACCUMULATE). probs,q,k,v are the saved forward. */
static void attention_backward(float *dq, float *dk, float *dv, const float *d_att,
                               const float *q, const float *k, const float *v,
                               const float *probs, int B, int T) {
    const float scale = 1.0f / sqrtf((float)D_HEAD);
#pragma omp parallel for schedule(static)
    for (int bh = 0; bh < B*N_HEADS; bh++) {
        int b = bh / N_HEADS, h = bh % N_HEADS, off = h*D_HEAD;
        float dprobs[T_SIZE];
        for (int i = 0; i < T; i++) {
            const float *dao = d_att + ((size_t)(b*T+i)*D_MODEL) + off;
            const float *p   = probs + (((size_t)bh*T + i)*T);

            /* d_probs[j] = dao . v_j ; and dv_j += p_j * dao */
            for (int j = 0; j <= i; j++) {
                const float *vj = v + ((size_t)(b*T+j)*D_MODEL) + off;
                float *dvj = dv + ((size_t)(b*T+j)*D_MODEL) + off;
                float dp = 0.0f;
                for (int d = 0; d < D_HEAD; d++) { dp += dao[d]*vj[d]; dvj[d] += p[j]*dao[d]; }
                dprobs[j] = dp;
            }
            /* softmax backward: ds_j = p_j*(dprobs_j - sum_k p_k dprobs_k) */
            float dot = 0.0f;
            for (int j = 0; j <= i; j++) dot += p[j]*dprobs[j];

            float *dqi = dq + ((size_t)(b*T+i)*D_MODEL) + off;
            const float *qi = q + ((size_t)(b*T+i)*D_MODEL) + off;
            for (int j = 0; j <= i; j++) {
                float ds = p[j]*(dprobs[j]-dot)*scale;
                const float *kj = k + ((size_t)(b*T+j)*D_MODEL) + off;
                float *dkj = dk + ((size_t)(b*T+j)*D_MODEL) + off;
                for (int d = 0; d < D_HEAD; d++) { dqi[d] += ds*kj[d]; dkj[d] += ds*qi[d]; }
            }
        }
    }
}

/* ==========================================================================
 * TIED HEAD (logits = lnf @ wte^T) forward + backward, and CROSS-ENTROPY.
 * ========================================================================== */
static void tied_head_forward(float *logits, const float *lnf, const float *wte, int M) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *x = lnf + (size_t)m*D_MODEL;
        float *lo = logits + (size_t)m*VOCAB_SIZE;
        for (int vv = 0; vv < VOCAB_SIZE; vv++) {
            const float *wr = wte + (size_t)vv*D_MODEL;
            float s = 0.0f; for (int d = 0; d < D_MODEL; d++) s += x[d]*wr[d];
            lo[vv] = s;
        }
    }
}
static void tied_head_backward(float *d_lnf, float *dwte, const float *d_logits,
                               const float *lnf, const float *wte, int M) {
    /* d_lnf[m,d] = sum_vv d_logits[m,vv]*wte[vv,d]   (SET) */
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *dl = d_logits + (size_t)m*VOCAB_SIZE;
        float *dx = d_lnf + (size_t)m*D_MODEL;
        for (int d = 0; d < D_MODEL; d++) dx[d] = 0.0f;
        for (int vv = 0; vv < VOCAB_SIZE; vv++) {
            float g = dl[vv]; const float *wr = wte + (size_t)vv*D_MODEL;
            for (int d = 0; d < D_MODEL; d++) dx[d] += g*wr[d];
        }
    }
    /* dwte[vv,d] += sum_m d_logits[m,vv]*lnf[m,d]   (ACCUMULATE, parallel over vv) */
#pragma omp parallel for schedule(static)
    for (int vv = 0; vv < VOCAB_SIZE; vv++) {
        float *dw = dwte + (size_t)vv*D_MODEL;
        for (int m = 0; m < M; m++) {
            float g = d_logits[(size_t)m*VOCAB_SIZE+vv];
            const float *lr = lnf + (size_t)m*D_MODEL;
            for (int d = 0; d < D_MODEL; d++) dw[d] += g*lr[d];
        }
    }
}

/* Softmax + cross-entropy. Returns mean loss; fills d_logits = (softmax-onehot)/M. */
static float cross_entropy(const float *logits, const int *targets, float *d_logits, int M) {
    double loss = 0.0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (int m = 0; m < M; m++) {
        const float *lo = logits + (size_t)m*VOCAB_SIZE;
        float *dl = d_logits + (size_t)m*VOCAB_SIZE;
        int tgt = targets[m];
        float maxl = lo[0]; for (int v = 1; v < VOCAB_SIZE; v++) if (lo[v] > maxl) maxl = lo[v];
        float sum = 0.0f; for (int v = 0; v < VOCAB_SIZE; v++) { float e = expf(lo[v]-maxl); dl[v] = e; sum += e; }
        float inv = 1.0f/sum;
        for (int v = 0; v < VOCAB_SIZE; v++) dl[v] *= inv;          /* dl = softmax prob */
        loss += -log((double)dl[tgt] + 1e-30);
        for (int v = 0; v < VOCAB_SIZE; v++) dl[v] = (dl[v] - (v==tgt?1.0f:0.0f)) / (float)M;
    }
    return (float)(loss / M);
}

/* ==========================================================================
 * ACTIVATIONS (saved forward values) + backward scratch. Allocated once.
 * ========================================================================== */
static float *A_xin[N_LAYERS+1];                     /* residual stream in/out per layer */
static float *A_ln1[N_LAYERS], *A_m1[N_LAYERS], *A_r1[N_LAYERS];
static float *A_q[N_LAYERS], *A_k[N_LAYERS], *A_v[N_LAYERS];
static float *A_att[N_LAYERS], *A_probs[N_LAYERS];
static float *A_xmid[N_LAYERS];
static float *A_ln2[N_LAYERS], *A_m2[N_LAYERS], *A_r2[N_LAYERS];
static float *A_h[N_LAYERS], *A_hg[N_LAYERS];
static float *A_lnf, *A_mf, *A_rf, *A_logits;
/* scratch */
static float *S_proj, *S_ff, *S_dx, *S_dlnf, *S_dln, *S_dq, *S_dk, *S_dv, *S_datt, *S_dh, *S_dhg, *S_dlogits;

static void alloc_acts(void) {
    int MD = M_SIZE*D_MODEL, MF = M_SIZE*D_FF, MV = M_SIZE*VOCAB_SIZE;
    int PB = B_SIZE*N_HEADS*T_SIZE*T_SIZE;
    for (int l = 0; l <= N_LAYERS; l++) A_xin[l] = fzeros(MD);
    for (int l = 0; l < N_LAYERS; l++) {
        A_ln1[l]=fzeros(MD); A_m1[l]=fzeros(M_SIZE); A_r1[l]=fzeros(M_SIZE);
        A_q[l]=fzeros(MD); A_k[l]=fzeros(MD); A_v[l]=fzeros(MD);
        A_att[l]=fzeros(MD); A_probs[l]=fzeros(PB);
        A_xmid[l]=fzeros(MD);
        A_ln2[l]=fzeros(MD); A_m2[l]=fzeros(M_SIZE); A_r2[l]=fzeros(M_SIZE);
        A_h[l]=fzeros(MF); A_hg[l]=fzeros(MF);
    }
    A_lnf=fzeros(MD); A_mf=fzeros(M_SIZE); A_rf=fzeros(M_SIZE); A_logits=fzeros(MV);
    S_proj=fzeros(MD); S_ff=fzeros(MD); S_dx=fzeros(MD); S_dlnf=fzeros(MD); S_dln=fzeros(MD);
    S_dq=fzeros(MD); S_dk=fzeros(MD); S_dv=fzeros(MD); S_datt=fzeros(MD);
    S_dh=fzeros(MF); S_dhg=fzeros(MF); S_dlogits=fzeros(MV);
}

/* ==========================================================================
 * FULL FORWARD PASS (fills activations, returns nothing; logits in A_logits)
 * ========================================================================== */
static void forward(const int *x_tokens) {
    const int M = M_SIZE, D = D_MODEL, DF = D_FF;

    /* embedding: x = wte[token] + wpe[pos] */
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        int t = m % T_SIZE, tok = x_tokens[m];
        const float *we = W.wte + (size_t)tok*D;
        const float *pe = W.wpe + (size_t)t*D;
        float *o = A_xin[0] + (size_t)m*D;
        for (int d = 0; d < D; d++) o[d] = we[d] + pe[d];
    }

    for (int l = 0; l < N_LAYERS; l++) {
        layernorm_forward(A_ln1[l], A_xin[l], W.g1[l], W.b1[l], A_m1[l], A_r1[l], M, D);
        matmul_forward(A_q[l], A_ln1[l], W.Wq[l], W.bq[l], M, D, D);
        matmul_forward(A_k[l], A_ln1[l], W.Wk[l], W.bk[l], M, D, D);
        matmul_forward(A_v[l], A_ln1[l], W.Wv[l], W.bv[l], M, D, D);
        attention_forward(A_q[l], A_k[l], A_v[l], A_att[l], A_probs[l], B_SIZE, T_SIZE);
        matmul_forward(S_proj, A_att[l], W.Wo[l], W.bo[l], M, D, D);
        add_vec(A_xmid[l], A_xin[l], S_proj, M*D);                 /* residual 1 */
        layernorm_forward(A_ln2[l], A_xmid[l], W.g2[l], W.b2[l], A_m2[l], A_r2[l], M, D);
        matmul_forward(A_h[l], A_ln2[l], W.W1[l], W.bff1[l], M, D, DF);
        gelu_forward(A_hg[l], A_h[l], M*DF);
        matmul_forward(S_ff, A_hg[l], W.W2[l], W.bff2[l], M, DF, D);
        add_vec(A_xin[l+1], A_xmid[l], S_ff, M*D);                 /* residual 2 */
    }

    layernorm_forward(A_lnf, A_xin[N_LAYERS], W.gf, W.bf, A_mf, A_rf, M, D);
    tied_head_forward(A_logits, A_lnf, W.wte, M);
}

/* ==========================================================================
 * FULL BACKWARD PASS. d_logits (S_dlogits) must already be filled by CE.
 * Accumulates all gradients into G. Assumes G was zeroed (AdamW zeros it).
 * ========================================================================== */
static void backward(const int *x_tokens) {
    const int M = M_SIZE, D = D_MODEL, DF = D_FF;

    /* tied head -> d_lnf, dwte */
    tied_head_backward(S_dlnf, G.wte, S_dlogits, A_lnf, W.wte, M);

    /* final LN -> dx (residual grad entering the top layer). Zero dx first. */
    memset(S_dx, 0, (size_t)M*D*sizeof(float));
    layernorm_backward(S_dx, G.gf, G.bf, S_dlnf, A_xin[N_LAYERS], W.gf, A_mf, A_rf, M, D);

    for (int l = N_LAYERS-1; l >= 0; l--) {
        /* ---- FFN backward (dx currently = grad of x_out[l]) ---- */
        memset(S_dhg, 0, (size_t)M*DF*sizeof(float));
        matmul_backward(S_dx, A_hg[l], W.W2[l], S_dhg, G.W2[l], G.bff2[l], M, DF, D);
        gelu_backward(S_dh, S_dhg, A_h[l], M*DF);
        memset(S_dln, 0, (size_t)M*D*sizeof(float));
        matmul_backward(S_dh, A_ln2[l], W.W1[l], S_dln, G.W1[l], G.bff1[l], M, D, DF);
        /* LN2: dx += contribution (now dx = grad of x_mid) */
        layernorm_backward(S_dx, G.g2[l], G.b2[l], S_dln, A_xmid[l], W.g2[l], A_m2[l], A_r2[l], M, D);

        /* ---- attention backward (dx currently = grad of x_mid) ---- */
        memset(S_datt, 0, (size_t)M*D*sizeof(float));
        matmul_backward(S_dx, A_att[l], W.Wo[l], S_datt, G.Wo[l], G.bo[l], M, D, D);
        memset(S_dq, 0, (size_t)M*D*sizeof(float));
        memset(S_dk, 0, (size_t)M*D*sizeof(float));
        memset(S_dv, 0, (size_t)M*D*sizeof(float));
        attention_backward(S_dq, S_dk, S_dv, S_datt, A_q[l], A_k[l], A_v[l], A_probs[l], B_SIZE, T_SIZE);
        /* q,k,v projections -> accumulate into d_ln1 */
        memset(S_dln, 0, (size_t)M*D*sizeof(float));
        matmul_backward(S_dq, A_ln1[l], W.Wq[l], S_dln, G.Wq[l], G.bq[l], M, D, D);
        matmul_backward(S_dk, A_ln1[l], W.Wk[l], S_dln, G.Wk[l], G.bk[l], M, D, D);
        matmul_backward(S_dv, A_ln1[l], W.Wv[l], S_dln, G.Wv[l], G.bv[l], M, D, D);
        /* LN1: dx += contribution (now dx = grad of x_in[l]) */
        layernorm_backward(S_dx, G.g1[l], G.b1[l], S_dln, A_xin[l], W.g1[l], A_m1[l], A_r1[l], M, D);
    }

    /* embedding backward (scatter-add; serial to avoid races on repeated tokens) */
    for (int m = 0; m < M; m++) {
        int t = m % T_SIZE, tok = x_tokens[m];
        const float *dxr = S_dx + (size_t)m*D;
        float *dwe = G.wte + (size_t)tok*D;
        float *dpe = G.wpe + (size_t)t*D;
        for (int d = 0; d < D; d++) { dwe[d] += dxr[d]; dpe[d] += dxr[d]; }
    }
}

/* ==========================================================================
 * ADAMW  --  multithreaded, decoupled weight decay; zeros dw afterward.
 * ========================================================================== */
static void adamw(int step, float lr, float beta1, float beta2, float wd) {
    float bc1 = 1.0f - powf(beta1, (float)step);   /* bias corrections */
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int p = 0; p < g_nparams; p++) {
        Param *P = &g_params[p];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < P->n; i++) {
            float g = P->dw[i];
            P->m[i] = beta1*P->m[i] + (1.0f-beta1)*g;
            P->v[i] = beta2*P->v[i] + (1.0f-beta2)*g*g;
            float mhat = P->m[i] / bc1;
            float vhat = P->v[i] / bc2;
            P->w[i] -= lr * (mhat / (sqrtf(vhat) + ADAM_EPS) + wd*P->w[i]);
            P->dw[i] = 0.0f;                        /* zero grad for next step */
        }
    }
}

/* ==========================================================================
 * DATA: load train.bin (Piece 1), or synthesize a learnable pattern.
 * ========================================================================== */
static int *load_or_synth(long *out_n) {
    FILE *f = fopen("train.bin", "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
        long ni = bytes / (long)sizeof(int);
        int *raw = (int *)xmalloc((size_t)ni * sizeof(int));
        if (fread(raw, sizeof(int), (size_t)ni, f) != (size_t)ni) { fprintf(stderr,"read error\n"); exit(1); }
        fclose(f);
        /* Piece 1 tokens.bin is [int count][ints]; detect and skip the header. */
        int *toks = raw; long n = ni;
        if (ni >= 1 && raw[0] == ni - 1) { toks = raw + 1; n = ni - 1; }
        for (long i = 0; i < n; i++) if (toks[i] < 0 || toks[i] >= VOCAB_SIZE) toks[i] = ((toks[i] % VOCAB_SIZE) + VOCAB_SIZE) % VOCAB_SIZE;
        printf("loaded train.bin: %ld tokens\n", n);
        *out_n = n; return toks;
    }
    /* synthetic: a period-64 cycle (token t is followed by t+1 mod 64) -- trivially
     * learnable, so the loss should visibly fall. Swap in a real train.bin anytime. */
    long n = 100000;
    int *d = (int *)xmalloc((size_t)n*sizeof(int));
    for (long i = 0; i < n; i++) d[i] = (int)(i % 64);
    printf("train.bin not found -> using synthetic period-64 data (%ld tokens)\n", n);
    *out_n = n; return d;
}

/* ==========================================================================
 * MAIN  --  training loop
 * ========================================================================== */
int main(int argc, char **argv) {
    int num_steps = (argc > 1) ? atoi(argv[1]) : 50;
    float lr = (argc > 2) ? (float)atof(argv[2]) : 1e-4f;   /* optional CLI override */
    const float beta1 = 0.9f, beta2 = 0.999f, wd = 0.01f;

    long n_tokens;
    int *data = load_or_synth(&n_tokens);
    if (n_tokens < T_SIZE + 1) { fprintf(stderr, "not enough tokens\n"); return 1; }

    build_params();
    alloc_acts();

    /* count parameters */
    long P = 0; for (int p = 0; p < g_nparams; p++) P += g_params[p].n;

    printf("model: %d layers, d_model=%d, d_ff=%d, heads=%d, vocab=%d\n",
           N_LAYERS, D_MODEL, D_FF, N_HEADS, VOCAB_SIZE);
    printf("params: %.2f M   |   batch=%d  seq=%d   |   threads=%d\n\n",
           P/1e6, B_SIZE, T_SIZE, omp_get_max_threads());

    int x[M_SIZE], y[M_SIZE];
    srand(1234);

    for (int step = 1; step <= num_steps; step++) {
        /* sample a batch of B contiguous windows: x = chunk[0:T], y = chunk[1:T+1] */
        for (int b = 0; b < B_SIZE; b++) {
            size_t s = ((size_t)rand()*32768u + (size_t)rand()) % (size_t)(n_tokens - T_SIZE - 1);
            for (int t = 0; t < T_SIZE; t++) {
                x[b*T_SIZE + t] = data[s + t];
                y[b*T_SIZE + t] = data[s + t + 1];
            }
        }

        double t0 = omp_get_wtime();
        forward(x);
        float loss = cross_entropy(A_logits, y, S_dlogits, M_SIZE);
        backward(x);
        adamw(step, lr, beta1, beta2, wd);
        double t1 = omp_get_wtime();

        double ms = (t1 - t0) * 1000.0;
        printf("step %4d | loss %7.4f | %7.1f ms/step | %6.0f tok/s\n",
               step, loss, ms, (M_SIZE) / (ms/1000.0));
    }

    printf("\ndone.\n");
    return 0;
}
