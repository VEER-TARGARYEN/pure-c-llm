/*
 * gpu_kernels.cu  --  CUDA kernels for the 343M trainer. See gpu_kernels.h.
 *
 * Design notes:
 *  - GEMMs use 16x16 shared-memory tiling with +1 padding (no bank conflicts)
 *    and coalesced global loads in all three transpose variants.
 *  - Attention runs one block per (batch*head, position): 128 threads = d_head.
 *    The causal mask is by construction (loops stop at j <= i).
 *  - Reductions (LayerNorm, softmax, CE) are classic shared-memory tree
 *    reductions, one block per row.
 *  - Everything is FP32. Tensor-core FP16 is the next 4x lever, not v1.
 */

#include "gpu_kernels.h"
#include <math.h>
#include <cublas_v2.h>

#define TS 16   /* GEMM tile size */

static cublasHandle_t g_cublas = NULL;
void gpu_init(void) {
    if (!g_cublas && cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n"); exit(1);
    }
}

/* broadcast bias over rows: C[m][n] += bias[n] */
__global__ void k_bias_add(float *C, const float* __restrict__ bias, long total, int N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) C[i] += bias[i % N];
}

/* ========================== GEMM family ========================== */

__global__ void k_gemm_nn(const float* __restrict__ A, const float* __restrict__ B,
                          const float* __restrict__ bias, float* __restrict__ C,
                          int M, int K, int N) {
    __shared__ float As[TS][TS + 1], Bs[TS][TS + 1];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TS + ty;
    int col = blockIdx.x * TS + tx;
    float acc = 0.0f;
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int ak = t * TS + tx, bk = t * TS + ty;
        As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TS; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[(size_t)row * N + col] = acc + (bias ? bias[col] : 0.0f);
}

/* C(M,N) ?= A(M,K) @ B(N,K)^T : dX and tied-head logits */
__global__ void k_gemm_nt(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int K, int N, int accum) {
    __shared__ float As[TS][TS + 1], Bs[TS][TS + 1];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TS + ty;   /* over M */
    int col = blockIdx.x * TS + tx;   /* over N */
    float acc = 0.0f;
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int ak = t * TS + tx;
        int brow = blockIdx.x * TS + ty, bk = t * TS + tx;
        As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.0f;
        Bs[ty][tx] = (brow < N && bk < K) ? B[(size_t)brow * K + bk] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TS; k++) acc += As[ty][k] * Bs[tx][k];
        __syncthreads();
    }
    if (row < M && col < N) {
        size_t idx = (size_t)row * N + col;
        C[idx] = (accum ? C[idx] : 0.0f) + acc;
    }
}

/* C(K,N) += A(M,K)^T @ B(M,N) : weight gradients */
__global__ void k_gemm_tn(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TS][TS + 1], Bs[TS][TS + 1];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TS + ty;   /* over K */
    int col = blockIdx.x * TS + tx;   /* over N */
    float acc = 0.0f;
    for (int t = 0; t < (M + TS - 1) / TS; t++) {
        int am = t * TS + ty, ak = blockIdx.y * TS + tx;
        As[ty][tx] = (am < M && ak < K) ? A[(size_t)am * K + ak] : 0.0f;  /* As[m][k] */
        int bm = t * TS + ty;
        Bs[ty][tx] = (bm < M && col < N) ? B[(size_t)bm * N + col] : 0.0f; /* Bs[m][n] */
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TS; k++) acc += As[k][ty] * Bs[k][tx];
        __syncthreads();
    }
    if (row < K && col < N)
        C[(size_t)row * N + col] += acc;
}

__global__ void k_bias_bwd(float *db, const float* __restrict__ dY, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.0f;
    for (int m = 0; m < M; m++) s += dY[(size_t)m * N + n];
    db[n] += s;
}

/* Row-major GEMMs via cuBLAS: with everything row-major, compute the
 * column-major transpose identity  C_cm(N,M) = op(B) . op(A). */
void gemm_nn(const float *A, const float *B, const float *bias, float *C, int M, int K, int N) {
#ifdef USE_HAND_GEMM
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS), blk(TS, TS);
    k_gemm_nn<<<grid, blk>>>(A, B, bias, C, M, K, N);
#else
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);
    if (bias) k_bias_add<<<(unsigned)(((long)M * N + 255) / 256), 256>>>(C, bias, (long)M * N, N);
#endif
}
void gemm_nt(const float *A, const float *B, float *C, int M, int K, int N, int accum) {
#ifdef USE_HAND_GEMM
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS), blk(TS, TS);
    k_gemm_nt<<<grid, blk>>>(A, B, C, M, K, N, accum);
#else
    const float one = 1.0f, beta = accum ? 1.0f : 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, B, K, A, K, &beta, C, N);
#endif
}
void gemm_tn(const float *A, const float *B, float *C, int M, int K, int N) {
#ifdef USE_HAND_GEMM
    dim3 grid((N + TS - 1) / TS, (K + TS - 1) / TS), blk(TS, TS);
    k_gemm_tn<<<grid, blk>>>(A, B, C, M, K, N);
#else
    const float one = 1.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &one, B, N, A, K, &one, C, N);
#endif
}
void bias_bwd(float *db, const float *dY, int M, int N) {
    k_bias_bwd<<<(N + 255) / 256, 256>>>(db, dY, M, N);
}

/* ========================== LayerNorm ========================== */

__global__ void k_ln_fwd(const float* __restrict__ X, const float* __restrict__ g,
                         const float* __restrict__ b, float* __restrict__ Y,
                         float *mean, float *rstd) {
    int m = blockIdx.x, tid = threadIdx.x;
    const float *x = X + (size_t)m * DIM;
    float *y = Y + (size_t)m * DIM;
    __shared__ float red[256];

    float s = 0.0f;
    for (int i = tid; i < DIM; i += blockDim.x) s += x[i];
    red[tid] = s; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float mu = red[0] / DIM; __syncthreads();

    float v = 0.0f;
    for (int i = tid; i < DIM; i += blockDim.x) { float d = x[i] - mu; v += d * d; }
    red[tid] = v; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float rs = rsqrtf(red[0] / DIM + LN_EPS);

    if (tid == 0) { mean[m] = mu; rstd[m] = rs; }
    for (int i = tid; i < DIM; i += blockDim.x)
        y[i] = (x[i] - mu) * rs * g[i] + b[i];
}

__global__ void k_ln_bwd_dx(float* __restrict__ dX, const float* __restrict__ dY,
                            const float* __restrict__ X, const float* __restrict__ g,
                            const float *mean, const float *rstd) {
    int m = blockIdx.x, tid = threadIdx.x;
    const float *dy = dY + (size_t)m * DIM, *x = X + (size_t)m * DIM;
    float *dx = dX + (size_t)m * DIM;
    float mu = mean[m], rs = rstd[m];
    __shared__ float r1[256], r2[256];

    float a = 0.0f, c = 0.0f;
    for (int i = tid; i < DIM; i += blockDim.x) {
        float xh = (x[i] - mu) * rs, dxh = dy[i] * g[i];
        a += dxh; c += dxh * xh;
    }
    r1[tid] = a; r2[tid] = c; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) { r1[tid] += r1[tid + st]; r2[tid] += r2[tid + st]; }
        __syncthreads();
    }
    float c1 = r1[0] / DIM, c2 = r2[0] / DIM;
    for (int i = tid; i < DIM; i += blockDim.x) {
        float xh = (x[i] - mu) * rs, dxh = dy[i] * g[i];
        dx[i] += rs * (dxh - c1 - xh * c2);          /* ACCUMULATES, like train.c */
    }
}

__global__ void k_ln_bwd_dgb(float *dg, float *db, const float* __restrict__ dY,
                             const float* __restrict__ X, const float *mean,
                             const float *rstd, int Mrows) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= DIM) return;
    float sg = 0.0f, sb = 0.0f;
    for (int m = 0; m < Mrows; m++) {
        float dy = dY[(size_t)m * DIM + d];
        sg += dy * (X[(size_t)m * DIM + d] - mean[m]) * rstd[m];
        sb += dy;
    }
    dg[d] += sg; db[d] += sb;
}

void ln_fwd(const float *X, const float *g, const float *b, float *Y,
            float *mean, float *rstd, int Mrows) {
    k_ln_fwd<<<Mrows, 256>>>(X, g, b, Y, mean, rstd);
}
void ln_bwd_dx(float *dX, const float *dY, const float *X, const float *g,
               const float *mean, const float *rstd, int Mrows) {
    k_ln_bwd_dx<<<Mrows, 256>>>(dX, dY, X, g, mean, rstd);
}
void ln_bwd_dgb(float *dg, float *db, const float *dY, const float *X,
                const float *mean, const float *rstd, int Mrows) {
    k_ln_bwd_dgb<<<(DIM + 255) / 256, 256>>>(dg, db, dY, X, mean, rstd, Mrows);
}

/* ========================== elementwise ========================== */

#define GELU_K 0.7978845608028654f

__global__ void k_gelu_fwd(float *Y, const float* __restrict__ X, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    Y[i] = 0.5f * x * (1.0f + tanhf(GELU_K * (x + 0.044715f * x * x * x)));
}
__global__ void k_gelu_bwd(float *dX, const float* __restrict__ dY,
                           const float* __restrict__ X, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    float u = GELU_K * (x + 0.044715f * x * x * x);
    float t = tanhf(u);
    float dudx = GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
    dX[i] = dY[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dudx);  /* SET */
}
__global__ void k_add(float *C, const float* __restrict__ A, const float* __restrict__ B, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void gelu_fwd(float *Y, const float *X, long n) { k_gelu_fwd<<<(unsigned)((n + 255) / 256), 256>>>(Y, X, n); }
void gelu_bwd(float *dX, const float *dY, const float *X, long n) { k_gelu_bwd<<<(unsigned)((n + 255) / 256), 256>>>(dX, dY, X, n); }
void vec_add(float *C, const float *A, const float *B, long n) { k_add<<<(unsigned)((n + 255) / 256), 256>>>(C, A, B, n); }

/* ========================== attention ========================== */
/*
 * Default path: attention as strided-batched cuBLAS GEMMs.
 *   fwd:  S = scale * Q K^T   -> causal softmax kernel ->  att = P V
 *   bwd:  dV = P^T dAtt ; dP = dAtt V^T ; softmax-bwd kernel (dS in place) ;
 *         dQ = dS K ; dK = dS^T Q
 * Head slices live inside the (T, DIM) activation rows (pitch DIM), which maps
 * exactly onto strided-batched GEMMs (head stride D_HEAD, probs stride T*T).
 * Only the two softmax kernels below remain custom.
 */

__global__ void k_causal_softmax(float *scores) {
    int bh = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
    float *row = scores + ((size_t)bh * T_SIZE + i) * T_SIZE;
    __shared__ float red[256];
    float lmax = -1e30f;
    for (int j = tid; j <= i; j += blockDim.x) lmax = fmaxf(lmax, row[j]);
    red[tid] = lmax; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float ls = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) { float e = __expf(row[j] - mx); row[j] = e; ls += e; }
    red[tid] = ls; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float inv = 1.0f / red[0];
    for (int j = tid; j <= i; j += blockDim.x) row[j] *= inv;
    for (int j = i + 1 + tid; j < T_SIZE; j += blockDim.x) row[j] = 0.0f;  /* mask the future */
}

/* dS = P * (dP - rowdot(P, dP)) * scale, computed in the dP buffer in place */
__global__ void k_softmax_bwd(float *dP, const float* __restrict__ probs, float scale) {
    int bh = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
    float *dp = dP + ((size_t)bh * T_SIZE + i) * T_SIZE;
    const float *p = probs + ((size_t)bh * T_SIZE + i) * T_SIZE;
    __shared__ float red[256];
    float lo = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) lo += p[j] * dp[j];
    red[tid] = lo; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float Di = red[0];
    for (int j = tid; j <= i; j += blockDim.x) dp[j] = p[j] * (dp[j] - Di) * scale;
    for (int j = i + 1 + tid; j < T_SIZE; j += blockDim.x) dp[j] = 0.0f;
}

#ifdef USE_HAND_GEMM
/* forward: one block per (b*h, i); 128 threads == D_HEAD */
__global__ void k_attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                           const float* __restrict__ V, float *att, float *probs) {
    int bh = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
    int b = bh / N_HEADS, h = bh % N_HEADS, off = h * D_HEAD;
    const float scale = rsqrtf((float)D_HEAD);
    const float *qi = Q + ((size_t)(b * T_SIZE + i) * DIM) + off;
    float *p = probs + ((size_t)bh * T_SIZE + i) * T_SIZE;
    __shared__ float qs[D_HEAD], red[D_HEAD];

    qs[tid] = qi[tid]; __syncthreads();

    float lmax = -1e30f;
    for (int j = tid; j <= i; j += blockDim.x) {
        const float *kj = K + ((size_t)(b * T_SIZE + j) * DIM) + off;
        float dot = 0.0f;
        for (int d = 0; d < D_HEAD; d++) dot += qs[d] * kj[d];
        dot *= scale; p[j] = dot;
        if (dot > lmax) lmax = dot;
    }
    red[tid] = lmax; __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    float maxv = red[0]; __syncthreads();

    float lsum = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) { float e = __expf(p[j] - maxv); p[j] = e; lsum += e; }
    red[tid] = lsum; __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float inv = 1.0f / red[0];
    for (int j = tid; j <= i; j += blockDim.x) p[j] *= inv;
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j <= i; j++)
        acc += p[j] * V[((size_t)(b * T_SIZE + j) * DIM) + off + tid];
    att[((size_t)(b * T_SIZE + i) * DIM) + off + tid] = acc;
}

/* backward pass A: per (bh, i) -> dQ row + D_i = sum_j p*dp (softmax-jacobian dot) */
__global__ void k_attn_bwd_dq(float *dQ, float *Dbuf, const float* __restrict__ dAtt,
                              const float* __restrict__ K, const float* __restrict__ V,
                              const float* __restrict__ probs) {
    int bh = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
    int b = bh / N_HEADS, h = bh % N_HEADS, off = h * D_HEAD;
    const float scale = rsqrtf((float)D_HEAD);
    const float *dai = dAtt + ((size_t)(b * T_SIZE + i) * DIM) + off;
    const float *p = probs + ((size_t)bh * T_SIZE + i) * T_SIZE;
    __shared__ float das[D_HEAD], red[D_HEAD];

    das[tid] = dai[tid]; __syncthreads();

    /* D_i = sum_j p[j] * dot(dAtt_i, V_j) */
    float lD = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) {
        const float *vj = V + ((size_t)(b * T_SIZE + j) * DIM) + off;
        float dp = 0.0f;
        for (int d = 0; d < D_HEAD; d++) dp += das[d] * vj[d];
        lD += p[j] * dp;
    }
    red[tid] = lD; __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float Di = red[0];
    if (tid == 0) Dbuf[(size_t)bh * T_SIZE + i] = Di;
    __syncthreads();

    /* dQ[i][d] = sum_j ds_j * K_j[d],  ds_j = p_j * (dp_j - D_i) * scale */
    float acc = 0.0f;
    for (int j = 0; j <= i; j++) {
        const float *vj = V + ((size_t)(b * T_SIZE + j) * DIM) + off;
        red[tid] = das[tid] * vj[tid]; __syncthreads();
        for (int st = 64; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
        float dp = red[0]; __syncthreads();
        float ds = p[j] * (dp - Di) * scale;
        acc += ds * K[((size_t)(b * T_SIZE + j) * DIM) + off + tid];
    }
    dQ[((size_t)(b * T_SIZE + i) * DIM) + off + tid] = acc;   /* each (i,d) owned once */
}

/* backward pass B: per (bh, j) -> dK and dV rows (gather over i >= j; no atomics) */
__global__ void k_attn_bwd_dkv(float *dK, float *dV, const float* __restrict__ dAtt,
                               const float* __restrict__ Q, const float* __restrict__ V,
                               const float* __restrict__ probs, const float* __restrict__ Dbuf) {
    int bh = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    int b = bh / N_HEADS, h = bh % N_HEADS, off = h * D_HEAD;
    const float scale = rsqrtf((float)D_HEAD);
    __shared__ float vjs[D_HEAD], red[D_HEAD];

    vjs[tid] = V[((size_t)(b * T_SIZE + j) * DIM) + off + tid]; __syncthreads();

    float dkacc = 0.0f, dvacc = 0.0f;
    for (int i = j; i < T_SIZE; i++) {
        const float *dai = dAtt + ((size_t)(b * T_SIZE + i) * DIM) + off;
        float pij = probs[((size_t)bh * T_SIZE + i) * T_SIZE + j];
        red[tid] = dai[tid] * vjs[tid]; __syncthreads();
        for (int st = 64; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
        float dp = red[0]; __syncthreads();
        float ds = pij * (dp - Dbuf[(size_t)bh * T_SIZE + i]) * scale;
        dvacc += pij * dai[tid];
        dkacc += ds * Q[((size_t)(b * T_SIZE + i) * DIM) + off + tid];
    }
    dK[((size_t)(b * T_SIZE + j) * DIM) + off + tid] = dkacc;
    dV[((size_t)(b * T_SIZE + j) * DIM) + off + tid] = dvacc;
}
#endif /* USE_HAND_GEMM */

void attn_fwd(const float *Q, const float *K, const float *V, float *att, float *probs) {
#ifdef USE_HAND_GEMM
    dim3 grid(B_SIZE * N_HEADS, T_SIZE);
    k_attn_fwd<<<grid, D_HEAD>>>(Q, K, V, att, probs);
#else
    const float scale = 1.0f / sqrtf((float)D_HEAD), one = 1.0f, zero = 0.0f;
    const long long sTT = (long long)T_SIZE * T_SIZE;
    for (int b = 0; b < B_SIZE; b++) {
        const float *Qb = Q + (size_t)b * T_SIZE * DIM;
        const float *Kb = K + (size_t)b * T_SIZE * DIM;
        const float *Vb = V + (size_t)b * T_SIZE * DIM;
        float *Pb = probs + (size_t)b * N_HEADS * sTT;
        float *Ab = att + (size_t)b * T_SIZE * DIM;
        /* S = scale * Q K^T  (row-major, per head) */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            T_SIZE, T_SIZE, D_HEAD, &scale,
            Kb, DIM, D_HEAD, Qb, DIM, D_HEAD,
            &zero, Pb, T_SIZE, sTT, N_HEADS);
    }
    dim3 g(B_SIZE * N_HEADS, T_SIZE);
    k_causal_softmax<<<g, 256>>>(probs);
    for (int b = 0; b < B_SIZE; b++) {
        const float *Vb = V + (size_t)b * T_SIZE * DIM;
        const float *Pb = probs + (size_t)b * N_HEADS * sTT;
        float *Ab = att + (size_t)b * T_SIZE * DIM;
        /* att = P V */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D_HEAD, T_SIZE, T_SIZE, &one,
            Vb, DIM, D_HEAD, Pb, T_SIZE, sTT,
            &zero, Ab, DIM, D_HEAD, N_HEADS);
    }
#endif
}

/* `scratch` must hold B_SIZE*N_HEADS*T_SIZE*T_SIZE floats (dP/dS workspace) */
void attn_bwd(float *dQ, float *dK, float *dV, float *scratch,
              const float *dAtt, const float *Q, const float *K, const float *V,
              const float *probs) {
#ifdef USE_HAND_GEMM
    dim3 grid(B_SIZE * N_HEADS, T_SIZE);
    k_attn_bwd_dq<<<grid, D_HEAD>>>(dQ, scratch, dAtt, K, V, probs);
    k_attn_bwd_dkv<<<grid, D_HEAD>>>(dK, dV, dAtt, Q, V, probs, scratch);
#else
    const float scale = 1.0f / sqrtf((float)D_HEAD), one = 1.0f, zero = 0.0f;
    const long long sTT = (long long)T_SIZE * T_SIZE;
    for (int b = 0; b < B_SIZE; b++) {
        const float *dAb = dAtt + (size_t)b * T_SIZE * DIM;
        const float *Qb = Q + (size_t)b * T_SIZE * DIM;
        const float *Kb = K + (size_t)b * T_SIZE * DIM;
        const float *Vb = V + (size_t)b * T_SIZE * DIM;
        const float *Pb = probs + (size_t)b * N_HEADS * sTT;
        float *dPb = scratch + (size_t)b * N_HEADS * sTT;
        float *dQb = dQ + (size_t)b * T_SIZE * DIM;
        float *dKb = dK + (size_t)b * T_SIZE * DIM;
        float *dVb = dV + (size_t)b * T_SIZE * DIM;
        /* dV = P^T dAtt */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            D_HEAD, T_SIZE, T_SIZE, &one,
            dAb, DIM, D_HEAD, Pb, T_SIZE, sTT,
            &zero, dVb, DIM, D_HEAD, N_HEADS);
        /* dP = dAtt V^T */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            T_SIZE, T_SIZE, D_HEAD, &one,
            Vb, DIM, D_HEAD, dAb, DIM, D_HEAD,
            &zero, dPb, T_SIZE, sTT, N_HEADS);
    }
    dim3 g(B_SIZE * N_HEADS, T_SIZE);
    k_softmax_bwd<<<g, 256>>>(scratch, probs, scale);
    for (int b = 0; b < B_SIZE; b++) {
        const float *Qb = Q + (size_t)b * T_SIZE * DIM;
        const float *Kb = K + (size_t)b * T_SIZE * DIM;
        const float *dSb = scratch + (size_t)b * N_HEADS * sTT;
        float *dQb = dQ + (size_t)b * T_SIZE * DIM;
        float *dKb = dK + (size_t)b * T_SIZE * DIM;
        /* dQ = dS K */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D_HEAD, T_SIZE, T_SIZE, &one,
            Kb, DIM, D_HEAD, dSb, T_SIZE, sTT,
            &zero, dQb, DIM, D_HEAD, N_HEADS);
        /* dK = dS^T Q */
        cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            D_HEAD, T_SIZE, T_SIZE, &one,
            Qb, DIM, D_HEAD, dSb, T_SIZE, sTT,
            &zero, dKb, DIM, D_HEAD, N_HEADS);
    }
#endif
}

/* ========================== embeddings ========================== */

__global__ void k_emb_fwd(float *X, const float* __restrict__ wte,
                          const float* __restrict__ wpe, const int* __restrict__ toks) {
    int m = blockIdx.x, tpos = m % T_SIZE, tok = toks[m];
    for (int d = threadIdx.x; d < DIM; d += blockDim.x)
        X[(size_t)m * DIM + d] = wte[(size_t)tok * DIM + d] + wpe[(size_t)tpos * DIM + d];
}
__global__ void k_emb_bwd(float *dwte, float *dwpe, const float* __restrict__ dX,
                          const int* __restrict__ toks) {
    int m = blockIdx.x, tpos = m % T_SIZE, tok = toks[m];
    for (int d = threadIdx.x; d < DIM; d += blockDim.x) {
        float g = dX[(size_t)m * DIM + d];
        atomicAdd(&dwte[(size_t)tok * DIM + d], g);
        atomicAdd(&dwpe[(size_t)tpos * DIM + d], g);
    }
}

void emb_fwd(float *X, const float *wte, const float *wpe, const int *tokens) {
    k_emb_fwd<<<M_ROWS, 256>>>(X, wte, wpe, tokens);
}
void emb_bwd(float *dwte, float *dwpe, const float *dX, const int *tokens) {
    k_emb_bwd<<<M_ROWS, 256>>>(dwte, dwpe, dX, tokens);
}

/* ========================== cross-entropy ========================== */

__global__ void k_ce(const float* __restrict__ logits, const int* __restrict__ tgt,
                     float *dlogits, float *loss) {
    int m = blockIdx.x, tid = threadIdx.x;
    const float *lo = logits + (size_t)m * VOCAB_SIZE;
    float *dl = dlogits + (size_t)m * VOCAB_SIZE;
    __shared__ float red[256];

    float lmax = -1e30f;
    for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) lmax = fmaxf(lmax, lo[v]);
    red[tid] = lmax; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    float mx = red[0]; __syncthreads();

    float ls = 0.0f;
    for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) { float e = __expf(lo[v] - mx); dl[v] = e; ls += e; }
    red[tid] = ls; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    float sum = red[0];

    int t = tgt[m];
    if (tid == 0) atomicAdd(loss, -logf(dl[t] / sum + 1e-30f) / M_ROWS);
    __syncthreads();                     /* keep dl[t] intact until read above */

    float inv = 1.0f / sum;
    for (int v = tid; v < VOCAB_SIZE; v += blockDim.x)
        dl[v] = (dl[v] * inv - (v == t ? 1.0f : 0.0f)) / M_ROWS;
}

void ce_fwd_bwd(const float *logits, const int *targets, float *dlogits, float *loss_dev) {
    k_ce<<<M_ROWS, 256>>>(logits, targets, dlogits, loss_dev);
}

/* ========================== AdamW ========================== */

__global__ void k_adamw(float *w, float *g, float *m, float *v, long n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gr = g[i];
    float mi = m[i] = b1 * m[i] + (1.0f - b1) * gr;
    float vi = v[i] = b2 * v[i] + (1.0f - b2) * gr * gr;
    w[i] -= lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps) + wd * w[i]);
    g[i] = 0.0f;                          /* zero grad for the next step */
}

void adamw_step(float *w, float *g, float *m, float *v, long n,
                float lr, float beta1, float beta2, float eps, float wd, int step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    k_adamw<<<(unsigned)((n + 255) / 256), 256>>>(w, g, m, v, n, lr, beta1, beta2, eps, wd, bc1, bc2);
}
