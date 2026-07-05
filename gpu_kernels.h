/*
 * gpu_kernels.h  --  CUDA kernel API for the 343M C-LLM trainer (Kaggle T4).
 *
 * Pure C-style CUDA. Every function below is a host-side launcher that runs
 * one GPU kernel; all pointers are DEVICE pointers unless noted. The math
 * mirrors the CPU train.c (verified to convergence) operation-for-operation:
 * pre-norm GPT-2 blocks, LayerNorm + tanh-GELU, learned positions, tied head.
 */
#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ---- 343M architecture (matches the local CKP1 ecosystem) ---- */
#define DIM         1536
#define N_LAYERS    12
#define N_HEADS     12              /* d_head = 128 */
#define D_HEAD      (DIM / N_HEADS)
#define D_FF        (4 * DIM)       /* 6144 */
#define MAX_SEQ_LEN 1024
#define VOCAB_SIZE  2048
#define LN_EPS      1e-5f

/* ---- training shape: B=4 x T=1024 = 4096 tokens/step fits a 16GB T4 with
 *      full stored activations (~13.5GB total). If you OOM: -DB_SIZE=2. ---- */
#ifndef B_SIZE
#define B_SIZE 4
#endif
#define T_SIZE MAX_SEQ_LEN
#define M_ROWS (B_SIZE * T_SIZE)

#define CUDA_CHECK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
    exit(1); } } while (0)

/* one-time GPU setup (creates the cuBLAS handle) -- call before any gemm */
void gpu_init(void);

/* ---- GEMM family ----
 * Default: cuBLAS (part of the CUDA toolkit -- pure C, no frameworks).
 * Compile with -DUSE_HAND_GEMM to use the hand-written tiled kernels instead.
 * nn: C(M,N)  = A(M,K) @ B(K,N) + bias          (forward projections)
 * nt: C(M,N) ?= A(M,K) @ B(N,K)^T               (dX, and tied-head logits)
 * tn: C(K,N) += A(M,K)^T @ B(M,N)               (dW, dwte -- always accumulates)
 */
void gemm_nn(const float *A, const float *B, const float *bias, float *C, int M, int K, int N);
void gemm_nt(const float *A, const float *B, float *C, int M, int K, int N, int accum);
void gemm_tn(const float *A, const float *B, float *C, int M, int K, int N);
void bias_bwd(float *db, const float *dY, int M, int N);           /* db += colsum(dY) */

/* ---- LayerNorm (saves mean/rstd for backward; dX accumulates like train.c) ---- */
void ln_fwd(const float *X, const float *gamma, const float *beta, float *Y,
            float *mean, float *rstd, int Mrows);
void ln_bwd_dx(float *dX, const float *dY, const float *X, const float *gamma,
               const float *mean, const float *rstd, int Mrows);
void ln_bwd_dgb(float *dgamma, float *dbeta, const float *dY, const float *X,
                const float *mean, const float *rstd, int Mrows);

/* ---- elementwise ---- */
void gelu_fwd(float *Y, const float *X, long n);
void gelu_bwd(float *dX, const float *dY, const float *X, long n);  /* SETs dX */
void vec_add(float *C, const float *A, const float *B, long n);     /* C = A + B */

/* ---- causal multi-head attention (probs stored for backward) ---- */
void attn_fwd(const float *Q, const float *K, const float *V,
              float *att, float *probs);
/* scratch: B_SIZE*N_HEADS*T_SIZE*T_SIZE floats (dP/dS workspace) */
void attn_bwd(float *dQ, float *dK, float *dV, float *scratch,
              const float *dAtt, const float *Q, const float *K, const float *V,
              const float *probs);

/* ---- embeddings ---- */
void emb_fwd(float *X, const float *wte, const float *wpe, const int *tokens);
void emb_bwd(float *dwte, float *dwpe, const float *dX, const int *tokens);

/* ---- fused softmax + cross-entropy: mean loss added into *loss_dev (zero it
 *      first); dlogits = (softmax - onehot)/M_ROWS ---- */
void ce_fwd_bwd(const float *logits, const int *targets, float *dlogits, float *loss_dev);

/* ---- AdamW over a flat parameter arena; zeroes grads afterwards ---- */
void adamw_step(float *w, float *g, float *m, float *v, long n,
                float lr, float beta1, float beta2, float eps, float wd, int step);

#endif /* GPU_KERNELS_H */
