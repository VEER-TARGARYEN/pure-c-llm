/*
 * tensor.c  --  Implementation of the flat-array CPU tensor-math library.
 *
 * Build (library object):      gcc -O3 -fopenmp -c tensor.c -o tensor.o
 * Build the self-test:         gcc -O3 -fopenmp -DTEST_TENSOR tensor.c -o tensor_test.exe -lm
 * Run the self-test:           .\tensor_test.exe
 *
 * Everything is row-major and the caller owns all memory (see tensor.h).
 */

#include "tensor.h"
#include <math.h>     /* sqrtf, expf, tanhf */
#include <stddef.h>   /* size_t            */

/* ==========================================================================
 * matmul:  C(M,N) = A(M,K) @ B(K,N)
 * --------------------------------------------------------------------------
 * Loop order is i-k-j, NOT the textbook i-j-k. Why it is cache-friendly:
 *
 *   - The innermost loop runs over j. In that loop we touch B[k*N + j] and
 *     C[i*N + j], both of which are CONTIGUOUS in j. Sequential memory access
 *     is what the CPU prefetcher and SIMD units want, so this inner loop
 *     auto-vectorizes well under -O3.
 *   - A[i*K + k] is loop-invariant across j, so we hoist it into the scalar
 *     `a` and reuse it for the whole row of B.
 *
 * The cost of i-k-j is that C must be ACCUMULATED into, so we zero each output
 * row first. We parallelize the outer i loop: every thread owns a disjoint set
 * of output rows, so there are no write conflicts and no locks are needed.
 *
 * `restrict` promises the compiler that A, B, C do not overlap, which lets it
 * vectorize the inner loop aggressively. (Hence: C must be a distinct buffer.)
 * ========================================================================== */
void matmul(const float *restrict A, const float *restrict B, float *restrict C,
            int M, int K, int N)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float *crow = C + (size_t)i * N;

        /* 1) zero this output row so we can accumulate into it */
        for (int j = 0; j < N; j++)
            crow[j] = 0.0f;

        /* 2) accumulate row i of C as a weighted sum of rows of B */
        for (int k = 0; k < K; k++) {
            const float  a    = A[(size_t)i * K + k];
            const float *brow = B + (size_t)k * N;
            for (int j = 0; j < N; j++)
                crow[j] += a * brow[j];
        }
    }
}

/* ==========================================================================
 * add:  C = A + B, element-wise.  (Residual connections.)
 * One flat loop; trivially parallel since every index is independent.
 * ========================================================================== */
void add(const float *A, const float *B, float *C, int n)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

/* ==========================================================================
 * layernorm:  per-row normalize over the feature (cols) dimension, then affine.
 * --------------------------------------------------------------------------
 * For each row we make three passes over its `cols` features:
 *   pass 1 -> mean
 *   pass 2 -> variance (population variance: divide by cols)
 *   pass 3 -> normalize, then scale by gamma and shift by beta
 *
 * Each row is independent, so we parallelize the outer row loop. All the
 * per-row scratch values (mean, var, inv_std) are plain locals, which means
 * every thread automatically gets its own private copies -- no data races.
 * ========================================================================== */
void layernorm(const float *X, const float *gamma, const float *beta, float *Y,
               int rows, int cols, float eps)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *x = X + (size_t)r * cols;
        float       *y = Y + (size_t)r * cols;

        /* pass 1: mean */
        float mean = 0.0f;
        for (int c = 0; c < cols; c++)
            mean += x[c];
        mean /= (float)cols;

        /* pass 2: variance */
        float var = 0.0f;
        for (int c = 0; c < cols; c++) {
            float d = x[c] - mean;
            var += d * d;
        }
        var /= (float)cols;

        /* pass 3: normalize, scale (gamma), shift (beta) */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; c++)
            y[c] = (x[c] - mean) * inv_std * gamma[c] + beta[c];
    }
}

/* ==========================================================================
 * softmax:  numerically-stable, applied per row over `cols` elements.
 * --------------------------------------------------------------------------
 * The unstable formula exp(x_i) / sum(exp(x_j)) overflows when any x is large.
 * The fix is to subtract the row max first: exp(x_i - max) is always <= 1, and
 * the result is mathematically identical because the constant factor exp(-max)
 * cancels between numerator and denominator.
 * ========================================================================== */
void softmax(const float *X, float *Y, int rows, int cols)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *x = X + (size_t)r * cols;
        float       *y = Y + (size_t)r * cols;

        /* 1) find the row max for numerical stability */
        float maxv = x[0];
        for (int c = 1; c < cols; c++)
            if (x[c] > maxv) maxv = x[c];

        /* 2) exponentiate the shifted values and accumulate their sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float e = expf(x[c] - maxv);
            y[c] = e;
            sum += e;
        }

        /* 3) normalize so the row sums to 1 */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++)
            y[c] *= inv_sum;
    }
}

/* ==========================================================================
 * gelu:  element-wise activation, tanh approximation.
 *
 *   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
 *
 * This is the same approximation GPT-2 and most transformer code use. It is a
 * smooth gate: large positive x passes through ~unchanged, large negative x is
 * squashed toward 0, and values near 0 are scaled by a soft curve.
 * ========================================================================== */
void gelu(const float *X, float *Y, int n)
{
    const float k = 0.7978845608028654f; /* sqrt(2 / pi) */

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float x     = X[i];
        float inner = k * (x + 0.044715f * x * x * x);
        Y[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/* ==========================================================================
 * SELF-TEST  --  compile with -DTEST_TENSOR to get a standalone executable.
 * --------------------------------------------------------------------------
 * It runs one big matmul on a single thread, then again on all cores, prints
 * both wall-clock times, and reports the speedup. If OpenMP is actually doing
 * work the multi-threaded run should be several times faster. A small
 * correctness check against a double-precision reference guards against the
 * loop math being wrong.
 * ========================================================================== */
#ifdef TEST_TENSOR

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Wall-clock seconds. We MUST use omp_get_wtime() (not clock()) when OpenMP is
 * on: clock() returns summed CPU time across all threads, which would hide the
 * speedup entirely. */
static double now_seconds(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

int main(void)
{
    const int M = 1024, K = 1024, N = 1024;

    float *A = (float *)malloc((size_t)M * K * sizeof(float));
    float *B = (float *)malloc((size_t)K * N * sizeof(float));
    float *C = (float *)malloc((size_t)M * N * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "out of memory\n"); return 1; }

    /* Deterministic, varied data so the result isn't trivially constant. */
    for (int i = 0; i < M * K; i++) A[i] = (float)((i % 13) - 6) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = (float)((i % 7)  - 3) * 0.1f;

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
    printf("OpenMP: ENABLED  (max threads = %d)\n", max_threads);
#else
    printf("OpenMP: DISABLED (rebuild with -fopenmp to parallelize)\n");
#endif
    printf("matmul size: %d x %d x %d\n\n", M, K, N);

    /* ---- single-threaded run ---- */
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    double t0 = now_seconds();
    matmul(A, B, C, M, K, N);
    double single = now_seconds() - t0;
    printf("  1 thread      : %.4f s\n", single);

    /* ---- multi-threaded run ---- */
#ifdef _OPENMP
    omp_set_num_threads(max_threads);
#endif
    double t1 = now_seconds();
    matmul(A, B, C, M, K, N);
    double multi = now_seconds() - t1;
    printf("  %2d threads     : %.4f s\n", max_threads, multi);

    if (multi > 0.0)
        printf("  speedup       : %.2fx\n", single / multi);

    /* ---- correctness check on one output element ---- */
    double ref = 0.0;
    for (int k = 0; k < K; k++)
        ref += (double)A[0 * K + k] * (double)B[k * N + 0];
    printf("\n  check C[0,0]  : got %.5f, reference %.5f\n", C[0], (float)ref);

    free(A);
    free(B);
    free(C);
    return 0;
}

#endif /* TEST_TENSOR */
