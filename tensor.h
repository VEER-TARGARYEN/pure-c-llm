/*
 * tensor.h  --  Minimal CPU tensor-math library for a from-scratch C LLM.
 *
 * Piece 2 of a zero-dependency next-word-prediction model.
 *
 * --------------------------------------------------------------------------
 * DESIGN PHILOSOPHY
 * --------------------------------------------------------------------------
 * There is NO tensor struct. Every function takes raw, flat `float*` arrays
 * plus the dimensions as explicit integers. This is deliberate:
 *
 *   - It is exactly how you hand data to a CUDA/GPU kernel later: a bare
 *     device pointer and a few ints describing the shape.
 *   - It keeps the data layout obvious. Everything is ROW-MAJOR: element
 *     (r, c) of a (rows x cols) matrix lives at index  r * cols + c.
 *
 * MEMORY CONTRACT
 *   The CALLER allocates every buffer (input and output) and frees them.
 *   These functions never malloc/free; they only read inputs and fill the
 *   output pointer. Output buffers must be large enough for the stated shape.
 * --------------------------------------------------------------------------
 */
#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * matmul: C = A @ B   (standard matrix multiply)
 *
 *   A : (M x K) row-major
 *   B : (K x N) row-major
 *   C : (M x N) row-major  -- fully overwritten (internally zeroed first)
 *
 * NOTE: C must NOT alias A or B (it is a distinct output buffer).
 * Parallelized over the M output rows with OpenMP.
 */
void matmul(const float *A, const float *B, float *C, int M, int K, int N);

/*
 * add: C = A + B, element-wise over `n` elements. Used for residual streams.
 * Safe to call in-place (C may equal A or B). Parallelized with OpenMP.
 */
void add(const float *A, const float *B, float *C, int n);

/*
 * layernorm: normalize each of `rows` rows independently across its `cols`
 * features, then apply the learned affine transform:
 *
 *     y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 *   X     : (rows x cols) input
 *   gamma : (cols) scale  (the learned "weight")
 *   beta  : (cols) shift  (the learned "bias")
 *   Y     : (rows x cols) output
 *   eps   : small constant for numerical stability (e.g. 1e-5f)
 *
 * Safe to call in-place (Y may equal X). Parallelized over rows with OpenMP.
 */
void layernorm(const float *X, const float *gamma, const float *beta, float *Y,
               int rows, int cols, float eps);

/*
 * softmax: numerically-stable softmax applied to each of `rows` rows over its
 * `cols` elements (the max is subtracted before exp to avoid overflow).
 * Safe to call in-place (Y may equal X). Parallelized over rows with OpenMP.
 */
void softmax(const float *X, float *Y, int rows, int cols);

/*
 * gelu: Gaussian Error Linear Unit activation, element-wise over `n` elements,
 * using the standard tanh approximation. Safe to call in-place (Y may equal X).
 */
void gelu(const float *X, float *Y, int n);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
