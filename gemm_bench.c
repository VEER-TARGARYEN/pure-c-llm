/*
 * gemm_bench.c  --  GEMM optimization ladder for the C-LLM engine (roofline).
 *
 * Single-threaded on purpose: measure raw KERNEL quality vs a measured compute
 * roof before adding threading. C(M,N) = A(M,K) @ B(K,N), row-major float32
 * (B is the weight, stored (K,N)), matching the model's matmul layout.
 *
 * Build:  gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp \
 *             gemm_bench.c -o gemm_bench.exe -lm
 *
 * Rungs:
 *   v0 naive i-j-k        textbook; B column-walk -> cache death
 *   v1 i-k-j (train.c)    contiguous inner FMA, but streams C row 2K times
 *   v2 microkernel 8x16   C tile in ZMM regs; 1 B-load feeds 8 FMAs
 *   v3 packed + blocked   Goto/BLIS: pack A/B into cache-sized panels so the
 *                         big matrix is read from DRAM ~once (this is the rung
 *                         that holds up once a matrix exceeds L3)
 * Later: multithread v3 over M-tiles, v4 VNNI INT8, v5 vs OpenBLAS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <stdint.h>
#include <omp.h>          /* omp_get_wtime (wall clock); kernels stay 1-thread */

#ifdef USE_OPENBLAS
#include <cblas.h>
extern void openblas_set_num_threads(int);
/* v5: production reference. Row-major C = A@B, matching our layout. */
static void gemm_blas(const float *A, const float *B, float *C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0f, A, K, B, N, 0.0f, C, N);
}
#endif

/* ---- register-tile + cache-block sizes (tuned to this chip's L1/L2/L3) ----
 * Bp panel Kc*Nc = 256*512*4 = 512 KB  (fits L2 1.25 MB)
 * Ap panel Mc*Kc = 128*256*4 = 128 KB  (fits L2 alongside Bp)
 * Bp micro-panel Kc*NR = 256*16*4 = 16 KB (fits L1D 48 KB)               */
#define MR 8
#define NR 16
#define MC 128
#define KC 256
#define NC 512

static double now(void) { return omp_get_wtime(); }
static void fill_rand(float *p, long n) {
    for (long i = 0; i < n; i++) p[i] = (float)rand()/(float)RAND_MAX*2.0f-1.0f;
}
static double max_abs_diff(const float *X, const float *Y, long n) {
    double d = 0.0;
    for (long i = 0; i < n; i++) { double e = fabs((double)X[i]-Y[i]); if (e>d) d=e; }
    return d;
}

/* ================= v0 naive i-j-k ================= */
static void gemm_naive(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k]*B[(size_t)k*N+j];
            C[(size_t)i*N+j] = acc;
        }
}

/* ================= v1 i-k-j (train.c's kernel) ================= */
static void gemm_ikj(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        float *o = C + (size_t)i*N;
        for (int n = 0; n < N; n++) o[n] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[(size_t)i*K+k];
            const float *wr = B + (size_t)k*N;
            for (int n = 0; n < N; n++) o[n] += a*wr[n];
        }
    }
}

/* ================= v2 unpacked 8x16 microkernel ================= */
static inline void micro_unpacked(const float *A, const float *B, float *C,
                                  int j0, int K, int N) {
    __m512 c0=_mm512_setzero_ps(),c1=_mm512_setzero_ps(),c2=_mm512_setzero_ps(),c3=_mm512_setzero_ps();
    __m512 c4=_mm512_setzero_ps(),c5=_mm512_setzero_ps(),c6=_mm512_setzero_ps(),c7=_mm512_setzero_ps();
    for (int k = 0; k < K; k++) {
        __m512 b = _mm512_loadu_ps(B + (size_t)k*N + j0);
        const float *a = A + k;
        c0=_mm512_fmadd_ps(_mm512_set1_ps(a[0*K]),b,c0); c1=_mm512_fmadd_ps(_mm512_set1_ps(a[1*K]),b,c1);
        c2=_mm512_fmadd_ps(_mm512_set1_ps(a[2*K]),b,c2); c3=_mm512_fmadd_ps(_mm512_set1_ps(a[3*K]),b,c3);
        c4=_mm512_fmadd_ps(_mm512_set1_ps(a[4*K]),b,c4); c5=_mm512_fmadd_ps(_mm512_set1_ps(a[5*K]),b,c5);
        c6=_mm512_fmadd_ps(_mm512_set1_ps(a[6*K]),b,c6); c7=_mm512_fmadd_ps(_mm512_set1_ps(a[7*K]),b,c7);
    }
    _mm512_storeu_ps(C+0*N+j0,c0);_mm512_storeu_ps(C+1*N+j0,c1);_mm512_storeu_ps(C+2*N+j0,c2);_mm512_storeu_ps(C+3*N+j0,c3);
    _mm512_storeu_ps(C+4*N+j0,c4);_mm512_storeu_ps(C+5*N+j0,c5);_mm512_storeu_ps(C+6*N+j0,c6);_mm512_storeu_ps(C+7*N+j0,c7);
}
static void gemm_micro(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i0 = 0; i0 < M; i0 += MR)
        for (int j0 = 0; j0 < N; j0 += NR)
            micro_unpacked(A + (size_t)i0*K, B, C + (size_t)i0*N, j0, K, N);
}

/* ================= v3 packed + cache-blocked (Goto/BLIS) ================= *
 * Pack B[pc:pc+kc, jc:jc+nc] into contiguous NR-wide micro-panels, and
 * A[ic:ic+mc, pc:pc+kc] into contiguous MR-wide micro-panels, so the kernel
 * streams unit-stride data and the big matrix is read from DRAM ~once.       */
static void pack_B(const float *B, float *Bp, int pc, int jc, int kc, int nc, int N) {
    for (int jp = 0; jp < nc/NR; jp++) {
        float *dst = Bp + (size_t)jp*kc*NR;
        for (int k = 0; k < kc; k++) {
            const float *src = B + (size_t)(pc+k)*N + jc + jp*NR;
            for (int n = 0; n < NR; n++) dst[k*NR+n] = src[n];
        }
    }
}
static void pack_A(const float *A, float *Ap, int ic, int pc, int mc, int kc, int K) {
    for (int ip = 0; ip < mc/MR; ip++) {
        float *dst = Ap + (size_t)ip*kc*MR;
        for (int k = 0; k < kc; k++)
            for (int r = 0; r < MR; r++)
                dst[k*MR+r] = A[(size_t)(ic+ip*MR+r)*K + pc + k];
    }
}
/* microkernel over packed panels; ACCUMULATES into C (K is blocked). */
static inline void micro_packed(const float *Ap, const float *Bp, float *C, int ldc, int kc) {
    __m512 c0=_mm512_loadu_ps(C+0*ldc),c1=_mm512_loadu_ps(C+1*ldc),c2=_mm512_loadu_ps(C+2*ldc),c3=_mm512_loadu_ps(C+3*ldc);
    __m512 c4=_mm512_loadu_ps(C+4*ldc),c5=_mm512_loadu_ps(C+5*ldc),c6=_mm512_loadu_ps(C+6*ldc),c7=_mm512_loadu_ps(C+7*ldc);
    for (int k = 0; k < kc; k++) {
        __m512 b = _mm512_loadu_ps(Bp + (size_t)k*NR);
        const float *a = Ap + (size_t)k*MR;
        c0=_mm512_fmadd_ps(_mm512_set1_ps(a[0]),b,c0); c1=_mm512_fmadd_ps(_mm512_set1_ps(a[1]),b,c1);
        c2=_mm512_fmadd_ps(_mm512_set1_ps(a[2]),b,c2); c3=_mm512_fmadd_ps(_mm512_set1_ps(a[3]),b,c3);
        c4=_mm512_fmadd_ps(_mm512_set1_ps(a[4]),b,c4); c5=_mm512_fmadd_ps(_mm512_set1_ps(a[5]),b,c5);
        c6=_mm512_fmadd_ps(_mm512_set1_ps(a[6]),b,c6); c7=_mm512_fmadd_ps(_mm512_set1_ps(a[7]),b,c7);
    }
    _mm512_storeu_ps(C+0*ldc,c0);_mm512_storeu_ps(C+1*ldc,c1);_mm512_storeu_ps(C+2*ldc,c2);_mm512_storeu_ps(C+3*ldc,c3);
    _mm512_storeu_ps(C+4*ldc,c4);_mm512_storeu_ps(C+5*ldc,c5);_mm512_storeu_ps(C+6*ldc,c6);_mm512_storeu_ps(C+7*ldc,c7);
}
static void gemm_packed(const float *A, const float *B, float *C, int M, int K, int N) {
    /* clean-size demo: dims divisible by the block config (asserted in main) */
    memset(C, 0, (size_t)M*N*sizeof(float));           /* v3 accumulates into C */
    float *Ap = _mm_malloc((size_t)MC*KC*sizeof(float), 64);
    float *Bp = _mm_malloc((size_t)KC*NC*sizeof(float), 64);
    for (int jc = 0; jc < N; jc += NC)
        for (int pc = 0; pc < K; pc += KC) {
            pack_B(B, Bp, pc, jc, KC, NC, N);           /* pack once, reuse over all M */
            for (int ic = 0; ic < M; ic += MC) {
                pack_A(A, Ap, ic, pc, MC, KC, K);
                for (int jr = 0; jr < NC; jr += NR) {
                    const float *Bpan = Bp + (size_t)(jr/NR)*KC*NR;
                    for (int ir = 0; ir < MC; ir += MR) {
                        const float *Apan = Ap + (size_t)(ir/MR)*KC*MR;
                        micro_packed(Apan, Bpan, C + (size_t)(ic+ir)*N + (jc+jr), N, KC);
                    }
                }
            }
        }
    _mm_free(Ap); _mm_free(Bp);
}

/* ================= v3-MT: packed + blocked, multithreaded ================= *
 * Persistent parallel region. One thread packs the shared B panel (omp single,
 * implicit barrier); all threads then split the M-tile loop (omp for, implicit
 * barrier) with a PRIVATE A panel each. Threads own disjoint C row-blocks, and
 * the serial pc loop + implicit barriers make the K-accumulation race-free.    */
static void gemm_packed_mt(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, (size_t)M*N*sizeof(float));
    float *Bp = _mm_malloc((size_t)KC*NC*sizeof(float), 64);   /* shared */
#pragma omp parallel
    {
        float *Ap = _mm_malloc((size_t)MC*KC*sizeof(float), 64); /* per-thread */
        for (int jc = 0; jc < N; jc += NC)
            for (int pc = 0; pc < K; pc += KC) {
#pragma omp single
                { pack_B(B, Bp, pc, jc, KC, NC, N); }          /* implicit barrier */
#pragma omp for schedule(dynamic)
                for (int ic = 0; ic < M; ic += MC) {
                    pack_A(A, Ap, ic, pc, MC, KC, K);
                    for (int jr = 0; jr < NC; jr += NR) {
                        const float *Bpan = Bp + (size_t)(jr/NR)*KC*NR;
                        for (int ir = 0; ir < MC; ir += MR) {
                            const float *Apan = Ap + (size_t)(ir/MR)*KC*MR;
                            micro_packed(Apan, Bpan, C + (size_t)(ic+ir)*N + (jc+jr), N, KC);
                        }
                    }
                }                                               /* implicit barrier */
            }
        _mm_free(Ap);
    }
    _mm_free(Bp);
}

/* ================= measured memory roof (STREAM triad) ================= */
static double measure_bandwidth_gbs(void) {
    long n = 16L*1024*1024;                    /* 64 MB/array -> far exceeds L3 */
    float *a=_mm_malloc(n*4,64),*b=_mm_malloc(n*4,64),*c=_mm_malloc(n*4,64);
#pragma omp parallel for schedule(static)
    for (long i=0;i<n;i++){ b[i]=1.0f; c[i]=2.0f; a[i]=0.0f; }
    float s=3.0f; double best=1e30;
    for (int r=0;r<5;r++){
        double t0=now();
#pragma omp parallel for schedule(static)
        for (long i=0;i<n;i++) a[i]=b[i]+s*c[i];
        double dt=now()-t0; if(dt<best)best=dt;
    }
    volatile float sink=a[n/2]; (void)sink;
    _mm_free(a);_mm_free(b);_mm_free(c);
    return (3.0*n*4.0)/best/1e9;                /* read b, read c, write a */
}

/* ============================================================================
 * v4 : INT8 GEMM via AVX-512 VNNI (vpdpbusd = 64 uint8*int8 MACs / instruction)
 * ----------------------------------------------------------------------------
 * activations -> uint8 (per-tensor symmetric, +128 offset for VNNI's u8*s8),
 * weights     -> int8  (per-output-column symmetric).  With Aq = Au-128:
 *     dot_true = sum(Aq*Bq) = VNNI(Au,Bq) - 128*colsum(Bq)     (offset fixup)
 *     C[m,n]   = a_scale * b_scale[n] * dot_true
 * Weights are packed once into VNNI "K-by-4 interleaved" layout (offline in
 * real inference); we time only the kernel + requantize. INT8 is LOSSY, so the
 * metric is RELATIVE error vs FP32, not bit-exactness.
 * ========================================================================== */
#define MRI 8

static float quantize_A(const float *A, uint8_t *Au, int M, int K, int Kpad) {
    float amax = 1e-8f;
    for (size_t i=0;i<(size_t)M*K;i++){ float v=fabsf(A[i]); if(v>amax)amax=v; }
    float scale = amax/127.0f, inv = 1.0f/scale;
    for (int m=0;m<M;m++)
        for (int k=0;k<Kpad;k++){
            int q=0;
            if (k<K){ q=(int)lroundf(A[(size_t)m*K+k]*inv); q=q>127?127:(q<-127?-127:q); }
            Au[(size_t)m*Kpad+k]=(uint8_t)(q+128);        /* Aq in [-127,127] -> [1,255] */
        }
    return scale;
}
/* per-column int8 quantize + pack into VNNI layout, and accumulate colsum(Bq). */
static void quantize_B_pack(const float *B, int8_t *Bp, float *bscale, int32_t *colsum,
                            int K, int N, int Kpad) {
    for (int n=0;n<N;n++){
        float bmax=1e-8f;
        for (int k=0;k<K;k++){ float v=fabsf(B[(size_t)k*N+n]); if(v>bmax)bmax=v; }
        bscale[n]=bmax/127.0f;
    }
    int kgc=Kpad/4, panels=N/16;
    for (int p=0;p<panels;p++){
        int n0=p*16; int8_t *base=Bp+(size_t)p*kgc*64;
        for (int i=0;i<16;i++){                            /* column n0+i */
            float inv=1.0f/bscale[n0+i]; int s=0;
            for (int kg=0;kg<kgc;kg++)
                for (int j=0;j<4;j++){                      /* byte (i*4+j) of k-group kg */
                    int kk=kg*4+j, q=0;
                    if (kk<K){ q=(int)lroundf(B[(size_t)kk*N+n0+i]*inv); q=q>127?127:(q<-127?-127:q); }
                    base[(size_t)kg*64 + i*4 + j]=(int8_t)q; s+=q;
                }
            colsum[n0+i]=s;
        }
    }
}
static inline void vnni_micro(const uint8_t *Au, int lda, const int8_t *Bp, int kgc,
                              const int32_t *corr, const float *dq, float *C, int ldc) {
    __m512i acc[MRI];
    for (int m=0;m<MRI;m++) acc[m]=_mm512_setzero_si512();
    for (int kg=0;kg<kgc;kg++){
        __m512i b=_mm512_loadu_si512((const void*)(Bp+(size_t)kg*64));
        for (int m=0;m<MRI;m++){
            int a4; memcpy(&a4, Au+(size_t)m*lda+kg*4, 4);  /* 4 uint8 acts, broadcast */
            acc[m]=_mm512_dpbusd_epi32(acc[m], _mm512_set1_epi32(a4), b);
        }
    }
    __m512i corrv=_mm512_loadu_si512((const void*)corr);    /* 128*colsum for these 16 cols */
    __m512  dqv =_mm512_loadu_ps(dq);                       /* a_scale*b_scale for these 16 cols */
    for (int m=0;m<MRI;m++){
        __m512i v=_mm512_sub_epi32(acc[m], corrv);
        _mm512_storeu_ps(C+(size_t)m*ldc, _mm512_mul_ps(_mm512_cvtepi32_ps(v), dqv));
    }
}
static void gemm_int8_kernel(const uint8_t *Au, int Kpad, const int8_t *Bp,
                             const int32_t *corr, const float *dq, float *C, int M, int N) {
    int kgc=Kpad/4, panels=N/16;
    for (int i0=0;i0<M;i0+=MRI)
        for (int p=0;p<panels;p++)
            vnni_micro(Au+(size_t)i0*Kpad, Kpad, Bp+(size_t)p*kgc*64, kgc,
                       corr+(size_t)p*16, dq+(size_t)p*16, C+(size_t)i0*N+p*16, N);
}
/* measured INT8 compute roof: 16 independent dpbusd chains. */
static double measure_peak_int8_gops(void) {
    __m512i acc[16]; for(int i=0;i<16;i++) acc[i]=_mm512_setzero_si512();
    __m512i a=_mm512_set1_epi32(0x01010101), b=_mm512_set1_epi32(0x01010101);
    const long iters=30000000L; double t0=now();
    for (long it=0;it<iters;it++){
        acc[0]=_mm512_dpbusd_epi32(acc[0],a,b);acc[1]=_mm512_dpbusd_epi32(acc[1],a,b);acc[2]=_mm512_dpbusd_epi32(acc[2],a,b);acc[3]=_mm512_dpbusd_epi32(acc[3],a,b);
        acc[4]=_mm512_dpbusd_epi32(acc[4],a,b);acc[5]=_mm512_dpbusd_epi32(acc[5],a,b);acc[6]=_mm512_dpbusd_epi32(acc[6],a,b);acc[7]=_mm512_dpbusd_epi32(acc[7],a,b);
        acc[8]=_mm512_dpbusd_epi32(acc[8],a,b);acc[9]=_mm512_dpbusd_epi32(acc[9],a,b);acc[10]=_mm512_dpbusd_epi32(acc[10],a,b);acc[11]=_mm512_dpbusd_epi32(acc[11],a,b);
        acc[12]=_mm512_dpbusd_epi32(acc[12],a,b);acc[13]=_mm512_dpbusd_epi32(acc[13],a,b);acc[14]=_mm512_dpbusd_epi32(acc[14],a,b);acc[15]=_mm512_dpbusd_epi32(acc[15],a,b);
    }
    double t1=now();
    __m512i s=_mm512_setzero_si512(); for(int i=0;i<16;i++) s=_mm512_add_epi32(s,acc[i]);
    volatile int sink=_mm512_reduce_add_epi32(s); (void)sink;
    return (double)iters*16.0*64.0*2.0/(t1-t0)/1e9;         /* 16 chains * 64 MAC * 2 */
}
static double rms_rel(const float *X, const float *R, long n) {
    double se=0.0, sr=0.0;
    for (long i=0;i<n;i++){ double d=(double)X[i]-R[i]; se+=d*d; sr+=(double)R[i]*R[i]; }
    return sqrt(se/(sr+1e-30));
}

/* ================= measured compute roof ================= */
static double measure_peak_gflops(void) {
    __m512 acc[16];
    for (int i=0;i<16;i++) acc[i]=_mm512_set1_ps((float)i*0.01f+0.5f);
    __m512 a=_mm512_set1_ps(1.0000001f), b=_mm512_set1_ps(0.9999999f);
    const long iters = 30000000L;
    double t0=now();
    for (long it=0; it<iters; it++) {
        acc[0]=_mm512_fmadd_ps(a,b,acc[0]);acc[1]=_mm512_fmadd_ps(a,b,acc[1]);acc[2]=_mm512_fmadd_ps(a,b,acc[2]);acc[3]=_mm512_fmadd_ps(a,b,acc[3]);
        acc[4]=_mm512_fmadd_ps(a,b,acc[4]);acc[5]=_mm512_fmadd_ps(a,b,acc[5]);acc[6]=_mm512_fmadd_ps(a,b,acc[6]);acc[7]=_mm512_fmadd_ps(a,b,acc[7]);
        acc[8]=_mm512_fmadd_ps(a,b,acc[8]);acc[9]=_mm512_fmadd_ps(a,b,acc[9]);acc[10]=_mm512_fmadd_ps(a,b,acc[10]);acc[11]=_mm512_fmadd_ps(a,b,acc[11]);
        acc[12]=_mm512_fmadd_ps(a,b,acc[12]);acc[13]=_mm512_fmadd_ps(a,b,acc[13]);acc[14]=_mm512_fmadd_ps(a,b,acc[14]);acc[15]=_mm512_fmadd_ps(a,b,acc[15]);
    }
    double t1=now();
    __m512 s=_mm512_setzero_ps(); for (int i=0;i<16;i++) s=_mm512_add_ps(s,acc[i]);
    volatile float sink=_mm512_reduce_add_ps(s); (void)sink;
    return (double)iters*16.0*16.0*2.0 / (t1-t0) / 1e9;
}

/* ================= harness ================= */
typedef void (*gemm_fn)(const float*,const float*,float*,int,int,int);
static double bench(gemm_fn fn, const float *A, const float *B, float *C,
                    int M, int K, int N, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now(); fn(A, B, C, M, K, N); double dt = now()-t0;
        if (dt < best) best = dt;
    }
    return (2.0*M*N*K)/best/1e9;
}

static void run_shape(int M, int K, int N, double peak, double peak_i8, int run_naive) {
    /* v3 demo requires dims divisible by the block config */
    if (M%MC || K%KC || N%NC || MC%MR || NC%NR) { printf("shape %dx%dx%d skipped (indivisible)\n",M,K,N); return; }
    float *A=_mm_malloc((size_t)M*K*4,64),*B=_mm_malloc((size_t)K*N*4,64);
    float *C=_mm_malloc((size_t)M*N*4,64),*Cref=_mm_malloc((size_t)M*N*4,64);
    fill_rand(A,(size_t)M*K); fill_rand(B,(size_t)K*N);
    gemm_ikj(A,B,Cref,M,K,N);                            /* trusted reference */

    double gB = (K==N && (size_t)K*N*4 > 12*1024*1024) ? 16.0 : 16.0; (void)gB;
    printf("--- shape M=%d K=%d N=%d   (B matrix = %.1f MB, L3 = 12 MB) ---\n",
           M, K, N, (double)K*N*4/1e6);
    if (run_naive) {
        double g = bench(gemm_naive, A,B,C, M,K,N, 1);
        printf("  %-20s %7.1f GF/s  %4.0f%%\n", "v0 naive", g, 100*g/peak);
    }
    double g1=bench(gemm_ikj,   A,B,C,M,K,N,6);
    double g2=bench(gemm_micro, A,B,C,M,K,N,6); double d2=max_abs_diff(C,Cref,(size_t)M*N);
    double g3=bench(gemm_packed,A,B,C,M,K,N,6); double d3=max_abs_diff(C,Cref,(size_t)M*N);
    omp_set_num_threads(4);
    double g34=bench(gemm_packed_mt,A,B,C,M,K,N,6); double d34=max_abs_diff(C,Cref,(size_t)M*N);
    omp_set_num_threads(8);
    double g38=bench(gemm_packed_mt,A,B,C,M,K,N,6); double d38=max_abs_diff(C,Cref,(size_t)M*N);
    omp_set_num_threads(1);
    printf("  %-22s %7.1f GF/s  %4.0f%%\n", "v1 i-k-j (train.c)", g1, 100*g1/peak);
    printf("  %-22s %7.1f GF/s  %4.0f%%   |diff| %.1e\n", "v2 microkernel",  g2, 100*g2/peak, d2);
    printf("  %-22s %7.1f GF/s  %4.0f%%   |diff| %.1e\n", "v3 packed (1 thread)",g3, 100*g3/peak, d3);
    printf("  %-22s %7.1f GF/s  %4.1fx/1T   |diff| %.1e\n", "v3-MT (4 threads)", g34, g34/g3, d34);
    printf("  %-22s %7.1f GF/s  %4.1fx/1T   |diff| %.1e\n", "v3-MT (8 threads)", g38, g38/g3, d38);

    /* ---- v4 INT8 VNNI (weights packed once offline; time kernel+requant) ---- */
    int Kpad = (K+3)&~3;
    uint8_t *Au = _mm_malloc((size_t)M*Kpad, 64);
    int8_t  *Bp = _mm_malloc((size_t)(N/16)*(Kpad/4)*64, 64);
    float   *bscale = malloc((size_t)N*sizeof(float));
    int32_t *colsum = malloc((size_t)N*sizeof(int32_t));
    int32_t *corr = _mm_malloc((size_t)N*sizeof(int32_t), 64);
    float   *dq   = _mm_malloc((size_t)N*sizeof(float), 64);
    quantize_B_pack(B, Bp, bscale, colsum, K, N, Kpad);
    float a_scale = quantize_A(A, Au, M, K, Kpad);
    for (int n=0;n<N;n++){ corr[n]=128*colsum[n]; dq[n]=a_scale*bscale[n]; }
    double bi = 1e30;
    for (int r=0;r<6;r++){ double t0=now(); gemm_int8_kernel(Au,Kpad,Bp,corr,dq,C,M,N); double dt=now()-t0; if(dt<bi)bi=dt; }
    double gi = (2.0*M*N*K)/bi/1e9;
    double rel = rms_rel(C, Cref, (size_t)M*N);
    printf("  %-22s %7.1f GOP/s %4.0f%%   rel.err %.1e  (%.1fx vs v3-1T)\n",
           "v4 INT8 VNNI (1T)", gi, 100*gi/peak_i8, rel, gi/g3);
    _mm_free(Au);_mm_free(Bp);free(bscale);free(colsum);_mm_free(corr);_mm_free(dq);

#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    double gb1=bench(gemm_blas,A,B,C,M,K,N,6); double dbo=max_abs_diff(C,Cref,(size_t)M*N);
    openblas_set_num_threads(8);
    double gb8=bench(gemm_blas,A,B,C,M,K,N,6);
    openblas_set_num_threads(1);
    printf("  %-22s %7.1f GF/s  %4.0f%%   |diff| %.1e\n","v5 OpenBLAS (1T)",gb1,100*gb1/peak,dbo);
    printf("  %-22s %7.1f GF/s\n","v5 OpenBLAS (8T)",gb8);
    printf("  -> v3 1T = %.0f%% of BLAS-1T,  v3-MT 8T = %.0f%% of BLAS-8T\n",100*g3/gb1,100*g38/gb8);
#endif

    printf("\n");
    _mm_free(A);_mm_free(B);_mm_free(C);_mm_free(Cref);
}

int main(void) {
    srand(1234);
    int maxt = omp_get_max_threads();
    double peak   = measure_peak_gflops();       /* 1-core AVX-512 FP32 FMA peak */
    double peak_i8= measure_peak_int8_gops();     /* 1-core VNNI INT8 peak        */
    double bw     = measure_bandwidth_gbs();       /* aggregate DRAM triad        */
    printf("=== roofline anchors ===\n");
    printf("  compute roof FP32 (1 core, FMA):    %.1f GFLOP/s\n", peak);
    printf("  compute roof INT8 (1 core, VNNI):   %.1f GOP/s  (%.1fx FP32)\n", peak_i8, peak_i8/peak);
    printf("  memory  roof (STREAM triad, %d thr): %.1f GB/s\n", maxt, bw);
    printf("  (single-thread %%peak is vs the matching 1-core roof)\n\n");
    run_shape(1024, 1024, 1024, peak, peak_i8, 1);   /* headline: B (4MB) fits in L3 */
    run_shape(1024, 1024, 4096, peak, peak_i8, 0);   /* FFN-up: B (16MB) EXCEEDS L3  */
    return 0;
}
