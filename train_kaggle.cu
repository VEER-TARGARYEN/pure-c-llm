/*
 * train_kaggle.cu  --  343M GPT training orchestrator for Kaggle T4 (CUDA).
 *
 * Mirrors the CPU train.c (verified to convergence) op-for-op, on the GPU:
 * pre-norm blocks, LayerNorm+GELU(tanh), learned positions, tied wte head,
 * AdamW. All weights live in ONE flat device arena laid out in the exact CKP1
 * registry order, so checkpoints are directly loadable by the local
 * generate.exe on the user's PC.
 *
 * Build (Kaggle):  nvcc -O3 -arch=sm_75 --use_fast_math \
 *                       -o train_gpu train_kaggle.cu gpu_kernels.cu
 * Run:             ./train_gpu [max_steps]
 *
 * Data:      train.bin = raw uint16 tokens, headerless, memory-mapped (mmap),
 *            streamed sequentially with wraparound.
 * Resume:    if checkpoint_latest.bin exists (searched in /kaggle/working and
 *            /kaggle/input/ * ), training resumes from the exact step + data
 *            cursor. Checkpoints are written ATOMICALLY (tmp file + rename)
 *            every CKPT_EVERY steps AND every CKPT_SECS seconds, plus a final
 *            save before the session-time guard exits (Kaggle kills at 9h; we
 *            stop ourselves at 8.4h and save).
 * Formats:   checkpoint_latest.bin = 'CKG1' full state (weights + Adam m,v +
 *            step + data cursor).  model_ckp1.bin = weights-only 'CKP1' for
 *            the local C inference stack.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include "gpu_kernels.h"

/* ---- hyperparameters ---- */
#define LR_MAX      3e-4f
#define WARMUP      200
#define BETA1       0.9f
#define BETA2       0.999f
#define ADAM_EPS    1e-8f
#define WDECAY      0.01f
#define CKPT_EVERY  5000          /* steps  */
#define CKPT_SECS   1800          /* and every 30 min, whichever first */
#define SESSION_SECS 30240        /* 8.4 h: save + exit before Kaggle's 9h kill */
#define LOG_EVERY   10

#define CKG_MAGIC 0x434B4731      /* 'CKG1' full training state   */
#define CKP_MAGIC 0x434B5031      /* 'CKP1' weights-only (local)  */

/* ============================================================
 * Parameter registry: one flat arena, CKP1 order.
 * ============================================================ */
typedef struct { long off, n; } Slice;

static Slice reg_[2 + N_LAYERS * 16 + 2];
static int   nreg_ = 0;
static long  total_params_ = 0;

static long reg_add(long n) {
    reg_[nreg_].off = total_params_;
    reg_[nreg_].n   = n;
    nreg_++;
    total_params_ += n;
    return total_params_ - n;
}

/* offsets (into the arena) for every tensor, filled by build_registry() */
static long o_wte, o_wpe, o_gf, o_bf;
static long o_g1[N_LAYERS], o_b1[N_LAYERS], o_Wq[N_LAYERS], o_bq[N_LAYERS],
            o_Wk[N_LAYERS], o_bk[N_LAYERS], o_Wv[N_LAYERS], o_bv[N_LAYERS],
            o_Wo[N_LAYERS], o_bo[N_LAYERS], o_g2[N_LAYERS], o_b2[N_LAYERS],
            o_W1[N_LAYERS], o_bf1[N_LAYERS], o_W2[N_LAYERS], o_bf2[N_LAYERS];

static void build_registry(void) {
    o_wte = reg_add((long)VOCAB_SIZE * DIM);
    o_wpe = reg_add((long)MAX_SEQ_LEN * DIM);
    for (int l = 0; l < N_LAYERS; l++) {
        o_g1[l] = reg_add(DIM);            o_b1[l]  = reg_add(DIM);
        o_Wq[l] = reg_add((long)DIM*DIM);  o_bq[l]  = reg_add(DIM);
        o_Wk[l] = reg_add((long)DIM*DIM);  o_bk[l]  = reg_add(DIM);
        o_Wv[l] = reg_add((long)DIM*DIM);  o_bv[l]  = reg_add(DIM);
        o_Wo[l] = reg_add((long)DIM*DIM);  o_bo[l]  = reg_add(DIM);
        o_g2[l] = reg_add(DIM);            o_b2[l]  = reg_add(DIM);
        o_W1[l] = reg_add((long)DIM*D_FF); o_bf1[l] = reg_add(D_FF);
        o_W2[l] = reg_add((long)D_FF*DIM); o_bf2[l] = reg_add(DIM);
    }
    o_gf = reg_add(DIM);
    o_bf = reg_add(DIM);
}

/* device arenas */
static float *d_w, *d_g, *d_m, *d_v;
#define W(o) (d_w + (o))
#define G(o) (d_g + (o))

/* ============================================================
 * Activations (stored for backward, mirroring train.c's A_* buffers)
 * ============================================================ */
static float *a_xin[N_LAYERS + 1];
static float *a_ln1[N_LAYERS], *a_m1[N_LAYERS], *a_r1[N_LAYERS];
static float *a_q[N_LAYERS], *a_k[N_LAYERS], *a_v[N_LAYERS];
static float *a_att[N_LAYERS], *a_probs[N_LAYERS], *a_xmid[N_LAYERS];
static float *a_ln2[N_LAYERS], *a_m2[N_LAYERS], *a_r2[N_LAYERS];
static float *a_h[N_LAYERS], *a_hg[N_LAYERS];
static float *a_lnf, *a_mf, *a_rf, *a_logits;
static float *s_proj, *s_ff, *s_dx, *s_dlnf, *s_dln, *s_dq, *s_dk, *s_dv,
             *s_datt, *s_dh, *s_dhg, *s_dlogits, *s_Dbuf, *d_loss;
static int   *d_x, *d_y;

static float *dmalloc(long nfloats) {
    float *p;
    CUDA_CHECK(cudaMalloc(&p, (size_t)nfloats * sizeof(float)));
    return p;
}

static void alloc_acts(void) {
    long MD = (long)M_ROWS * DIM, MF = (long)M_ROWS * D_FF;
    long MV = (long)M_ROWS * VOCAB_SIZE;
    long PB = (long)B_SIZE * N_HEADS * T_SIZE * T_SIZE;
    for (int l = 0; l <= N_LAYERS; l++) a_xin[l] = dmalloc(MD);
    for (int l = 0; l < N_LAYERS; l++) {
        a_ln1[l] = dmalloc(MD); a_m1[l] = dmalloc(M_ROWS); a_r1[l] = dmalloc(M_ROWS);
        a_q[l] = dmalloc(MD); a_k[l] = dmalloc(MD); a_v[l] = dmalloc(MD);
        a_att[l] = dmalloc(MD); a_probs[l] = dmalloc(PB); a_xmid[l] = dmalloc(MD);
        a_ln2[l] = dmalloc(MD); a_m2[l] = dmalloc(M_ROWS); a_r2[l] = dmalloc(M_ROWS);
        a_h[l] = dmalloc(MF); a_hg[l] = dmalloc(MF);
    }
    a_lnf = dmalloc(MD); a_mf = dmalloc(M_ROWS); a_rf = dmalloc(M_ROWS);
    a_logits = dmalloc(MV);
    s_proj = dmalloc(MD); s_ff = dmalloc(MD); s_dx = dmalloc(MD);
    s_dlnf = dmalloc(MD); s_dln = dmalloc(MD);
    s_dq = dmalloc(MD); s_dk = dmalloc(MD); s_dv = dmalloc(MD); s_datt = dmalloc(MD);
    s_dh = dmalloc(MF); s_dhg = dmalloc(MF); s_dlogits = dmalloc(MV);
    s_Dbuf = dmalloc((long)B_SIZE * N_HEADS * T_SIZE * T_SIZE);  /* attn-bwd dP/dS workspace */
    d_loss = dmalloc(1);
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)M_ROWS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)M_ROWS * sizeof(int)));
}

/* ============================================================
 * Forward + backward (exact GPU port of train.c's sequences)
 * ============================================================ */
static void forward(void) {
    const int M = M_ROWS;
    emb_fwd(a_xin[0], W(o_wte), W(o_wpe), d_x);
    for (int l = 0; l < N_LAYERS; l++) {
        ln_fwd(a_xin[l], W(o_g1[l]), W(o_b1[l]), a_ln1[l], a_m1[l], a_r1[l], M);
        gemm_nn(a_ln1[l], W(o_Wq[l]), W(o_bq[l]), a_q[l], M, DIM, DIM);
        gemm_nn(a_ln1[l], W(o_Wk[l]), W(o_bk[l]), a_k[l], M, DIM, DIM);
        gemm_nn(a_ln1[l], W(o_Wv[l]), W(o_bv[l]), a_v[l], M, DIM, DIM);
        attn_fwd(a_q[l], a_k[l], a_v[l], a_att[l], a_probs[l]);
        gemm_nn(a_att[l], W(o_Wo[l]), W(o_bo[l]), s_proj, M, DIM, DIM);
        vec_add(a_xmid[l], a_xin[l], s_proj, (long)M * DIM);
        ln_fwd(a_xmid[l], W(o_g2[l]), W(o_b2[l]), a_ln2[l], a_m2[l], a_r2[l], M);
        gemm_nn(a_ln2[l], W(o_W1[l]), W(o_bf1[l]), a_h[l], M, DIM, D_FF);
        gelu_fwd(a_hg[l], a_h[l], (long)M * D_FF);
        gemm_nn(a_hg[l], W(o_W2[l]), W(o_bf2[l]), s_ff, M, D_FF, DIM);
        vec_add(a_xin[l + 1], a_xmid[l], s_ff, (long)M * DIM);
    }
    ln_fwd(a_xin[N_LAYERS], W(o_gf), W(o_bf), a_lnf, a_mf, a_rf, M);
    gemm_nt(a_lnf, W(o_wte), a_logits, M, DIM, VOCAB_SIZE, 0);   /* tied head */
}

static void backward(void) {
    const int M = M_ROWS;
    long MD = (long)M * DIM, MF = (long)M * D_FF;

    /* tied head: d_lnf = dlogits @ wte ; dwte += dlogits^T @ lnf */
    gemm_nn(s_dlogits, W(o_wte), NULL, s_dlnf, M, VOCAB_SIZE, DIM);
    gemm_tn(s_dlogits, a_lnf, G(o_wte), M, VOCAB_SIZE, DIM);

    CUDA_CHECK(cudaMemset(s_dx, 0, MD * sizeof(float)));
    ln_bwd_dx(s_dx, s_dlnf, a_xin[N_LAYERS], W(o_gf), a_mf, a_rf, M);
    ln_bwd_dgb(G(o_gf), G(o_bf), s_dlnf, a_xin[N_LAYERS], a_mf, a_rf, M);

    for (int l = N_LAYERS - 1; l >= 0; l--) {
        /* FFN backward (s_dx = grad of x_out) */
        CUDA_CHECK(cudaMemset(s_dhg, 0, MF * sizeof(float)));
        gemm_nt(s_dx, W(o_W2[l]), s_dhg, M, DIM, D_FF, 1);
        gemm_tn(a_hg[l], s_dx, G(o_W2[l]), M, D_FF, DIM);
        bias_bwd(G(o_bf2[l]), s_dx, M, DIM);
        gelu_bwd(s_dh, s_dhg, a_h[l], MF);
        CUDA_CHECK(cudaMemset(s_dln, 0, MD * sizeof(float)));
        gemm_nt(s_dh, W(o_W1[l]), s_dln, M, D_FF, DIM, 1);
        gemm_tn(a_ln2[l], s_dh, G(o_W1[l]), M, DIM, D_FF);
        bias_bwd(G(o_bf1[l]), s_dh, M, D_FF);
        ln_bwd_dx(s_dx, s_dln, a_xmid[l], W(o_g2[l]), a_m2[l], a_r2[l], M);
        ln_bwd_dgb(G(o_g2[l]), G(o_b2[l]), s_dln, a_xmid[l], a_m2[l], a_r2[l], M);

        /* attention backward (s_dx = grad of x_mid) */
        CUDA_CHECK(cudaMemset(s_datt, 0, MD * sizeof(float)));
        gemm_nt(s_dx, W(o_Wo[l]), s_datt, M, DIM, DIM, 1);
        gemm_tn(a_att[l], s_dx, G(o_Wo[l]), M, DIM, DIM);
        bias_bwd(G(o_bo[l]), s_dx, M, DIM);
        attn_bwd(s_dq, s_dk, s_dv, s_Dbuf, s_datt, a_q[l], a_k[l], a_v[l], a_probs[l]);
        CUDA_CHECK(cudaMemset(s_dln, 0, MD * sizeof(float)));
        gemm_nt(s_dq, W(o_Wq[l]), s_dln, M, DIM, DIM, 1);
        gemm_tn(a_ln1[l], s_dq, G(o_Wq[l]), M, DIM, DIM);
        bias_bwd(G(o_bq[l]), s_dq, M, DIM);
        gemm_nt(s_dk, W(o_Wk[l]), s_dln, M, DIM, DIM, 1);
        gemm_tn(a_ln1[l], s_dk, G(o_Wk[l]), M, DIM, DIM);
        bias_bwd(G(o_bk[l]), s_dk, M, DIM);
        gemm_nt(s_dv, W(o_Wv[l]), s_dln, M, DIM, DIM, 1);
        gemm_tn(a_ln1[l], s_dv, G(o_Wv[l]), M, DIM, DIM);
        bias_bwd(G(o_bv[l]), s_dv, M, DIM);
        ln_bwd_dx(s_dx, s_dln, a_xin[l], W(o_g1[l]), a_m1[l], a_r1[l], M);
        ln_bwd_dgb(G(o_g1[l]), G(o_b1[l]), s_dln, a_xin[l], a_m1[l], a_r1[l], M);
    }
    emb_bwd(G(o_wte), G(o_wpe), s_dx, d_x);
}

/* ============================================================
 * Weight init (GPT-2-style) -- host-side, then upload once
 * ============================================================ */
static unsigned rng_ = 1234u;
static float frand(float s) {
    rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5;
    return ((float)(rng_ & 0xFFFFFFu) / (float)0x1000000 * 2.0f - 1.0f) * s;
}

static void init_weights(void) {
    float *hw = (float *)malloc((size_t)total_params_ * sizeof(float));
    if (!hw) { fprintf(stderr, "host OOM\n"); exit(1); }
    memset(hw, 0, (size_t)total_params_ * sizeof(float));   /* biases/betas = 0 */
    for (long i = 0; i < (long)VOCAB_SIZE * DIM; i++)  hw[o_wte + i] = frand(0.02f);
    for (long i = 0; i < (long)MAX_SEQ_LEN * DIM; i++) hw[o_wpe + i] = frand(0.02f);
    for (int l = 0; l < N_LAYERS; l++) {
        for (long i = 0; i < (long)DIM*DIM; i++) {
            hw[o_Wq[l]+i] = frand(0.02f); hw[o_Wk[l]+i] = frand(0.02f);
            hw[o_Wv[l]+i] = frand(0.02f); hw[o_Wo[l]+i] = frand(0.02f);
        }
        for (long i = 0; i < (long)DIM*D_FF; i++) { hw[o_W1[l]+i] = frand(0.02f); hw[o_W2[l]+i] = frand(0.02f); }
        for (int i = 0; i < DIM; i++) { hw[o_g1[l]+i] = 1.0f; hw[o_g2[l]+i] = 1.0f; }
    }
    for (int i = 0; i < DIM; i++) hw[o_gf + i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_w, hw, (size_t)total_params_ * sizeof(float), cudaMemcpyHostToDevice));
    free(hw);
}

/* ============================================================
 * Checkpointing (atomic: write .tmp then rename)
 * ============================================================ */
static const char *CKPT_PATH = "/kaggle/working/checkpoint_latest.bin";
static const char *CKP1_PATH = "/kaggle/working/model_ckp1.bin";

static void write_arena(FILE *f, const float *dptr, long n, float *staging) {
    CUDA_CHECK(cudaMemcpy(staging, dptr, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(staging, sizeof(float), (size_t)n, f);
}

void save_checkpoint(int step, float loss, long cursor, const char *filename) {
    float *staging = (float *)malloc((size_t)total_params_ * sizeof(float));
    if (!staging) { fprintf(stderr, "host OOM (ckpt)\n"); return; }
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s.tmp", filename);

    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", tmp); free(staging); return; }
    int hdr[8] = { CKG_MAGIC, MAX_SEQ_LEN, DIM, N_HEADS, D_FF, N_LAYERS, VOCAB_SIZE, step };
    fwrite(hdr, sizeof(int), 8, f);
    fwrite(&cursor, sizeof(long), 1, f);
    fwrite(&loss, sizeof(float), 1, f);
    write_arena(f, d_w, total_params_, staging);
    write_arena(f, d_m, total_params_, staging);
    write_arena(f, d_v, total_params_, staging);
    fclose(f);
    rename(tmp, filename);                                   /* atomic swap */

    /* weights-only CKP1 for the local C inference stack */
    snprintf(tmp, sizeof(tmp), "%s.tmp", CKP1_PATH);
    f = fopen(tmp, "wb");
    if (f) {
        int h1[8] = { CKP_MAGIC, MAX_SEQ_LEN, DIM, N_HEADS, D_FF, N_LAYERS, VOCAB_SIZE, step };
        fwrite(h1, sizeof(int), 8, f);
        write_arena(f, d_w, total_params_, staging);
        fclose(f);
        rename(tmp, CKP1_PATH);
    }
    free(staging);
    printf("  [checkpoint @ step %d, loss %.4f -> %s]\n", step, loss, filename);
    fflush(stdout);
}

/* returns start step; fills *cursor. Searches working dir then Kaggle inputs. */
static int try_resume(long *cursor) {
    const char *cands[3] = { CKPT_PATH, "checkpoint_latest.bin", NULL };
    char inputbuf[512];
    FILE *probe = popen("ls /kaggle/input/*/checkpoint_latest.bin 2>/dev/null | head -1", "r");
    if (probe) {
        if (fgets(inputbuf, sizeof(inputbuf), probe)) {
            inputbuf[strcspn(inputbuf, "\n")] = 0;
            cands[2] = inputbuf;
        }
        pclose(probe);
    }
    for (int c = 0; c < 3; c++) {
        if (!cands[c]) continue;
        FILE *f = fopen(cands[c], "rb");
        if (!f) continue;
        int hdr[8];
        if (fread(hdr, sizeof(int), 8, f) != 8 || hdr[0] != CKG_MAGIC ||
            hdr[1] != MAX_SEQ_LEN || hdr[2] != DIM || hdr[3] != N_HEADS ||
            hdr[4] != D_FF || hdr[5] != N_LAYERS || hdr[6] != VOCAB_SIZE) {
            fprintf(stderr, "skipping %s (config mismatch)\n", cands[c]);
            fclose(f); continue;
        }
        float lastloss;
        if (fread(cursor, sizeof(long), 1, f) != 1) { fclose(f); continue; }
        if (fread(&lastloss, sizeof(float), 1, f) != 1) { fclose(f); continue; }
        float *staging = (float *)malloc((size_t)total_params_ * sizeof(float));
        long n = total_params_;
        int ok = 1;
        float *dsts[3] = { d_w, d_m, d_v };
        for (int a = 0; a < 3 && ok; a++) {
            if (fread(staging, sizeof(float), (size_t)n, f) != (size_t)n) ok = 0;
            else CUDA_CHECK(cudaMemcpy(dsts[a], staging, (size_t)n * sizeof(float),
                                       cudaMemcpyHostToDevice));
        }
        free(staging); fclose(f);
        if (!ok) { fprintf(stderr, "short read in %s\n", cands[c]); continue; }
        printf("RESUMED from %s: step %d, loss %.4f, data cursor %ld\n",
               cands[c], hdr[7], lastloss, *cursor);
        return hdr[7];
    }
    return 0;
}

/* ============================================================
 * Data: mmap'd uint16 token stream, sequential with wraparound
 * ============================================================ */
static const uint16_t *map_tokens(const char *path, long *n_tokens) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    struct stat st; fstat(fd, &st);
    const uint16_t *p = (const uint16_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    madvise((void *)p, st.st_size, MADV_SEQUENTIAL);
    *n_tokens = st.st_size / (long)sizeof(uint16_t);
    return p;
}

int main(int argc, char **argv) {
    long max_steps = (argc > 1) ? atol(argv[1]) : 1000000000L;

    gpu_init();
    build_registry();
    printf("343M CUDA trainer | dim=%d heads=%d ff=%d layers=%d vocab=%d | B=%d T=%d\n",
           DIM, N_HEADS, D_FF, N_LAYERS, VOCAB_SIZE, B_SIZE, T_SIZE);
    printf("params: %.2fM (%.2f GB fp32)\n", total_params_ / 1e6,
           total_params_ * 4.0 / 1e9);

    long n_tokens;
    const uint16_t *toks = map_tokens("train.bin", &n_tokens);
    printf("train.bin: %ld tokens (mmap'd)\n", n_tokens);
    if (n_tokens < (long)M_ROWS + T_SIZE + 1) { fprintf(stderr, "dataset too small\n"); return 1; }

    d_w = dmalloc(total_params_); d_g = dmalloc(total_params_);
    d_m = dmalloc(total_params_); d_v = dmalloc(total_params_);
    CUDA_CHECK(cudaMemset(d_g, 0, (size_t)total_params_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m, 0, (size_t)total_params_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, (size_t)total_params_ * sizeof(float)));
    alloc_acts();

    size_t freeb, totalb;
    cudaMemGetInfo(&freeb, &totalb);
    printf("GPU memory: %.1f GB used / %.1f GB\n", (totalb - freeb) / 1e9, totalb / 1e9);

    long cursor = 0;
    int step0 = try_resume(&cursor);
    if (step0 == 0) { printf("fresh start: initializing weights\n"); init_weights(); }

    int *hx = (int *)malloc((size_t)M_ROWS * sizeof(int));
    int *hy = (int *)malloc((size_t)M_ROWS * sizeof(int));

    time_t t_start = time(NULL), t_last_ckpt = t_start;
    long tokens_done = (long)step0 * M_ROWS;
    float loss_h = 0.0f;
    double ema_ms = 0.0;

    for (long step = step0 + 1; step <= max_steps; step++) {
        /* ---- next sequential batch from the mmap'd stream ---- */
        if (cursor + (long)M_ROWS + 1 >= n_tokens) cursor = 0;   /* wraparound */
        for (int b = 0; b < B_SIZE; b++)
            for (int t = 0; t < T_SIZE; t++) {
                long idx = cursor + (long)b * T_SIZE + t;
                hx[b * T_SIZE + t] = toks[idx]     % VOCAB_SIZE;
                hy[b * T_SIZE + t] = toks[idx + 1] % VOCAB_SIZE;
            }
        cursor += (long)M_ROWS;
        CUDA_CHECK(cudaMemcpy(d_x, hx, (size_t)M_ROWS * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, hy, (size_t)M_ROWS * sizeof(int), cudaMemcpyHostToDevice));

        /* ---- lr schedule: linear warmup then constant ---- */
        float lr = (step < WARMUP) ? LR_MAX * (float)step / WARMUP : LR_MAX;

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        int profile = (step % 100 == 1);   /* phase timing every 100 steps */
        struct timespec p0, p1, p2;
        if (profile) { CUDA_CHECK(cudaDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &p0); }

        forward();
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        ce_fwd_bwd(a_logits, d_y, s_dlogits, d_loss);

        if (profile) { CUDA_CHECK(cudaDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &p1); }

        backward();
        adamw_step(d_w, d_g, d_m, d_v, total_params_, lr, BETA1, BETA2, ADAM_EPS, WDECAY, (int)step);

        if (profile) {
            CUDA_CHECK(cudaDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &p2);
            double fms = (p1.tv_sec - p0.tv_sec) * 1e3 + (p1.tv_nsec - p0.tv_nsec) / 1e6;
            double bms = (p2.tv_sec - p1.tv_sec) * 1e3 + (p2.tv_nsec - p1.tv_nsec) / 1e6;
            printf("  [profile: forward %.0f ms | backward+opt %.0f ms]\n", fms, bms);
        }

        CUDA_CHECK(cudaMemcpy(&loss_h, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double ms = (ts1.tv_sec - ts0.tv_sec) * 1e3 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        ema_ms = (ema_ms == 0.0) ? ms : 0.9 * ema_ms + 0.1 * ms;
        tokens_done += M_ROWS;

        if (step % LOG_EVERY == 0) {
            printf("step %6ld | loss %7.4f | %7.0f ms | %6.0f tok/s | %ld tokens total\n",
                   step, loss_h, ema_ms, M_ROWS / (ema_ms / 1e3), tokens_done);
            fflush(stdout);
        }

        /* ---- checkpoint triggers ---- */
        time_t now = time(NULL);
        int step_trigger = (step % CKPT_EVERY == 0);
        int time_trigger = (now - t_last_ckpt >= CKPT_SECS);
        int session_over = (now - t_start >= SESSION_SECS);
        if (step_trigger || time_trigger || session_over) {
            save_checkpoint((int)step, loss_h, cursor, CKPT_PATH);
            t_last_ckpt = now;
            if (session_over) {
                printf("session time budget reached (%.1f h): saved and exiting cleanly.\n",
                       (now - t_start) / 3600.0);
                return 0;
            }
        }
    }

    save_checkpoint((int)max_steps, loss_h, cursor, CKPT_PATH);
    printf("done.\n");
    return 0;
}
