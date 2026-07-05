/*
 * main.c  --  Interactive autoregressive generation for the C-LLM.
 *
 * Piece 5a. Ties everything together:
 *   Piece 1 tokenizer  (text <-> token ids)
 *   Piece 2 tensor math
 *   Piece 3 transformer block (KV-cached decode path)
 *   Piece 4 full model
 *
 * Build:
 *   gcc -O3 -fopenmp -DTOKENIZER_LIB -Wno-unused-function \
 *       main.c model.c transformer.c tensor.c tokenizer.c -o llm.exe -lm
 * Run (needs a vocab.bin from Piece 1 in the current directory):
 *   .\llm.exe
 *
 * NOTE: the model runs on RANDOM weights (training is a later piece), so the
 * generated text is well-formed bytes but semantically meaningless. What this
 * demonstrates is the *machinery*: KV cache, sampling, decode, streaming.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "model.h"
#include "tokenizer.h"

/* ==========================================================================
 * SAMPLING
 * --------------------------------------------------------------------------
 * Turns a row of logits into a chosen token id. Supports:
 *   temperature : <=0 -> greedy argmax; else scale logits by 1/temperature.
 *   top_k       : keep only the k highest-probability tokens (<=0 disables).
 *   top_p       : keep the smallest set whose cumulative prob >= p (nucleus;
 *                 <=0 or >=1 disables).
 * The kept set is renormalized and sampled from.
 * ========================================================================== */

/* Number of logits to consider = the tokenizer's vocab (so every sampled id is
 * decodable). Set by main after the tokenizer is loaded. */
static int g_sample_vocab = VOCAB_SIZE;

/* qsort has no context parameter in standard C, so the comparator reads the
 * probabilities through this file-scope pointer. Set just before each qsort. */
static const float *g_sort_probs;
static int cmp_by_prob_desc(const void *a, const void *b)
{
    int ia = *(const int *)a, ib = *(const int *)b;
    float pa = g_sort_probs[ia], pb = g_sort_probs[ib];
    if (pa < pb) return 1;
    if (pa > pb) return -1;
    return 0;
}

int sample_token(float *logits, float temperature, int top_k, float top_p)
{
    const int V = g_sample_vocab;

    /* greedy decoding when temperature is (near) zero */
    if (temperature <= 1e-6f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    /* temperature-scaled, numerically-stable softmax */
    float maxl = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > maxl) maxl = logits[i];

    float *probs = (float *)malloc((size_t)V * sizeof(float));
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        float e = expf((logits[i] - maxl) / temperature);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < V; i++) probs[i] = (float)(probs[i] / sum);

    /* sort token indices by probability, descending */
    int *idx = (int *)malloc((size_t)V * sizeof(int));
    for (int i = 0; i < V; i++) idx[i] = i;
    g_sort_probs = probs;
    qsort(idx, (size_t)V, sizeof(int), cmp_by_prob_desc);

    /* how many of the sorted tokens to keep */
    int keep = V;
    if (top_k > 0 && top_k < keep) keep = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
        double cum = 0.0; int kp = 0;
        for (int i = 0; i < keep; i++) { cum += probs[idx[i]]; kp++; if (cum >= top_p) break; }
        keep = kp;
    }
    if (keep < 1) keep = 1;

    /* renormalize the kept set and sample from it */
    double kept_sum = 0.0;
    for (int i = 0; i < keep; i++) kept_sum += probs[idx[i]];

    double r = ((double)rand() / ((double)RAND_MAX + 1.0)) * kept_sum;
    double c = 0.0;
    int chosen = idx[keep - 1];
    for (int i = 0; i < keep; i++) {
        c += probs[idx[i]];
        if (r < c) { chosen = idx[i]; break; }
    }

    free(probs);
    free(idx);
    return chosen;
}

/* ==========================================================================
 * MAIN  --  interactive prompt -> generate up to 50 tokens
 * ========================================================================== */
int main(void)
{
    /* sampling hyperparameters */
    const float temperature = 0.9f;
    const int   top_k       = 40;
    const float top_p       = 0.95f;
    const int   max_new     = 50;

    /* ---- 1. tokenizer ---- */
    int tok_vocab = 0, num_merges = 0;
    Merge *merges = load_vocab("vocab.bin", &tok_vocab, &num_merges);

    unsigned char **tok_bytes;
    int            *tok_len;
    build_token_bytes(tok_vocab, merges, num_merges, &tok_bytes, &tok_len);

    g_sample_vocab = (tok_vocab < VOCAB_SIZE) ? tok_vocab : VOCAB_SIZE;
    printf("tokenizer loaded: %d tokens (sampling over %d)\n", tok_vocab, g_sample_vocab);

    /* ---- 2. model + KV cache ---- */
    printf("building dummy %d-layer model (~1.4 GB of random weights)...\n", N_LAYERS);
    GPT2Weights model;
    build_dummy_model(&model);

    KVCache cache;
    alloc_kv_cache(&cache);

    float *logits = (float *)malloc((size_t)VOCAB_SIZE * sizeof(float));

    srand((unsigned)time(NULL));

    printf("OpenMP threads = %d\n", omp_get_max_threads());
    printf("\nType a prompt and press Enter. Empty line quits.\n");
    printf("(random weights -> gibberish; this proves the pipeline, not the model)\n");

    char line[4096];
    while (1) {
        printf("\nprompt> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;   /* EOF (Ctrl+Z / Ctrl+D) */

        /* strip trailing newline / carriage return */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) break;                            /* empty line quits */

        /* encode the prompt into token ids */
        int  n_ids = 0;
        int *ids = encode((const unsigned char *)line, (long)len, merges, num_merges, &n_ids);
        if (n_ids == 0) { free(ids); continue; }

        /* echo prompt, then stream the continuation on the same line */
        printf("%s", line);
        fflush(stdout);

        /* ---- feed the prompt one token at a time, filling the KV cache ---- */
        int pos = 0;
        for (int i = 0; i < n_ids && pos < BLOCK_SIZE; i++) {
            int t = ids[i];
            if (t >= VOCAB_SIZE) t = 0;                 /* safety clamp */
            forward_gpt2(logits, t, &model, &cache, pos);
            pos++;
        }
        free(ids);

        /* ---- autoregressive generation loop ---- */
        double t0 = omp_get_wtime();
        int generated = 0;
        for (int step = 0; step < max_new && pos < BLOCK_SIZE; step++) {
            int next = sample_token(logits, temperature, top_k, top_p);

            /* decode + stream this one token */
            fwrite(tok_bytes[next], 1, (size_t)tok_len[next], stdout);
            fflush(stdout);

            /* feed it back in to get the next position's logits */
            forward_gpt2(logits, next, &model, &cache, pos);
            pos++;
            generated++;
        }
        double t1 = omp_get_wtime();

        printf("\n");
        if (generated > 0) {
            double ms = (t1 - t0) * 1000.0;
            fprintf(stderr, "[%d tokens in %.0f ms, %.1f tok/s]\n",
                    generated, ms, generated / (ms / 1000.0));
        }
    }

    printf("\nbye.\n");

    free(logits);
    free_kv_cache(&cache);
    free_model(&model);
    free_token_bytes(tok_bytes, tok_len, tok_vocab);
    free(merges);
    return 0;
}
