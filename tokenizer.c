/*
 * tokenizer.c  --  A minimal Byte-Pair Encoding (BPE) tokenizer in pure C.
 *
 * Piece 1 of a from-scratch, zero-dependency next-word-prediction LLM.
 *
 * Only the C standard library is used: stdio.h, stdlib.h, string.h, limits.h.
 *
 * --------------------------------------------------------------------------
 * WHAT IS BPE, IN ONE PARAGRAPH
 * --------------------------------------------------------------------------
 * We start by treating the text as a sequence of raw bytes. There are 256
 * possible byte values, so our starting ("base") vocabulary is tokens 0..255,
 * where token i simply means "the byte i". We then repeatedly look at every
 * adjacent pair of tokens in the corpus, find the pair that occurs most often,
 * and "merge" it into a brand-new token. The first merge creates token 256,
 * the second creates token 257, and so on. We stop once we have as many tokens
 * as we asked for (e.g. 2048). The list of merges IS the vocabulary: from it we
 * can encode new text and decode token IDs back to bytes.
 *
 * --------------------------------------------------------------------------
 * USAGE
 * --------------------------------------------------------------------------
 *   ./tokenizer train  <input.txt> <vocab_size> <vocab.bin>
 *   ./tokenizer encode <vocab.bin> "<text>" [out_tokens.bin]
 *   ./tokenizer decode <vocab.bin> <tokens.bin>
 * --------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>   /* INT_MAX */

#include "tokenizer.h"  /* Merge type + public function prototypes */

/* Magic number written at the start of vocab.bin so we can sanity-check files. */
#define VOCAB_MAGIC 0x42504531  /* the bytes 'B' 'P' 'E' '1' */

/* ==========================================================================
 * SMALL FILE HELPERS
 * ========================================================================== */

/* Read an entire file into a freshly malloc'd buffer. Caller must free(). */
static unsigned char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); exit(1); }

    fseek(f, 0, SEEK_END);
    long n = ftell(f);          /* number of bytes in the file */
    fseek(f, 0, SEEK_SET);

    unsigned char *buf = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (!buf) { fprintf(stderr, "error: out of memory reading '%s'\n", path); exit(1); }

    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "error: short read on '%s'\n", path); exit(1);
    }
    fclose(f);

    *out_len = n;
    return buf;
}

/* ==========================================================================
 * VOCAB FILE I/O  (vocab.bin)
 * --------------------------------------------------------------------------
 * Format (all little/native-endian ints; fine because we only read it back on
 * the same kind of machine that wrote it):
 *
 *     int  magic        = VOCAB_MAGIC
 *     int  vocab_size   = 256 + num_merges
 *     int  num_merges
 *     Merge merges[num_merges]   (each is two ints: left, right)
 *
 * The 256 base byte-tokens are implicit, so we only need to store the merges.
 * ========================================================================== */

static void save_vocab(const char *path, int vocab_size, Merge *merges, int num_merges) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "error: cannot write '%s'\n", path); exit(1); }

    int magic = VOCAB_MAGIC;
    fwrite(&magic,      sizeof(int),   1,          f);
    fwrite(&vocab_size, sizeof(int),   1,          f);
    fwrite(&num_merges, sizeof(int),   1,          f);
    fwrite(merges,      sizeof(Merge), num_merges, f);

    fclose(f);
    printf("saved vocab: %d tokens (%d merges) -> %s\n", vocab_size, num_merges, path);
}

/* Load merges back from disk. Caller must free() the returned array. */
Merge *load_vocab(const char *path, int *out_vocab_size, int *out_num_merges) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); exit(1); }

    int magic = 0, vocab_size = 0, num_merges = 0;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != VOCAB_MAGIC) {
        fprintf(stderr, "error: '%s' is not a valid vocab file\n", path); exit(1);
    }
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&num_merges, sizeof(int), 1, f);

    Merge *merges = (Merge *)malloc((num_merges > 0 ? num_merges : 1) * sizeof(Merge));
    if (!merges) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    fread(merges, sizeof(Merge), num_merges, f);
    fclose(f);

    *out_vocab_size = vocab_size;
    *out_num_merges = num_merges;
    return merges;
}

/* ==========================================================================
 * EXPANDING TOKENS BACK TO BYTES
 * --------------------------------------------------------------------------
 * To decode, every token ID needs to know the exact byte string it stands for.
 * Base tokens 0..255 expand to a single byte. A merged token expands to the
 * concatenation of its two children's byte strings. Because every merge only
 * references tokens created earlier, we can build these in increasing ID order
 * and each child is guaranteed to already be built.
 *
 * token_bytes[id] -> pointer to the bytes; token_len[id] -> how many bytes.
 * ========================================================================== */

void build_token_bytes(int vocab_size, Merge *merges, int num_merges,
                       unsigned char ***out_bytes, int **out_lens) {
    unsigned char **token_bytes = (unsigned char **)malloc(vocab_size * sizeof(unsigned char *));
    int            *token_len   = (int *)           malloc(vocab_size * sizeof(int));
    if (!token_bytes || !token_len) { fprintf(stderr, "error: out of memory\n"); exit(1); }

    /* Base vocabulary: one byte each. */
    for (int i = 0; i < 256; i++) {
        token_bytes[i] = (unsigned char *)malloc(1);
        token_bytes[i][0] = (unsigned char)i;
        token_len[i] = 1;
    }

    /* Merged tokens: child(left) bytes followed by child(right) bytes. */
    for (int k = 0; k < num_merges; k++) {
        int id = 256 + k;
        int a  = merges[k].left;
        int b  = merges[k].right;

        token_len[id]   = token_len[a] + token_len[b];
        token_bytes[id] = (unsigned char *)malloc(token_len[id]);

        memcpy(token_bytes[id],                token_bytes[a], token_len[a]);
        memcpy(token_bytes[id] + token_len[a], token_bytes[b], token_len[b]);
    }

    *out_bytes = token_bytes;
    *out_lens  = token_len;
}

void free_token_bytes(unsigned char **token_bytes, int *token_len, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) free(token_bytes[i]);
    free(token_bytes);
    free(token_len);
}

/* ==========================================================================
 * TRAINING  --  learn the merges from a corpus
 * ========================================================================== */

static void write_tokens(const char *path, int *ids, int n);

static void train(const char *input_path, int target_vocab, const char *out_path,
                  const char *tokens_out_path) {
    /* ---- 1. Load the corpus and turn each byte into a token ID. ---------- */
    long file_len;
    unsigned char *data = read_file(input_path, &file_len);

    /* The working corpus is an array of ints because token IDs grow past 255. */
    int *corpus = (int *)malloc((file_len > 0 ? file_len : 1) * sizeof(int));
    if (!corpus) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    for (long i = 0; i < file_len; i++) corpus[i] = data[i];
    long corpus_len = file_len;
    free(data);

    printf("loaded %ld bytes from %s\n", file_len, input_path);

    /* ---- 2. Allocate the pair-count table. ------------------------------ *
     * counts[a * mv + b] holds how many times token `a` is directly followed
     * by token `b`. This is a flat 2D table of size vocab_size^2. It is the
     * simplest possible "map from pair to count" -- no hash map needed.
     *
     * Memory cost: target_vocab^2 ints. For 2048 that is ~16 MB, which is fine.
     * (If you ever push target_vocab very high, switch this to a hash map.)    */
    long mv = target_vocab;                       /* row stride of the table   */
    int *counts = (int *)calloc((size_t)mv * mv, sizeof(int));
    if (!counts) { fprintf(stderr, "error: out of memory for count table\n"); exit(1); }

    /* ---- 3. Storage for the learned merge rules. ------------------------ */
    Merge *merges = (Merge *)malloc(target_vocab * sizeof(Merge));
    if (!merges) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    int num_merges = 0;
    int vocab_size = 256;                         /* start with the 256 bytes  */

    /* ---- 4. The merge loop. -------------------------------------------- */
    while (vocab_size < target_vocab && corpus_len >= 2) {

        /* (a) Count every adjacent pair in one pass over the corpus. */
        for (long i = 0; i + 1 < corpus_len; i++)
            counts[(long)corpus[i] * mv + corpus[i + 1]]++;

        /* (b) Find the most frequent pair. We scan the corpus again and look
         *     each pair's count up; tracking the running maximum. */
        int best_a = -1, best_b = -1, best_count = 0;
        for (long i = 0; i + 1 < corpus_len; i++) {
            int a = corpus[i], b = corpus[i + 1];
            int c = counts[(long)a * mv + b];
            if (c > best_count) { best_count = c; best_a = a; best_b = b; }
        }

        /* (c) Reset the table back to zero -- but only the cells we touched,
         *     so this stays O(corpus) instead of O(vocab_size^2). We must do
         *     this using the CURRENT corpus, before we rewrite it in step (e). */
        for (long i = 0; i + 1 < corpus_len; i++)
            counts[(long)corpus[i] * mv + corpus[i + 1]] = 0;

        /* (d) If the best pair never repeats, there is nothing useful left to
         *     merge -- stop early even if we are below the target size. */
        if (best_count < 2) {
            printf("no repeated pairs left; stopping at %d tokens\n", vocab_size);
            break;
        }

        /* Record the new merge. The new token's ID is implicitly 256+num_merges. */
        int new_id = vocab_size;
        merges[num_merges].left  = best_a;
        merges[num_merges].right = best_b;
        num_merges++;
        vocab_size++;

        /* (e) Rewrite the corpus, replacing every (best_a, best_b) with new_id.
         *     We compact in place: the write index `w` never gets ahead of the
         *     read index `i`, so this is safe. */
        long w = 0;
        for (long i = 0; i < corpus_len; ) {
            if (i + 1 < corpus_len && corpus[i] == best_a && corpus[i + 1] == best_b) {
                corpus[w++] = new_id;
                i += 2;
            } else {
                corpus[w++] = corpus[i++];
            }
        }
        corpus_len = w;

        /* Progress: show the first few merges in detail, then a periodic tick. */
        if (num_merges <= 10 || num_merges % 200 == 0) {
            printf("merge %4d: (%d, %d) -> %d   count=%d   corpus_len=%ld\n",
                   num_merges, best_a, best_b, new_id, best_count, corpus_len);
        }
    }

    /* ---- 5. Save and clean up. ----------------------------------------- */
    save_vocab(out_path, vocab_size, merges, num_merges);

    /* The corpus array is already fully BPE-encoded (encoding happened as a
     * side effect of learning the merges), so we can dump the model's training
     * data for free -- no separate, slow encode pass needed. */
    if (tokens_out_path) {
        write_tokens(tokens_out_path, corpus, (int)corpus_len);
        printf("wrote %ld training tokens -> %s\n", corpus_len, tokens_out_path);
    }

    free(corpus);
    free(counts);
    free(merges);
}

/* ==========================================================================
 * ENCODING  --  text  ->  token IDs
 * --------------------------------------------------------------------------
 * Greedy BPE: start from raw bytes, then repeatedly apply the EARLIEST-learned
 * merge that currently appears. "Earliest-learned" == lowest token ID == lowest
 * rank, which guarantees the same tokenization the training process would give.
 * ========================================================================== */

/* Return the new-token ID for pair (a,b) if it is a known merge, else INT_MAX.
 * A linear scan is plenty fast for the short strings a typing assistant sees. */
static int pair_rank(int a, int b, Merge *merges, int num_merges) {
    for (int k = 0; k < num_merges; k++)
        if (merges[k].left == a && merges[k].right == b)
            return 256 + k;
    return INT_MAX;
}

/* Returns a malloc'd array of token IDs; writes the count to *out_n. */
int *encode(const unsigned char *text, long n,
            Merge *merges, int num_merges, int *out_n) {
    /* Begin with one token per raw byte. */
    int *seq = (int *)malloc((n > 0 ? n : 1) * sizeof(int));
    if (!seq) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    for (long i = 0; i < n; i++) seq[i] = text[i];
    long len = n;

    while (len >= 2) {
        /* Find the adjacent pair whose merge was learned earliest. */
        int  best_rank = INT_MAX;
        long best_i    = -1;
        for (long i = 0; i + 1 < len; i++) {
            int r = pair_rank(seq[i], seq[i + 1], merges, num_merges);
            if (r < best_rank) { best_rank = r; best_i = i; }
        }

        /* No adjacent pair is a known merge -> we are done. */
        if (best_rank == INT_MAX) break;

        /* Apply that merge everywhere it occurs, then loop and re-scan. */
        int a = seq[best_i], b = seq[best_i + 1], new_id = best_rank;
        long w = 0;
        for (long i = 0; i < len; ) {
            if (i + 1 < len && seq[i] == a && seq[i + 1] == b) {
                seq[w++] = new_id;
                i += 2;
            } else {
                seq[w++] = seq[i++];
            }
        }
        len = w;
    }

    *out_n = (int)len;
    return seq;
}

/* ==========================================================================
 * DECODING  --  token IDs  ->  text
 * ========================================================================== */

static void decode_to_stdout(int *ids, int n, unsigned char **token_bytes, int *token_len,
                             int vocab_size) {
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= vocab_size) {
            fprintf(stderr, "error: token id %d out of range\n", id); exit(1);
        }
        fwrite(token_bytes[id], 1, token_len[id], stdout);
    }
}

/* ==========================================================================
 * TOKEN-LIST FILE I/O  (tokens.bin)
 * --------------------------------------------------------------------------
 * A trivial container so encode and decode can hand data to each other (and so
 * the model in later pieces can read token streams). Format: int count, then
 * that many ints.
 * ========================================================================== */

static void write_tokens(const char *path, int *ids, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "error: cannot write '%s'\n", path); exit(1); }
    fwrite(&n,  sizeof(int), 1, f);
    fwrite(ids, sizeof(int), n, f);
    fclose(f);
}

static int *read_tokens(const char *path, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); exit(1); }
    int n = 0;
    fread(&n, sizeof(int), 1, f);
    int *ids = (int *)malloc((n > 0 ? n : 1) * sizeof(int));
    if (!ids) { fprintf(stderr, "error: out of memory\n"); exit(1); }
    fread(ids, sizeof(int), n, f);
    fclose(f);
    *out_n = n;
    return ids;
}

/* ==========================================================================
 * MAIN  --  command-line dispatch
 * ========================================================================== */

/* The CLI front-end below is compiled in by default. Define TOKENIZER_LIB when
 * linking tokenizer.c into another program (e.g. main.c) to drop this main(). */
#ifndef TOKENIZER_LIB

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  tokenizer train  <input.txt> <vocab_size> <vocab.bin>\n"
        "  tokenizer encode <vocab.bin> \"<text>\" [out_tokens.bin]\n"
        "  tokenizer decode <vocab.bin> <tokens.bin>\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    /* ----- train ----- */
    if (strcmp(argv[1], "train") == 0) {
        if (argc < 5) { usage(); return 1; }
        const char *input  = argv[2];
        int         target = atoi(argv[3]);
        const char *out    = argv[4];
        if (target <= 256) {
            fprintf(stderr, "error: vocab_size must be > 256 (you gave %d)\n", target);
            return 1;
        }
        train(input, target, out, (argc >= 6) ? argv[5] : NULL);
        return 0;
    }

    /* ----- encode ----- */
    if (strcmp(argv[1], "encode") == 0) {
        if (argc < 4) { usage(); return 1; }
        const char *vocab_path = argv[2];
        const char *text       = argv[3];
        const char *out_tokens = (argc >= 5) ? argv[4] : NULL;

        int vocab_size, num_merges;
        Merge *merges = load_vocab(vocab_path, &vocab_size, &num_merges);

        int  n_ids;
        int *ids = encode((const unsigned char *)text, (long)strlen(text),
                          merges, num_merges, &n_ids);

        /* Print the IDs for the human watching. */
        printf("%d tokens:", n_ids);
        for (int i = 0; i < n_ids; i++) printf(" %d", ids[i]);
        printf("\n");

        if (out_tokens) {
            write_tokens(out_tokens, ids, n_ids);
            printf("wrote %d tokens -> %s\n", n_ids, out_tokens);
        }

        free(ids);
        free(merges);
        return 0;
    }

    /* ----- decode ----- */
    if (strcmp(argv[1], "decode") == 0) {
        if (argc < 4) { usage(); return 1; }
        const char *vocab_path  = argv[2];
        const char *tokens_path = argv[3];

        int vocab_size, num_merges;
        Merge *merges = load_vocab(vocab_path, &vocab_size, &num_merges);

        unsigned char **token_bytes;
        int            *token_len;
        build_token_bytes(vocab_size, merges, num_merges, &token_bytes, &token_len);

        int  n_ids;
        int *ids = read_tokens(tokens_path, &n_ids);

        decode_to_stdout(ids, n_ids, token_bytes, token_len, vocab_size);
        fprintf(stderr, "\n[decoded %d tokens]\n", n_ids);

        free(ids);
        free_token_bytes(token_bytes, token_len, vocab_size);
        free(merges);
        return 0;
    }

    usage();
    return 1;
}

#endif /* TOKENIZER_LIB */
