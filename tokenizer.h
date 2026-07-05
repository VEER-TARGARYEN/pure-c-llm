/*
 * tokenizer.h  --  Public API for the Piece 1 BPE tokenizer, so other
 *                  translation units (e.g. main.c) can encode/decode without
 *                  pulling in the CLI's main().
 *
 * The tokenizer's own command-line front-end (train/encode/decode) still lives
 * in tokenizer.c and is compiled in by DEFAULT. When linking the tokenizer into
 * another program, compile tokenizer.c with -DTOKENIZER_LIB to drop its main().
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

/* One BPE merge rule. The k-th merge implicitly creates token id (256 + k). */
typedef struct {
    int left;
    int right;
} Merge;

/* Load merges from a vocab.bin. Returns a malloc'd array (caller frees) and
 * writes the total vocab size and merge count through the out-pointers. */
Merge *load_vocab(const char *path, int *out_vocab_size, int *out_num_merges);

/* Encode raw bytes into token ids. Returns a malloc'd array (caller frees);
 * writes the token count to *out_n. */
int *encode(const unsigned char *text, long n,
            Merge *merges, int num_merges, int *out_n);

/* Build the byte expansion of every token so ids can be decoded to text.
 * Allocates token_bytes[vocab_size] and token_len[vocab_size] (caller frees
 * via free_token_bytes). */
void build_token_bytes(int vocab_size, Merge *merges, int num_merges,
                       unsigned char ***out_bytes, int **out_lens);

void free_token_bytes(unsigned char **token_bytes, int *token_len, int vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
