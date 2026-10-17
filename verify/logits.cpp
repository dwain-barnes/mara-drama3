// Dump the last-position logits of a GGUF model for a list of token ids.
// usage: logits model.gguf tokens.txt out.bin [n_gpu_layers]
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf tokens.txt out.bin [n_gpu_layers]\n", argv[0]);
        return 1;
    }

    std::vector<llama_token> toks;
    std::ifstream in(argv[2]);
    for (long long t; in >> t;) {
        toks.push_back((llama_token) t);
    }
    if (toks.empty() || toks.size() > 512) {
        fprintf(stderr, "need between 1 and 512 tokens, got %zu\n", toks.size());
        return 1;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 4 ? atoi(argv[4]) : 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    int n = (int) toks.size();
    llama_batch batch = llama_batch_init(512, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == n - 1;
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * lg = llama_get_logits_ith(ctx, n - 1);

    std::ofstream out(argv[3], std::ios::binary);
    out.write((const char *) lg, sizeof(float) * nv);

    std::vector<int> idx(nv);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    printf("n_vocab %d\n", nv);
    for (int i = 0; i < 10; i++) {
        printf("%d %.6f\n", idx[i], lg[idx[i]]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
