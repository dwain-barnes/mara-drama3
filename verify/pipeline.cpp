// Run the llama-tts drama3 pipeline (mtmd-helper-gen) and check it against ref_tower.py: the prompt ids, the
// backbone hidden state of the last prompt row, and that a fixed seed repeats a run. The greedy codes are
// compared with the reference frames too, but only printed: they part at the first near tie and stay apart.
//
// With --prefix the reference comes from ref_prompt.py instead (a prompt with reference and history audio): every
// prompt row's hidden state is compared, and the prefill head logits are dumped for compare_prompt.py.
//
// usage: pipeline backbone.gguf tower.gguf outdir [--prefix ref_prompt] [--ref f.codes]... [--history f.codes]...
//                 [--text "..."], run from D:\drama3\verify
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// eos and the semantic band of the backbone vocab
static const llama_token tok_eos = 248046, sem_begin = 248099, sem_end = 252194;

struct turn {
    std::string          text;
    std::vector<int32_t> codes;
};

static void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static double cosine(const float * a, const float * b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb);
}

static std::string slurp(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<int> json_ints(const std::string & json, const char * key) {
    const size_t open = json.find('[', json.find(key));
    std::string list = json.substr(open + 1, json.find(']', open) - open - 1);
    std::replace(list.begin(), list.end(), ',', ' ');
    std::stringstream ss(list);
    std::vector<int> out;
    for (int v; ss >> v;) {
        out.push_back(v);
    }
    return out;
}

static turn read_turn(const std::string & path) {
    turn t;
    const std::string raw = slurp(path);
    t.codes.resize(raw.size() / sizeof(int32_t));
    memcpy(t.codes.data(), raw.data(), t.codes.size() * sizeof(int32_t));
    t.text = slurp(fs::path(path).replace_extension(".txt"));
    if (t.codes.empty() || t.codes.size() % 10 != 0) {
        fprintf(stderr, "%s: not a [T, 10] codes file\n", path.c_str());
        exit(1);
    }
    return t;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s backbone.gguf tower.gguf outdir [--prefix ref_prompt] [--ref f.codes]... "
                        "[--history f.codes]... [--text \"...\"]\n", argv[0]);
        return 1;
    }
    const fs::path outdir = argv[3];
    std::string prefix;
    std::string text = "Hello there, how are you today?";
    std::vector<turn> refs, history;
    for (int i = 4; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (!strcmp(argv[i], "--prefix") && more) {
            prefix = argv[++i];
        } else if (!strcmp(argv[i], "--ref") && more) {
            refs.push_back(read_turn(argv[++i]));
        } else if (!strcmp(argv[i], "--history") && more) {
            history.push_back(read_turn(argv[++i]));
        } else if (!strcmp(argv[i], "--text") && more) {
            text = argv[++i];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }
    const bool ref_mode = !prefix.empty();

    fs::create_directories(outdir);
    const std::string codes_path = (outdir / "pipeline.codes").string();
    set_env("MTMD_DRAMA3_CODES_OUT", codes_path);
    if (ref_mode) {
        set_env("MTMD_DRAMA3_LOGITS_OUT", (outdir / "prefill.logits").string());
    }

    std::vector<int64_t> fshape, hshape;
    auto frames = load_npy<int64_t>(ref_mode ? prefix + "_frames.npy" : "ref_tower_frames.npy", "'<i8'", fshape);
    auto ref_hidden = load_npy<float>(ref_mode ? prefix + "_hidden.npy" : "ref_tower_hidden.npy", "'<f4'", hshape);
    const int W = (int) fshape[1];
    const int n_embd = (int) hshape[1];

    std::vector<llama_token> want;
    llama_token ref_main;
    if (ref_mode) {
        const std::string meta = slurp(prefix + "_meta.json");
        for (int t : json_ints(meta, "\"prompt_ids\"")) {
            want.push_back(t);
        }
        ref_main = json_ints(meta, "\"top5_ids\"")[0];
    } else {
        std::ifstream f("prompt_tokens.txt");
        for (int t; f >> t;) {
            want.push_back(t);
        }
        ref_main = (llama_token) frames[want.size() * W];
    }
    const int P = (int) want.size();
    const int K = (int) fshape[0] - P;
    const int n_gen = ref_mode ? 6 : K;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ref_mode ? 1024 : 512;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.embeddings = true;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        fprintf(stderr, "failed to create the llama context\n");
        return 1;
    }

    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = true;
    mcp.print_timings = false;
    mtmd_context * mctx = mtmd_init_from_file(argv[2], model, mcp);
    if (!mctx) {
        fprintf(stderr, "failed to load %s\n", argv[2]);
        return 1;
    }
    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);

    auto codes_of = [](const std::vector<turn> & turns) {
        std::vector<mtmd_helper_gen_audio_codes> out;
        for (const turn & t : turns) {
            out.push_back({ t.text.c_str(), t.codes.data(), (int32_t) (t.codes.size() / 10) });
        }
        return out;
    };
    const auto ref_inp = codes_of(refs);
    const auto hist_inp = codes_of(history);

    std::vector<float> hidden((size_t) P * n_embd);
    double cos_last = 0;
    int rank_last = 0;

    // n_gen frames at most, returns codes 0..9 of each frame, empty on failure
    auto run = [&](uint32_t seed, int top_k) {
        llama_memory_clear(llama_get_memory(lctx), true);
        mtmd_helper_gen_audio_inp inp{};
        inp.prompt     = text.c_str();
        inp.prompt_len = text.size();
        inp.top_k      = top_k;
        inp.seed       = seed;
        inp.out_type   = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        inp.refs       = ref_inp.data();
        inp.n_refs     = (int32_t) ref_inp.size();
        inp.history    = hist_inp.data();
        inp.n_history  = (int32_t) hist_inp.size();
        std::vector<int32_t> codes;
        if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) {
            fprintf(stderr, "set_input failed\n");
            return codes;
        }
        int32_t left = P;
        while (left > 0) {
            const int32_t ret = mtmd_helper_gen_audio_step_prompt(gen, 128);
            if (ret < 0) {
                fprintf(stderr, "step_prompt failed\n");
                return codes;
            }
            const int32_t n = left - ret;
            if (n <= 0 || ret > left) {
                fprintf(stderr, "step_prompt left %d rows after %d\n", ret, left);
                return codes;
            }
            memcpy(hidden.data() + (size_t) (P - left) * n_embd, llama_get_embeddings(lctx), (size_t) n * n_embd * sizeof(float));
            left = ret;
        }

        const float * h = llama_get_embeddings_ith(lctx, -1);
        cos_last = cosine(ref_hidden.data() + (size_t) (P - 1) * n_embd, h, n_embd);
        const float * l = llama_get_logits_ith(lctx, -1);
        rank_last = l[tok_eos] > l[ref_main];
        for (llama_token t = sem_begin; t <= sem_end; t++) {
            rank_last += l[t] > l[ref_main];
        }

        for (int f = 0; f < n_gen; f++) {
            const float * h_next = nullptr;
            bool stop = false;
            if (mtmd_helper_gen_audio_step_gen(gen, LLAMA_TOKEN_NULL, h, &h_next, &stop) != 0) {
                fprintf(stderr, "step_gen failed at frame %d\n", f);
                return codes;
            }
            if (!h_next) {
                break;
            }
            h = h_next;
        }

        int32_t rate;
        const char * data;
        size_t len;
        if (mtmd_helper_gen_audio_get_output(gen, &rate, &data, &len, nullptr) != 0) {
            fprintf(stderr, "get_output failed\n");
            return codes;
        }
        std::ifstream cf(codes_path, std::ios::binary);
        for (int32_t c; cf.read((char *) &c, sizeof(c));) {
            codes.push_back(c);
        }
        return codes;
    };

    auto greedy = run(1, 1);
    if (greedy.empty()) {
        return 1;
    }

    const std::vector<int> side = json_ints(slurp(codes_path + ".json"), "\"prompt_tokens\"");
    const std::vector<llama_token> got(side.begin(), side.end());
    const bool ids_ok = got == want;
    printf("prompt ids: %zu, %s %s\n", got.size(), ids_ok ? "equal to" : "DIFFERENT from",
           ref_mode ? (prefix + "_meta.json").c_str() : "prompt_tokens.txt");

    printf("row %d (<|voice|>) hidden: cosine %.5f to the reference; the reference's next main token has rank %d of 4097\n",
           P - 1, cos_last, rank_last);

    bool hidden_ok = true;
    if (ref_mode) {
        std::vector<double> cos(P);
        for (int p = 0; p < P; p++) {
            cos[p] = cosine(ref_hidden.data() + (size_t) p * n_embd, hidden.data() + (size_t) p * n_embd, n_embd);
        }
        std::vector<char> audio(P);
        int a0 = P, a1 = -1;
        for (int p = 0; p < P; p++) {
            audio[p] = frames[(size_t) p * W] >= sem_begin && frames[(size_t) p * W] <= sem_end;
            if (audio[p]) {
                a0 = std::min(a0, p);
                a1 = p;
            }
        }
        if (a1 < 0) {
            fprintf(stderr, "no audio rows in %s_frames.npy\n", prefix.c_str());
            return 1;
        }

        int min_all = 0, min_audio = a0;
        for (int p = 0; p < P; p++) {
            if (cos[p] < cos[min_all]) {
                min_all = p;
            }
            if (audio[p] && cos[p] < cos[min_audio]) {
                min_audio = p;
            }
        }
        // single rows can come out different in every implementation (row 56 of the neutral_long prompt, a text row
        // before any audio, sits at 0.75 even with the bf16 backbone while its neighbours agree to 0.999: an expert
        // routing near a tie), so the rule is the rows this work adds and the last row, with the low rows counted
        int n_low = 0;
        for (int p = 0; p < P; p++) {
            n_low += cos[p] < 0.99;
        }
        hidden_ok = cos[P - 1] >= 0.99 && cos[a0] >= 0.99 && cos[a1 + 1] >= 0.99;
        printf("hidden cosine, %d rows (audio rows %d..%d): min %.5f at row %d; min over the audio rows %.5f at row %d; %d rows below 0.99\n",
               P, a0, a1, cos[min_all], min_all, cos[min_audio], min_audio, n_low);
        printf("  first 3 audio rows %d..%d: %.5f %.5f %.5f\n", a0, a0 + 2, cos[a0], cos[a0 + 1], cos[a0 + 2]);
        printf("  first 3 rows after the audio, %d..%d: %.5f %.5f %.5f\n", a1 + 1, a1 + 3, cos[a1 + 1], cos[a1 + 2], cos[a1 + 3]);
        printf("  last row %d: %.5f\n", P - 1, cos[P - 1]);

        std::ofstream lf(outdir / "hidden_cos.txt");
        for (int p = 0; p < P; p++) {
            char line[64];
            snprintf(line, sizeof(line), "%3d %s %.5f\n", p, audio[p] ? "audio" : "text ", cos[p]);
            lf << line;
        }
        std::ofstream hf(outdir / "hidden.bin", std::ios::binary);
        hf.write((const char *) hidden.data(), hidden.size() * sizeof(float));
    } else {
        hidden_ok = cos_last >= 0.99;
        const int n_greedy = (int) greedy.size() / 10;
        int same = 0;
        for (int f = 0; f < n_greedy && f < K; f++) {
            for (int k = 0; k < 10; k++) {
                same += greedy[f * 10 + k] == frames[(size_t) (P + f) * W + 1 + k];
            }
        }
        printf("greedy: %d frames, codes 0..9 equal to the reference frames: %d/%d\n", n_greedy, same, 10 * K);
    }

    auto a = run(42, 0);
    auto b = run(42, 0);
    auto c = run(43, 0);
    if (a.empty() || b.empty() || c.empty()) {
        return 1;
    }
    int same_ac = 0;
    for (size_t i = 0; i < a.size() && i < c.size(); i++) {
        same_ac += a[i] == c[i];
    }
    printf("sampled, seed 42 twice: %s; seed 43: %d of %zu codes equal to seed 42\n",
           a == b ? "identical" : "DIFFERENT", same_ac, a.size());

    if (!ids_ok || !hidden_ok || a != b) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    printf("ok\n");
    return 0;
}
