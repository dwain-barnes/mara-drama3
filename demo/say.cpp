// Loads a Drama 3 backbone and mmproj once, then renders speech for lines read from stdin:
//
//   say <tab> out.wav <tab> text    ->  ok <tab> out.wav <tab> frames <tab> ms   or   err <tab> message
//   stream <tab> text               ->  pcm <tab> n, then n float32 samples (44.1 kHz mono), once per decoded chunk,
//                                       then  ok <tab> frames <tab> ms to the first chunk <tab> total ms   or   err <tab> message
//   ref <tab> a.codes[;b.codes]     ->  ok   (the transcript of each is in the .txt of the same name)
//   reset                           ->  ok   (forgets the earlier turns)
//   quit
//
// stdout is in binary mode, so every reply line ends in \n only, also on Windows.
//
// usage: say backbone.gguf mmproj.gguf [--ref a.codes[;b.codes]] [--seed N] [--history-frames N] [-c N] [--n-cpu-moe N]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

struct turn {
    std::string          text;
    std::vector<int32_t> codes; // 10 per frame
};

// the pipeline only says why it refused an input in its log, so the last error is kept for the reply
static std::string last_err;

static void log_cb(ggml_log_level level, const char * text, void *) {
    static ggml_log_level prev = GGML_LOG_LEVEL_INFO;
    if (level == GGML_LOG_LEVEL_CONT) {
        level = prev;
    }
    prev = level;
    if (level == GGML_LOG_LEVEL_ERROR) {
        last_err = text;
        last_err.erase(last_err.find_last_not_of("\r\n") + 1);
        const char * prefix = "mtmd_helper_gen_audio: ";
        if (last_err.rfind(prefix, 0) == 0) {
            last_err.erase(0, strlen(prefix));
        }
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// load_turns from tools/tts/tts.cpp
static bool load_turns(const std::string & paths, std::vector<turn> & turns) {
    std::stringstream ss(paths);
    std::string path;
    while (std::getline(ss, path, ';')) {
        if (path.empty()) {
            continue;
        }
        const size_t ext = path.size() >= 6 ? path.size() - 6 : 0;
        if (path.compare(ext, std::string::npos, ".codes") != 0) {
            last_err = path + " is not a .codes file";
            return false;
        }
        std::ifstream fc(path, std::ios::binary);
        std::ifstream ft(path.substr(0, ext) + ".txt", std::ios::binary);
        if (!fc || !ft) {
            last_err = "cannot open " + path + " or its .txt";
            return false;
        }
        std::string raw((std::istreambuf_iterator<char>(fc)), std::istreambuf_iterator<char>());
        if (raw.empty() || raw.size() % (10 * sizeof(int32_t)) != 0) {
            last_err = path + " does not hold whole frames of 10 codes";
            return false;
        }
        turn t;
        t.codes.resize(raw.size() / sizeof(int32_t));
        memcpy(t.codes.data(), raw.data(), raw.size());

        t.text.assign((std::istreambuf_iterator<char>(ft)), std::istreambuf_iterator<char>());
        t.text.erase(t.text.find_last_not_of(" \t\r\n") + 1);
        turns.push_back(std::move(t));
    }
    return true;
}

template <typename T>
static std::vector<mtmd_helper_gen_audio_codes> codes_of(const T & turns) {
    std::vector<mtmd_helper_gen_audio_codes> out;
    for (const turn & t : turns) {
        out.push_back({ t.text.c_str(), t.codes.data(), (int32_t) (t.codes.size() / 10) });
    }
    return out;
}

static void reply(const std::string & line) {
    fputs((line + "\n").c_str(), stdout);
    fflush(stdout);
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 3) {
        fprintf(stderr, "usage: %s backbone.gguf mmproj.gguf [--ref a.codes[;b.codes]] [--seed N] [--history-frames N] [-c N] [--n-cpu-moe N]\n", argv[0]);
        return 1;
    }
    std::string ref_paths;
    uint32_t seed = UINT32_MAX;
    int max_hist = 128;
    int n_ctx = 4096;
    int n_cpu_moe = 0;
    for (int i = 3; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (!strcmp(argv[i], "--ref") && more) {
            ref_paths = argv[++i];
        } else if (!strcmp(argv[i], "--seed") && more) {
            seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (!strcmp(argv[i], "--history-frames") && more) {
            max_hist = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-c") && more) {
            n_ctx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-cpu-moe") && more) {
            n_cpu_moe = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    llama_log_set(log_cb, nullptr);
    mtmd_helper_log_set(log_cb, nullptr);

    std::vector<turn> refs;
    if (!load_turns(ref_paths, refs)) {
        fprintf(stderr, "%s\n", last_err.c_str());
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    // same as --n-cpu-moe in common/arg.cpp: the routed experts of the first N layers stay in host memory
    std::vector<std::string> patterns;
    for (int i = 0; i < n_cpu_moe; i++) {
        patterns.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate)_exps");
    }
    std::vector<llama_model_tensor_buft_override> overrides;
    for (const auto & p : patterns) {
        overrides.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
    }
    overrides.push_back({ nullptr, nullptr });
    mp.tensor_buft_overrides = overrides.data();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.embeddings = true; // the tower reads the backbone's hidden states
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        fprintf(stderr, "failed to create the llama context\n");
        return 1;
    }
    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.print_timings = false;
    mtmd_context * mctx = mtmd_init_from_file(argv[2], model, mcp);
    if (!mctx) {
        fprintf(stderr, "failed to load %s\n", argv[2]);
        return 1;
    }
    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);

    std::deque<turn> history;
    int hist_frames = 0;

    // an empty path streams the audio on stdout as it is decoded instead of writing a file
    auto say = [&](const std::string & path, const std::string & text) -> std::string {
        if (text.empty()) {
            return "err\tempty text";
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        };
        long long first_ms = -1;
        auto send = [&] {
            const float * pcm;
            size_t n;
            mtmd_helper_gen_audio_take_pcm(gen, &pcm, &n);
            if (n == 0) {
                return;
            }
            if (first_ms < 0) {
                first_ms = elapsed();
            }
            printf("pcm\t%zu\n", n);
            fwrite(pcm, sizeof(float), n, stdout);
            fflush(stdout);
        };
        llama_memory_clear(llama_get_memory(lctx), true);

        const auto ref_inp  = codes_of(refs);
        const auto hist_inp = codes_of(history);
        mtmd_helper_gen_audio_inp inp{};
        inp.prompt     = text.c_str();
        inp.prompt_len = text.size();
        inp.seed       = seed;
        inp.out_type   = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
        inp.refs       = ref_inp.data();
        inp.n_refs     = (int32_t) ref_inp.size();
        inp.history    = hist_inp.data();
        inp.n_history  = (int32_t) hist_inp.size();

        last_err.clear();
        if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) {
            return "err\t" + (last_err.empty() ? std::string("set_input failed") : last_err);
        }
        for (int32_t left = 1; left > 0;) {
            left = mtmd_helper_gen_audio_step_prompt(gen, 128);
            if (left < 0) {
                return "err\tprompt processing failed";
            }
        }

        const float * h = llama_get_embeddings_ith(lctx, -1);
        bool stop = false;
        for (int i = 0; !stop && i < 1024; i++) {
            const float * next = nullptr;
            if (mtmd_helper_gen_audio_step_gen(gen, LLAMA_TOKEN_NULL, h, &next, &stop) != 0) {
                return "err\tgeneration failed at frame " + std::to_string(i);
            }
            if (path.empty()) {
                send();
            }
            if (!next) {
                break;
            }
            h = next;
        }

        int32_t rate;
        const char * data;
        size_t len;
        // get_output decodes the frames since the last chunk, so the stream's tail comes after it
        if (mtmd_helper_gen_audio_get_output(gen, &rate, &data, &len, nullptr) != 0) {
            return "err\tget_output failed";
        }
        if (path.empty()) {
            send();
        } else {
            FILE * f = fopen(path.c_str(), "wb");
            if (!f) {
                return "err\tcannot open " + path;
            }
            const bool written = fwrite(data, 1, len, f) == len;
            if (fclose(f) != 0 || !written) {
                return "err\tcannot write " + path;
            }
        }

        const int32_t * codes;
        int32_t n_frames;
        mtmd_helper_gen_audio_get_codes(gen, &codes, &n_frames);
        // a turn without frames would make the pipeline refuse every later prompt
        if (n_frames > 0) {
            history.push_back({ text, std::vector<int32_t>(codes, codes + (size_t) n_frames * 10) });
            hist_frames += n_frames;
            while (history.size() > 1 && hist_frames > max_hist) {
                hist_frames -= (int) history.front().codes.size() / 10;
                history.pop_front();
            }
        }

        if (path.empty()) {
            return "ok\t" + std::to_string(n_frames) + "\t" + std::to_string(first_ms) + "\t" + std::to_string(elapsed());
        }
        return "ok\t" + path + "\t" + std::to_string(n_frames) + "\t" + std::to_string(elapsed());
    };

    reply("ready");

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t tab = line.find('\t');
        const std::string cmd = line.substr(0, tab);
        const std::string arg = tab == std::string::npos ? "" : line.substr(tab + 1);

        if (cmd == "quit") {
            break;
        } else if (cmd == "say") {
            const size_t tab2 = arg.find('\t');
            if (tab2 == std::string::npos || tab2 == 0) {
                reply("err\texpected say<tab>out.wav<tab>text");
                continue;
            }
            reply(say(arg.substr(0, tab2), arg.substr(tab2 + 1)));
        } else if (cmd == "stream") {
            reply(say("", arg));
        } else if (cmd == "ref") {
            std::vector<turn> next;
            if (!load_turns(arg, next)) {
                reply("err\t" + last_err);
                continue;
            }
            refs = std::move(next);
            reply("ok");
        } else if (cmd == "reset") {
            history.clear();
            hist_frames = 0;
            reply("ok");
        } else {
            reply("err\tunknown command " + cmd);
        }
    }

    mtmd_helper_gen_audio_free(gen);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
