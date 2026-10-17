// Run the Drama 3 codec (mtmd GEN_WAV) offline and streamed on saved codes.
// usage: codec mmproj.gguf backbone.gguf outdir [--gpu] [--bench], run from D:\drama3\verify
#include "llama.h"
#include "mtmd.h"
#include "npy.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static const int N_BOOK = 10;
static const int FRAME = 2048;

static std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        exit(1);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s mmproj.gguf backbone.gguf outdir [--gpu] [--bench]\n", argv[0]);
        return 1;
    }
    const fs::path outdir = argv[3];
    bool gpu = false;
    bool bench = false;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu")) {
            gpu = true;
        } else if (!strcmp(argv[i], "--bench")) {
            bench = true;
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    std::vector<int32_t> hello;
    {
        auto raw = read_file("out/tts/hello.codes");
        hello.resize(raw.size() / sizeof(int32_t));
        memcpy(hello.data(), raw.data(), hello.size() * sizeof(int32_t));
        auto js = read_file("out/tts/hello.codes.json");
        std::string s(js.begin(), js.end());
        size_t at = s.find("\"frames\":");
        if (at == std::string::npos) {
            fprintf(stderr, "no frames field in hello.codes.json\n");
            return 1;
        }
        const size_t n = (size_t) atoi(s.c_str() + at + 9);
        if (n * N_BOOK != hello.size()) {
            fprintf(stderr, "hello.codes has %zu values, expected %zu frames of %d\n", hello.size(), n, N_BOOK);
            return 1;
        }
    }

    std::vector<int32_t> ref6;
    {
        std::vector<int64_t> shape;
        auto frames = load_npy<int64_t>("ref_tower_frames.npy", "'<i8'", shape);
        const int W = (int) shape[1];
        for (int r = 29; r <= 34; r++) {
            for (int k = 1; k <= N_BOOK; k++) {
                ref6.push_back((int32_t) frames[(size_t) r * W + k]);
            }
        }
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[2]);
        return 1;
    }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.n_threads = (int) std::thread::hardware_concurrency();
    cp.print_timings = false;
    mtmd_context * ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    if (mtmd_gen_audio_get_info(ctx).type != MTMD_GEN_AUDIO_TYPE_DRAMA3) {
        fprintf(stderr, "%s is not a drama3 mmproj\n", argv[1]);
        return 1;
    }

    // the state goes through the blob like the helper does; returns the wall time in ms
    auto decode = [&](std::vector<int32_t> & codes, int first, int n, std::vector<char> & state, std::vector<float> * pcm) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(ctx);
        inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes      = codes.data() + (size_t) first * N_BOOK;
        inp.n_codes    = (size_t) n * N_BOOK;
        inp.state_data = state.empty() ? nullptr : state.data();
        inp.state_size = state.size();

        mtmd_gen_out out{};
        auto t0 = std::chrono::steady_clock::now();
        int rc = mtmd_gen_audio_process(ctx, &inp, &out);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rc != 0) {
            fprintf(stderr, "GEN_WAV failed at frame %d (%d frames), rc %d\n", first, n, rc);
            exit(1);
        }
        if (out.n_samples != (size_t) n * FRAME) {
            fprintf(stderr, "GEN_WAV gave %zu samples for %d frames, expected %d\n", out.n_samples, n, n * FRAME);
            exit(1);
        }
        if (pcm) {
            pcm->insert(pcm->end(), out.audio, out.audio + out.n_samples);
        }
        state.assign(out.state_data, out.state_data + out.state_size);
        return ms;
    };

    if (bench) {
        std::vector<int32_t> codes(hello.begin(), hello.begin() + 8 * N_BOOK);
        auto run = [&](int n) {
            std::vector<char> state;
            for (int i = 0; i < 3; i++) {
                decode(codes, 0, n, state, nullptr);
            }
            std::vector<double> t;
            for (int i = 0; i < 20; i++) {
                t.push_back(decode(codes, 0, n, state, nullptr));
            }
            std::sort(t.begin(), t.end());
            printf("%d frames x 20: p50 %.2f ms, p99 %.2f ms, max %.2f ms (%.2f ms of audio per call)\n",
                   n, t[10], t[std::min<size_t>(19, (size_t) (0.99 * t.size()))], t.back(), n * FRAME * 1000.0 / 44100);
        };
        run(8);
        run(4);
        return 0;
    }

    fs::create_directories(outdir);
    auto write = [&](const char * name, const std::vector<float> & pcm) {
        std::ofstream f(outdir / name, std::ios::binary);
        if (!f.write((const char *) pcm.data(), pcm.size() * sizeof(float))) {
            fprintf(stderr, "cannot write %s\n", (outdir / name).string().c_str());
            exit(1);
        }
    };

    auto offline = [&](const char * name, std::vector<int32_t> & codes) {
        const int n = (int) (codes.size() / N_BOOK);
        std::vector<char> state;
        std::vector<float> pcm;
        double ms = decode(codes, 0, n, state, &pcm);
        printf("%s offline: %d frames, %zu samples, %.2f ms, state %zu bytes\n", name, n, pcm.size(), ms, state.size());
        return pcm;
    };

    auto stream = [&](const char * name, std::vector<int32_t> & codes) {
        const int n = (int) (codes.size() / N_BOOK);
        std::vector<char> state;
        std::vector<float> pcm;
        int at = 0;
        int step = 4;
        while (at < n) {
            const int m = std::min(step, n - at);
            double ms = decode(codes, at, m, state, &pcm);
            printf("%s stream: frames %2d..%2d  %8.2f ms  %zu samples so far, state %zu bytes\n",
                   name, at, at + m - 1, ms, pcm.size(), state.size());
            at += m;
            step = 8;
        }
        return pcm;
    };

    write("hello_offline.bin", offline("hello", hello));
    auto hello_stream = stream("hello", hello);
    write("hello_stream.bin", hello_stream);
    write("ref6_offline.bin", offline("ref6", ref6));
    write("ref6_stream.bin", stream("ref6", ref6));

    auto again = stream("hello rerun", hello);
    const bool same = again.size() == hello_stream.size() &&
                      !memcmp(again.data(), hello_stream.data(), again.size() * sizeof(float));
    printf("streamed hello rerun: %s the first streamed run\n", same ? "bit-identical to" : "DIFFERENT from");
    return 0;
}
