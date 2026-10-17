// Run the Drama 3 codebook tower (mtmd drama3_gen) on the rows saved by ref_tower.py.
// usage: tower mmproj.gguf backbone.gguf outdir [--gpu] [--split k] [--bench | --decode], run from D:\drama3\verify
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
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s mmproj.gguf backbone.gguf outdir [--gpu] [--split k] [--bench | --decode]\n", argv[0]);
        return 1;
    }
    const fs::path outdir = argv[3];
    bool gpu = false;
    bool bench = false;
    bool decode = false;
    int split = 20;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu")) {
            gpu = true;
        } else if (!strcmp(argv[i], "--bench")) {
            bench = true;
        } else if (!strcmp(argv[i], "--decode")) {
            decode = true;
        } else if (!strcmp(argv[i], "--split") && i + 1 < argc) {
            split = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    std::vector<int64_t> fshape, hshape;
    auto frames = load_npy<int64_t>("ref_tower_frames.npy", "'<i8'", fshape);
    auto ref_hidden = load_npy<float>("ref_tower_hidden.npy", "'<f4'", hshape);
    const int P = (int) fshape[0];
    const int W = (int) fshape[1];
    const int n_embd = (int) hshape[1];
    const int R = P - 1;

    // a long prompt, a few seconds of frames, then one clause prefilled after a restored prefix;
    // only the shapes and the cache depth matter, so the bench cycles the reference rows
    const int n_prefill = 300, n_single = 200, n_clause = 130;
    const int N = bench ? n_prefill + n_single + n_clause : R;

    // semantic_start_token_id .. semantic_end_token_id: main tokens of audio frames
    const int64_t sem_begin = 248099, sem_end = 252194;
    auto is_audio = [&](int row) { return frames[(size_t) row * W] >= sem_begin && frames[(size_t) row * W] <= sem_end; };

    std::vector<float> hidden((size_t) N * n_embd);
    std::vector<int32_t> next_codes((size_t) N * 9), cur_idx((size_t) N * 8);
    std::unique_ptr<bool[]> next_audio(new bool[N]), cur_audio(new bool[N]);
    for (int q = 0; q < N; q++) {
        const int p = q % R;
        std::copy_n(ref_hidden.data() + (size_t) p * n_embd, n_embd, hidden.data() + (size_t) q * n_embd);
        for (int k = 0; k < 9; k++) {
            next_codes[(size_t) q * 9 + k] = (int32_t) frames[(size_t) (p + 1) * W + 1 + k];
        }
        for (int k = 2; k < 10; k++) {
            cur_idx[(size_t) q * 8 + k - 2] = (int32_t) (4096 * k + frames[(size_t) p * W + 1 + k]);
        }
        next_audio[q] = is_audio(p + 1);
        cur_audio[q]  = is_audio(p);
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

    // read once by the mmproj context, each run moves the files to its own directory
    const fs::path dump_dir = outdir / "dump";
    if (!bench && !decode) {
        fs::create_directories(dump_dir);
        set_env("MTMD_DRAMA3_DUMP_DIR", dump_dir.string());
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
        fprintf(stderr, "%s is not a drama3 tower\n", argv[1]);
        return 1;
    }

    // rows row .. row + n - 1 go to the state's n_past, returns the wall time in ms
    auto process = [&](mtmd_gen_state * state, int row, int n, std::vector<float> * logits) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(ctx);
        inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd       = hidden.data() + (size_t) row * n_embd;
        inp.n_rows     = n;
        inp.next_codes = next_codes.data() + (size_t) row * 9;
        inp.next_audio = next_audio.get() + row;
        inp.cur_idx    = cur_idx.data() + (size_t) row * 8;
        inp.cur_audio  = cur_audio.get() + row;
        inp.state      = state;

        mtmd_gen_out out;
        auto t0 = std::chrono::steady_clock::now();
        if (mtmd_gen_audio_process(ctx, &inp, &out) != 0) {
            fprintf(stderr, "process failed at row %d\n", row);
            exit(1);
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (logits) {
            logits->insert(logits->end(), out.logits, out.logits + out.n_logits);
        }
        return ms;
    };

    if (bench) {
        mtmd_gen_state * state = mtmd_gen_state_init(ctx, 2048);
        if (!state) {
            fprintf(stderr, "failed to allocate the tower state\n");
            return 1;
        }

        // repeated from the same n_past, the first call of a new shape also pays for the allocation
        auto prefill = [&](int row, int n) {
            const int n_past = mtmd_gen_state_get_n_past(state);
            std::vector<double> t;
            for (int r = 0; r < 6; r++) {
                mtmd_gen_state_set_n_past(state, n_past);
                t.push_back(process(state, row, n, nullptr));
            }
            std::sort(t.begin() + 1, t.end());
            printf("prefill %3d rows at n_past %3d: first %8.2f ms, median of the next 5 %8.2f ms\n", n, n_past, t[0], t[3]);
        };

        prefill(0, n_prefill);

        std::vector<double> t;
        for (int i = 0; i < n_single; i++) {
            t.push_back(process(state, n_prefill + i, 1, nullptr));
        }
        std::sort(t.begin(), t.end());
        auto pct = [&](double q) { return t[std::min(t.size() - 1, (size_t) (q * t.size()))]; };
        printf("single rows %d..%d: p50 %.2f ms, p90 %.2f ms, p99 %.2f ms, max %.2f ms\n",
               n_prefill, n_prefill + n_single - 1, pct(0.50), pct(0.90), pct(0.99), t.back());

        prefill(n_prefill + n_single, n_clause);

        return 0;
    }

    if (decode) {
        // the prompt rows teacher-forced, then the rows that predict an audio frame decoded one at a time with
        // the reference's c0; the codes of the row itself (cur_idx) come from the reference too
        std::vector<int64_t> lshape;
        auto ref_logits = load_npy<float>("ref_tower_logits.npy", "'<f4'", lshape);
        const int n_vocab = (int) lshape[2];
        int first = 0;
        while (!is_audio(first + 1)) {
            first++;
        }
        const int n = R - first;

        mtmd_gen_state * state = mtmd_gen_state_init(ctx, 512);
        if (!state) {
            fprintf(stderr, "failed to allocate the tower state\n");
            return 1;
        }
        process(state, 0, first, nullptr);

        auto decode_row = [&](int p, float temp, int top_k, float top_p, uint32_t seed, std::vector<int32_t> & codes) {
            mtmd_gen_inp inp = mtmd_gen_inp_default(ctx);
            inp.type      = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
            inp.embd      = hidden.data() + (size_t) p * n_embd;
            inp.n_rows    = 1;
            inp.code0     = (int32_t) frames[(size_t) (p + 1) * W + 1];
            inp.cur_idx   = cur_idx.data() + (size_t) p * 8;
            inp.cur_audio = cur_audio.get() + p;
            inp.state     = state;
            inp.temp      = temp;
            inp.top_k     = top_k;
            inp.top_p     = top_p;
            inp.seed      = seed;

            mtmd_gen_out out;
            auto t0 = std::chrono::steady_clock::now();
            if (mtmd_gen_audio_process(ctx, &inp, &out) != 0 || out.n_codes != 9) {
                fprintf(stderr, "decode failed at row %d\n", p);
                exit(1);
            }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            codes.assign(out.codes, out.codes + out.n_codes);
            return ms;
        };

        // the first calls of the new graph pay for the allocation and the CUDA graph capture
        auto sweep = [&](const char * name, float temp, int top_k, float top_p, uint32_t seed) {
            std::vector<int32_t> codes, all;
            for (int w = 0; w < 2; w++) {
                mtmd_gen_state_set_n_past(state, first);
                decode_row(first, temp, top_k, top_p, seed, codes);
            }
            mtmd_gen_state_set_n_past(state, first);
            std::vector<double> t;
            for (int p = first; p < R; p++) {
                t.push_back(decode_row(p, temp, top_k, top_p, seed, codes));
                all.insert(all.end(), codes.begin(), codes.end());
            }
            std::vector<double> ts = t;
            std::sort(ts.begin(), ts.end());
            printf("%s: per row", name);
            for (double ms : t) {
                printf(" %.2f", ms);
            }
            printf(" ms, p50 %.2f ms\n", (ts[(n - 1) / 2] + ts[n / 2]) / 2);
            return all;
        };

        auto greedy = sweep("greedy", 0.7f, 1, 0.7f, 1234);
        int hit = 0;
        for (int p = first; p < R; p++) {
            printf("row %d:", p);
            for (int k = 0; k < 9; k++) {
                const int32_t got = greedy[(size_t) (p - first) * 9 + k];
                const int32_t ref = (int32_t) frames[(size_t) (p + 1) * W + 2 + k];
                printf(" %4d", got);
                hit += got == ref;
            }
            printf("\n");
            bool missed = false;
            for (int k = 0; k < 9; k++) {
                const int32_t got = greedy[(size_t) (p - first) * 9 + k];
                const int32_t ref = (int32_t) frames[(size_t) (p + 1) * W + 2 + k];
                if (got != ref) {
                    // the reference logits are teacher-forced, after an earlier miss in the row this head saw other codes
                    const float * l = ref_logits.data() + ((size_t) p * 9 + k) * n_vocab;
                    printf("  miss c%d: got %d, reference %d; reference logits %.4f vs %.4f (gap %.4f)%s\n", k + 1, got, ref,
                           l[got], l[ref], l[ref] - l[got], missed ? ", after a miss" : "");
                    missed = true;
                }
            }
        }
        printf("greedy: %d/%d codes match the reference frames\n", hit, 9 * n);

        // the same rows teacher-forced with the codes decode mode picked: each head's argmax must be the code it sampled
        mtmd_gen_state_set_n_past(state, first);
        int own_hit = 0;
        for (int p = first; p < R; p++) {
            std::vector<int32_t> codes = { (int32_t) frames[(size_t) (p + 1) * W + 1] };
            codes.insert(codes.end(), greedy.begin() + (p - first) * 9, greedy.begin() + (p - first) * 9 + 8);
            bool audio = true;
            mtmd_gen_inp inp = mtmd_gen_inp_default(ctx);
            inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
            inp.embd       = hidden.data() + (size_t) p * n_embd;
            inp.n_rows     = 1;
            inp.next_codes = codes.data();
            inp.next_audio = &audio;
            inp.cur_idx    = cur_idx.data() + (size_t) p * 8;
            inp.cur_audio  = cur_audio.get() + p;
            inp.state      = state;
            mtmd_gen_out out;
            if (mtmd_gen_audio_process(ctx, &inp, &out) != 0) {
                fprintf(stderr, "teacher-forced row %d failed\n", p);
                return 1;
            }
            for (int k = 0; k < 9; k++) {
                const float * l = out.logits + (size_t) k * n_vocab;
                own_hit += (int32_t) (std::max_element(l, l + n_vocab) - l) == greedy[(size_t) (p - first) * 9 + k];
            }
        }
        printf("greedy vs teacher-forced on its own codes: %d/%d argmax equal\n", own_hit, 9 * n);

        // and with the reference codes on the same history, the no-cascade version of the greedy check
        mtmd_gen_state_set_n_past(state, first);
        std::vector<float> tf;
        for (int p = first; p < R; p++) {
            process(state, p, 1, &tf);
        }
        int tf_hit = 0;
        for (int p = first; p < R; p++) {
            printf("teacher-forced row %d misses:", p);
            for (int k = 0; k < 9; k++) {
                const float * l = tf.data() + ((size_t) (p - first) * 9 + k) * n_vocab;
                const int32_t got = (int32_t) (std::max_element(l, l + n_vocab) - l);
                const int32_t ref = (int32_t) frames[(size_t) (p + 1) * W + 2 + k];
                if (got == ref) {
                    tf_hit++;
                } else {
                    const float * rl = ref_logits.data() + ((size_t) p * 9 + k) * n_vocab;
                    printf(" c%d (%d vs %d, reference gap %.4f)", k + 1, got, ref, rl[ref] - rl[got]);
                }
            }
            printf("\n");
        }
        printf("teacher-forced with the reference codes: %d/%d argmax equal to the reference frames\n", tf_hit, 9 * n);

        // sampled, two seeds and a repeat of the first
        auto s1 = sweep("seed 1", 0.7f, 30, 0.7f, 1);
        auto s2 = sweep("seed 2", 0.7f, 30, 0.7f, 2);
        auto s1b = sweep("seed 1 again", 0.7f, 30, 0.7f, 1);
        for (const auto * s : { &s1, &s2 }) {
            printf("%s:\n", s == &s1 ? "seed 1" : "seed 2");
            int n_top = 0;
            for (int p = first; p < R; p++) {
                printf("  row %d:", p);
                for (int k = 0; k < 9; k++) {
                    const int32_t c = (*s)[(size_t) (p - first) * 9 + k];
                    if (c < 0 || c >= n_vocab) {
                        fprintf(stderr, "code out of range: %d\n", c);
                        return 1;
                    }
                    printf(" %4d", c);
                }
                // c1 has the same inputs as the reference's teacher-forced head 0, so its rank there is meaningful
                const int32_t c1 = (*s)[(size_t) (p - first) * 9];
                const float * l = ref_logits.data() + (size_t) p * 9 * n_vocab;
                int rank = 0;
                for (int v = 0; v < n_vocab; v++) {
                    rank += l[v] > l[c1];
                }
                n_top += rank < 30;
                printf("   (c1 rank %d in the reference head)\n", rank);
            }
            printf("  c1 within the reference top 30: %d/%d\n", n_top, n);
        }
        int same12 = 0;
        for (size_t i = 0; i < s1.size(); i++) {
            same12 += s1[i] == s2[i];
        }
        printf("seed 1 vs seed 2: %d/%d codes equal; seed 1 repeated: %s\n", same12, 9 * n, s1 == s1b ? "identical" : "DIFFERENT");

        // c1 of one row over many seeds against fish-speech's rule applied to the same row's head 0 logits,
        // taken teacher-forced (head 0 only depends on c0)
        {
            const int p = R - 3, n_draw = 1000;
            std::vector<int> hist(n_vocab, 0);
            std::vector<int32_t> codes;
            for (int s = 0; s < n_draw; s++) {
                mtmd_gen_state_set_n_past(state, p);
                decode_row(p, 0.7f, 30, 0.7f, 1000 + s, codes);
                hist[codes[0]]++;
            }
            mtmd_gen_state_set_n_past(state, p);
            std::vector<float> own;
            process(state, p, 1, &own);
            const float * l = own.data();
            std::vector<int> ids(n_vocab);
            for (int v = 0; v < n_vocab; v++) {
                ids[v] = v;
            }
            std::sort(ids.begin(), ids.end(), [&](int a, int b) { return l[a] > l[b]; });
            double z = 0;
            for (int v = 0; v < n_vocab; v++) {
                z += exp((l[v] - l[ids[0]]) / 0.7);
            }
            std::vector<double> pr;
            double cum = 0;
            for (int r = 0; r < 30; r++) {
                const double q = exp((l[ids[r]] - l[ids[0]]) / 0.7) / z;
                cum += q;
                if (r > 0 && cum > 0.7) {
                    break;
                }
                pr.push_back(q);
            }
            double kept = 0;
            for (double q : pr) {
                kept += q;
            }
            int inside = 0;
            printf("row %d c1 over %d seeds (rank: code expected observed):", p, n_draw);
            for (size_t r = 0; r < pr.size(); r++) {
                printf(" %zu: %d %.1f %d;", r, ids[r], n_draw * pr[r] / kept, hist[ids[r]]);
                inside += hist[ids[r]];
            }
            printf(" outside the kept set: %d\n", n_draw - inside);
            double chi2 = 0;
            for (size_t r = 0; r < pr.size(); r++) {
                const double e = n_draw * pr[r] / kept;
                chi2 += (hist[ids[r]] - e) * (hist[ids[r]] - e) / e;
            }
            printf("chi-square %.2f on %zu degrees of freedom\n", chi2, pr.size() - 1);
        }

        return 0;
    }

    auto run = [&](const char * name, mtmd_gen_state * state, int row, const std::vector<int> & chunks) {
        std::vector<float> logits;
        for (int n : chunks) {
            double ms = process(state, row, n, &logits);
            printf("%s: rows %2d..%2d  %8.2f ms\n", name, row, row + n - 1, ms);
            row += n;
        }

        const fs::path dir = outdir / name;
        fs::create_directories(dir);
        for (const auto & e : fs::directory_iterator(dump_dir)) {
            fs::rename(e.path(), dir / e.path().filename());
        }

        std::ofstream f(outdir / (std::string("logits_") + name + ".bin"), std::ios::binary);
        f.write((const char *) logits.data(), logits.size() * sizeof(float));
        return logits;
    };

    mtmd_gen_state * state = mtmd_gen_state_init(ctx, 512);
    if (!state) {
        fprintf(stderr, "failed to allocate the tower state\n");
        return 1;
    }
    run("a", state, 0, { R });
    mtmd_gen_state_free(state);

    state = mtmd_gen_state_init(ctx, 512);
    if (!state) {
        fprintf(stderr, "failed to allocate the tower state\n");
        return 1;
    }
    std::vector<int> chunks = { split };
    for (int i = split; i < R; i++) {
        chunks.push_back(1);
    }
    auto logits_b = run("b", state, 0, chunks);

    // rewind run b and redo its tail, same kernels so the logits must match bit for bit
    const int n_keep = 25;
    mtmd_gen_state_set_n_past(state, n_keep);
    auto logits_r = run("rewind", state, n_keep, std::vector<int>(R - n_keep, 1));
    const size_t off = (size_t) n_keep * 9 * 4096;
    const bool same = logits_r.size() + off == logits_b.size() &&
                      memcmp(logits_r.data(), logits_b.data() + off, logits_r.size() * sizeof(float)) == 0;
    printf("rewind to %d, rows %d..%d again: %s run b\n", n_keep, n_keep, R - 1, same ? "bit-identical to" : "DIFFERENT from");
    return 0;
}
