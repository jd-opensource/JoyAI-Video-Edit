// joyedge-bench: measure streaming DiT latency of a joyedge GGUF on this machine.
//
//   joyedge-bench -m model.gguf [--lat-h 20] [--lat-w 35] [--txt 400] [--chunks 4]
//                 [--steps 2] [-t N] [--stream-window W] [--ref HxW]
//                 [--frames-per-chunk 8] [--target-fps 24]
//
// Inputs are random; the numbers are pure compute/IO latency, not image quality.
#include "joyedge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <sys/resource.h>
#endif

static double peak_rss_mib() {
#if defined(_WIN32)
    return 0.0;
#else
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#  if defined(__APPLE__)
    return (double) ru.ru_maxrss / 1048576.0;
#  else
    return (double) ru.ru_maxrss / 1024.0;
#  endif
#endif
}

int main(int argc, char ** argv) {
    std::string model_path;
    joyedge_model_params mp = joyedge_model_default_params();
    joyedge_session_params sp = joyedge_session_default_params();
    int n_txt = 400, n_chunks = 4, frames_per_chunk = 8, ref_h = 0, ref_w = 0;
    double target_fps = 24.0;
    mp.verbose = 1;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model") model_path = next();
        else if (a == "--lat-h") sp.lat_h = atoi(next());
        else if (a == "--lat-w") sp.lat_w = atoi(next());
        else if (a == "--txt") n_txt = atoi(next());
        else if (a == "--chunks") n_chunks = atoi(next());
        else if (a == "--steps") sp.n_steps = atoi(next());
        else if (a == "--mti") sp.max_temporal_ids = atoi(next());
        else if (a == "-t" || a == "--threads") mp.n_threads = atoi(next());
        else if (a == "--stream-window") mp.stream_window = atoi(next());
        else if (a == "--frames-per-chunk") frames_per_chunk = atoi(next());
        else if (a == "--target-fps") target_fps = atof(next());
        else if (a == "--kv-f32") sp.kv_f32 = 1;
        else if (a == "--txt-cache") sp.txt_cache_interval = atoi(next());
        else if (a == "--ref") { if (sscanf(next(), "%dx%d", &ref_h, &ref_w) != 2) { fprintf(stderr, "--ref HxW\n"); return 1; } }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [--lat-h 20 --lat-w 35 --txt 400 --chunks 4 --steps 2 -t N "
                        "--stream-window W --ref HxW --frames-per-chunk 8 --target-fps 24]\n", argv[0]);
        return 1;
    }

    joyedge_model * model = joyedge_model_load(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed: %s\n", joyedge_last_error()); return 1; }
    joyedge_config cfg{};
    joyedge_model_get_config(model, &cfg);

    joyedge_session * sess = joyedge_session_new(model, sp);
    if (!sess) { fprintf(stderr, "session failed: %s\n", joyedge_last_error()); return 1; }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randn = [&](size_t n) { std::vector<float> v(n); for (auto & x : v) x = nd(rng); return v; };

    const size_t CN = (size_t) cfg.in_channels * sp.lat_h * sp.lat_w;
    {
        const std::vector<float> prompt = randn((size_t) n_txt * cfg.text_dim);
        if (joyedge_set_prompt(sess, prompt.data(), n_txt) != 0) { fprintf(stderr, "%s\n", joyedge_last_error()); return 1; }
    }
    if (ref_h > 0 && ref_w > 0) {
        const std::vector<float> ref = randn((size_t) cfg.in_channels * ref_h * ref_w);
        if (joyedge_set_reference_image(sess, ref.data(), ref_h, ref_w) != 0) { fprintf(stderr, "%s\n", joyedge_last_error()); return 1; }
    }

    printf("tokens/chunk=%d  text=%d  steps=%d  ref=%d  (per chunk: %d denoise forwards over %d img tokens + 1 store forward)\n",
           sp.lat_h * sp.lat_w, n_txt, sp.n_steps, ref_h * ref_w, sp.n_steps, 2 * sp.lat_h * sp.lat_w);

    std::vector<float> out(CN);
    double steady = 0.0;
    int n_steady = 0;
    joyedge_timings tm{};
    for (int k = 0; k < n_chunks; k++) {
        const std::vector<float> noise = randn(CN), src = randn(CN);
        if (joyedge_process_chunk(sess, noise.data(), src.data(), out.data(), 1) != 0) {
            fprintf(stderr, "chunk %d failed: %s\n", k, joyedge_last_error());
            return 1;
        }
        joyedge_get_timings(sess, &tm);
        printf("chunk %d: total %8.3fs  (denoise %8.3fs  store %7.3fs  publish %6.3fs)\n",
               k, tm.total_s, tm.denoise_s, tm.store_s, tm.commit_s);
        fflush(stdout);
        if (k >= 2 || n_chunks <= 2) { steady += tm.total_s; n_steady++; }
    }
    const double chunk_s = steady / (double) std::max(1, n_steady);
    const double fps = (double) frames_per_chunk / chunk_s;
    printf("\nsteady-state DiT latency : %.3f s per chunk (%d frames)\n", chunk_s, frames_per_chunk);
    printf("DiT-only throughput      : %.3f fps  (%.4fx of %.0f fps real time)\n", fps, fps / target_fps, target_fps);
    printf("KV state %.1f MiB | activation arena %.1f MiB | weights file %.1f MiB | peak RSS %.1f MiB\n",
           tm.kv_bytes / 1048576.0, tm.graph_bytes / 1048576.0, (double) cfg.file_bytes / 1048576.0, peak_rss_mib());

    joyedge_session_free(sess);
    joyedge_model_free(model);
    return 0;
}
