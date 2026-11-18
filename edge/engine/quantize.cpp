// joyedge-quantize: re-quantize a joyedge GGUF (F32/F16/BF16) to a ggml block format.
//
//   joyedge-quantize in.gguf out.gguf q8_0|q4_0|q4_k|q5_k|q6_k|f16 [n_threads]
//
// Tensors are processed one at a time and streamed to disk, so peak memory is one
// weight matrix (fp32) plus its quantized copy -- a 16B-parameter checkpoint
// converts on a 16 GB machine.
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static bool parse_type(const std::string & s, ggml_type & out) {
    static const struct { const char * name; ggml_type type; } table[] = {
        { "f16",  GGML_TYPE_F16  }, { "q8_0", GGML_TYPE_Q8_0 }, { "q4_0", GGML_TYPE_Q4_0 },
        { "q4_k", GGML_TYPE_Q4_K }, { "q5_k", GGML_TYPE_Q5_K }, { "q6_k", GGML_TYPE_Q6_K },
        { "q5_0", GGML_TYPE_Q5_0 }, { "q3_k", GGML_TYPE_Q3_K }, { "bf16", GGML_TYPE_BF16 }, { "f32", GGML_TYPE_F32 },
    };
    for (const auto & e : table) {
        if (s == e.name) { out = e.type; return true; }
    }
    return false;
}

static bool starts_with(const std::string & s, const char * p) { return s.rfind(p, 0) == 0; }
static bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Target type for one tensor, or GGML_TYPE_COUNT to copy it unchanged.
static ggml_type target_type(const std::string & name, const ggml_tensor * t, ggml_type want) {
    const bool is_matrix = ggml_n_dims(t) == 2 && ends_with(name, ".weight");
    if (!is_matrix || ggml_nelements(t) < 65536) return GGML_TYPE_COUNT;
    // first / last projections are tiny and the most sensitive: leave them alone
    if (starts_with(name, "img_in.") || starts_with(name, "proj_out.")) return GGML_TYPE_COUNT;
    ggml_type type = want;
    if (!starts_with(name, "double_blocks.") && ggml_is_quantized(want) && want != GGML_TYPE_Q8_0) {
        // conditioning embedders run once per session: keep them at 8 bit
        type = GGML_TYPE_Q8_0;
    }
    if (t->ne[0] % ggml_blck_size(type) != 0) return GGML_TYPE_COUNT;
    return type;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s in.gguf out.gguf q8_0|q4_0|q4_k|q5_k|q6_k|f16 [n_threads]\n", argv[0]);
        return 1;
    }
    const char * fin = argv[1];
    const char * fout = argv[2];
    ggml_type want;
    if (!parse_type(argv[3], want)) {
        fprintf(stderr, "unknown type: %s\n", argv[3]);
        return 1;
    }
    const int n_threads = argc > 4 ? std::max(1, atoi(argv[4])) : (int) std::max(1u, std::thread::hardware_concurrency());

    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * in = gguf_init_from_file(fin, gp);
    if (!in) { fprintf(stderr, "cannot read %s\n", fin); return 1; }

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    gguf_set_val_str(out, "joyedge.quantization", argv[3]);

    const int64_t n = gguf_get_n_tensors(in);
    std::vector<ggml_type> new_type((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        gguf_add_tensor(out, ggml_get_tensor(meta, name));
    }
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        const ggml_type tt = target_type(name, t, want);
        new_type[(size_t) i] = (tt == GGML_TYPE_COUNT || tt == t->type) ? GGML_TYPE_COUNT : tt;
        if (new_type[(size_t) i] != GGML_TYPE_COUNT) gguf_set_tensor_type(out, name, tt);
    }

    FILE * fi = fopen(fin, "rb");
    FILE * fo = fopen(fout, "wb");
    if (!fi || !fo) { fprintf(stderr, "cannot open files\n"); return 1; }

    const size_t meta_size = gguf_get_meta_size(out);
    const size_t align = gguf_get_alignment(out);
    {
        std::vector<uint8_t> zeros(meta_size, 0);
        fwrite(zeros.data(), 1, zeros.size(), fo);
    }

    if (ggml_is_quantized(want)) ggml_quantize_init(want);
    ggml_quantize_init(GGML_TYPE_Q8_0);

    const size_t data_off = gguf_get_data_offset(in);
    std::vector<uint8_t> raw, dst;
    std::vector<float> f32;
    size_t total_in = 0, total_out = 0;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t nbytes = gguf_get_tensor_size(in, i);
        raw.resize(nbytes);
#if defined(_WIN32)
        _fseeki64(fi, (long long) (data_off + gguf_get_tensor_offset(in, i)), SEEK_SET);
#else
        fseeko(fi, (off_t) (data_off + gguf_get_tensor_offset(in, i)), SEEK_SET);
#endif
        if (fread(raw.data(), 1, nbytes, fi) != nbytes) { fprintf(stderr, "short read: %s\n", name); return 1; }

        const uint8_t * wdata = raw.data();
        size_t wsize = nbytes;
        const ggml_type tt = new_type[(size_t) i];
        if (tt != GGML_TYPE_COUNT) {
            const int64_t nel = ggml_nelements(t);
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows = t->ne[1];
            const float * src = nullptr;
            if (t->type == GGML_TYPE_F32) {
                src = (const float *) raw.data();
            } else if (t->type == GGML_TYPE_F16) {
                f32.resize((size_t) nel);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), nel);
                src = f32.data();
            } else if (t->type == GGML_TYPE_BF16) {
                f32.resize((size_t) nel);
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), f32.data(), nel);
                src = f32.data();
            } else {
                fprintf(stderr, "%s: source type %s cannot be re-quantized\n", name, ggml_type_name(t->type));
                return 1;
            }
            const size_t row_size = ggml_row_size(tt, n_per_row);
            dst.resize(row_size * (size_t) nrows);
            std::vector<std::thread> pool;
            const int nt = (int) std::min<int64_t>(n_threads, nrows);
            for (int th = 0; th < nt; th++) {
                const int64_t r0 = nrows * th / nt, r1 = nrows * (th + 1) / nt;
                pool.emplace_back([=, &dst] {
                    ggml_quantize_chunk(tt, src, dst.data(), r0 * n_per_row, r1 - r0, n_per_row, nullptr);
                });
            }
            for (auto & th : pool) th.join();
            wdata = dst.data();
            wsize = dst.size();
            fprintf(stderr, "%-56s %-5s -> %-5s %8.1f MiB -> %8.1f MiB\n", name, ggml_type_name(t->type),
                    ggml_type_name(tt), (double) nbytes / 1048576.0, (double) wsize / 1048576.0);
        }
        fwrite(wdata, 1, wsize, fo);
        const size_t pad = GGML_PAD(wsize, align) - wsize;
        for (size_t p = 0; p < pad; p++) fputc(0, fo);
        total_in += nbytes;
        total_out += wsize;
    }
    ggml_quantize_free();

    std::vector<uint8_t> mbuf(meta_size);
    gguf_get_meta_data(out, mbuf.data());
    rewind(fo);
    fwrite(mbuf.data(), 1, mbuf.size(), fo);
    fclose(fo);
    fclose(fi);

    fprintf(stderr, "done: %.1f MiB -> %.1f MiB (%s)\n", (double) total_in / 1048576.0,
            (double) total_out / 1048576.0, argv[3]);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta);
    return 0;
}
