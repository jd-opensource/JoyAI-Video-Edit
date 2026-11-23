// joyedge - CPU streaming runtime for the JoyAI-Video-Edit autoregressive MMDiT.
// See joyedge.h for the public API and edge/README.md for the design.
#include "joyedge.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define JOYEDGE_HAVE_MMAP 1
#endif

namespace {

thread_local std::string g_err;

int fail(const std::string & msg) {
    g_err = msg;
    return -1;
}

constexpr float kNormEps = 1e-6f;
enum Stream { IMG = 0, TXT = 1 };
enum Lin { QKV = 0, PROJ = 1, UP = 2, DOWN = 3 };
// KV pool slots.  TXT0 + i holds the cached text-stream K/V of denoise step i (optional).
enum Slot { SINK = 0, PREV = 1, REF = 2, TXT0 = 3 };
constexpr int MAX_SEG = 4;   // cached segments one block can attend to: sink, prev, ref, text

// Small parallel-for for work that lives outside ggml graphs.
template <typename F>
void parallel_for(int n_threads, int64_t n, F && fn) {
    n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    if (n_threads == 1) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; t++) {
        const int64_t lo = n * t / n_threads;
        const int64_t hi = n * (t + 1) / n_threads;
        pool.emplace_back([lo, hi, &fn] { fn(lo, hi); });
    }
    for (auto & th : pool) th.join();
}

// --------------------------------------------------------------------------- //
// model
// --------------------------------------------------------------------------- //

struct Block {
    ggml_tensor * mod[2] = {nullptr, nullptr};
    ggml_tensor * w[2][4] = {};
    ggml_tensor * b[2][4] = {};
    ggml_tensor * qn[2] = {nullptr, nullptr};
    ggml_tensor * kn[2] = {nullptr, nullptr};
    size_t lo = SIZE_MAX;  // byte range of this block inside the mapping
    size_t hi = 0;
};

} // namespace

struct joyedge_model {
    joyedge_config cfg{};
    joyedge_model_params params{};
    int time_freq_dim = 256;

    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;

    uint8_t * map_base = nullptr;
    size_t    map_size = 0;
    bool      is_mmap  = false;
    int       fd       = -1;

    std::vector<Block> blocks;
    ggml_tensor * img_in_w = nullptr, * img_in_b = nullptr;
    ggml_tensor * te1_w = nullptr, * te1_b = nullptr, * te2_w = nullptr, * te2_b = nullptr;
    ggml_tensor * tp_w = nullptr, * tp_b = nullptr;
    ggml_tensor * tx1_w = nullptr, * tx1_b = nullptr, * tx2_w = nullptr, * tx2_b = nullptr;
    ggml_tensor * out_w = nullptr, * out_b = nullptr;

    ggml_backend_t    backend = nullptr;
    ggml_threadpool_t tp      = nullptr;
    int n_threads = 1;
};

namespace {

// ---- weight streaming ------------------------------------------------------ //

void advise_range(const joyedge_model * m, size_t lo, size_t hi, bool need) {
#if defined(JOYEDGE_HAVE_MMAP)
    if (!m->is_mmap || hi <= lo) return;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (need) {
        lo = lo / page * page;
        hi = std::min(m->map_size, (hi + page - 1) / page * page);
        if (hi > lo) (void) madvise(m->map_base + lo, hi - lo, MADV_WILLNEED);
    } else {
        // shrink inwards so that pages shared with neighbouring blocks stay mapped
        lo = (lo + page - 1) / page * page;
        hi = hi / page * page;
        if (hi > lo) (void) madvise(m->map_base + lo, hi - lo, MADV_DONTNEED);
    }
#else
    (void) m; (void) lo; (void) hi; (void) need;
#endif
}

// Called right before block `pl` is computed: read ahead the next block and
// release the one that just left the window.
void stream_advise(const joyedge_model * m, int pl) {
    const int window = m->params.stream_window;
    const int P = (int) m->blocks.size();
    if (window <= 0 || P <= 1) return;
    const Block & next = m->blocks[(pl + 1) % P];
    advise_range(m, next.lo, next.hi, true);
    if (P > window + 1) {
        const Block & old = m->blocks[((pl - window) % P + P) % P];
        advise_range(m, old.lo, old.hi, false);
    }
}

// ---- tiny graph helpers ---------------------------------------------------- //

enum Act { ACT_NONE = 0, ACT_SILU = 1, ACT_GELU = 2 };

// y = act(W x + b);  x: [K, M] token-major fp32  ->  y: [N, M]
std::vector<float> run_linear(joyedge_model * m, ggml_tensor * W, ggml_tensor * b,
                              const float * x, int64_t M, Act act = ACT_NONE) {
    const int64_t K = W->ne[0];
    const int64_t N = W->ne[1];

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    in->data = (void *) x;
    ggml_tensor * y = ggml_mul_mat(ctx, W, in);
    if (b) y = ggml_add(ctx, y, b);
    if (act == ACT_SILU) y = ggml_silu(ctx, y);
    if (act == ACT_GELU) y = ggml_gelu(ctx, y);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(m->backend, gf);
    std::vector<float> out((size_t) (N * M));
    memcpy(out.data(), y->data, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return out;
}

inline void silu_inplace(std::vector<float> & v) {
    for (auto & x : v) x = x / (1.0f + std::exp(-x));
}

// ---- custom CPU ops (the counterpart of the fused joyomni_ops CUDA kernels) -- //

struct LnModP { int shift_idx; int scale_idx; float eps; };

// dst = LayerNorm(x) * (1 + scale) + shift          args: x [D, M], mod [D, 6]
void op_ln_mod(ggml_tensor * dst, int ith, int nth, void * ud) {
    const LnModP * p = (const LnModP *) ud;
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * mod = dst->src[1];
    const int64_t D = x->ne[0];
    const int64_t M = x->ne[1];
    const float * shift = (const float *) mod->data + (int64_t) p->shift_idx * D;
    const float * scale = (const float *) mod->data + (int64_t) p->scale_idx * D;
    const float * X = (const float *) x->data;
    float * Y = (float *) dst->data;
    const int64_t m0 = M * ith / nth, m1 = M * (ith + 1) / nth;
    for (int64_t m = m0; m < m1; m++) {
        const float * xr = X + m * D;
        float * yr = Y + m * D;
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int64_t d = 0; d < D; d++) sum += xr[d];
        const float mean = sum / (float) D;
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (int64_t d = 0; d < D; d++) { const float c = xr[d] - mean; var += c * c; }
        const float r = 1.0f / std::sqrt(var / (float) D + p->eps);
        #pragma omp simd
        for (int64_t d = 0; d < D; d++) yr[d] = (xr[d] - mean) * r * (1.0f + scale[d]) + shift[d];
    }
}

struct GateP { int gate_idx; };

// dst = residual + x * gate                         args: residual [D, M], x [D, M], mod [D, 6]
void op_add_gate(ggml_tensor * dst, int ith, int nth, void * ud) {
    const GateP * p = (const GateP *) ud;
    const ggml_tensor * res = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const int64_t D = res->ne[0];
    const int64_t M = res->ne[1];
    const float * gate = (const float *) dst->src[2]->data + (int64_t) p->gate_idx * D;
    const float * R = (const float *) res->data;
    const float * X = (const float *) x->data;
    float * Y = (float *) dst->data;
    const int64_t m0 = M * ith / nth, m1 = M * (ith + 1) / nth;
    for (int64_t m = m0; m < m1; m++) {
        const float * rr = R + m * D;
        const float * xr = X + m * D;
        float * yr = Y + m * D;
        #pragma omp simd
        for (int64_t d = 0; d < D; d++) yr[d] = rr[d] + xr[d] * gate[d];
    }
}

struct AttnP {
    int64_t D, H, hd;
    int64_t Mi, Mt;   // image / text tokens produced by this block
    int64_t Lc;       // cached tokens in front of them (kv only)
    int n_seg;        // number of cached segments (kv only)
    int64_t seg_len[MAX_SEG];
    bool is_k;        // kv op: key (norm + rope) or value (copy)
    bool rope;        // apply rope to image tokens
    bool f32;         // kv storage type
    float eps;
    // indices into dst->src
    int a_qkv_img, a_norm_img, a_cos, a_sin, a_qkv_txt, a_norm_txt, a_seg0;
};

// per-head RMSNorm (+ interleaved RoPE) of one head vector into tmp
inline void norm_rope_head(const float * x, const float * w, const float * c, const float * s,
                           int64_t hd, float eps, bool rope, float * out) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int64_t d = 0; d < hd; d++) ss += x[d] * x[d];
    const float r = 1.0f / std::sqrt(ss / (float) hd + eps);
    if (rope) {
        for (int64_t j = 0; j < hd / 2; j++) {
            const float a = x[2 * j] * r * w[2 * j];
            const float b = x[2 * j + 1] * r * w[2 * j + 1];
            out[2 * j]     = a * c[j] - b * s[j];
            out[2 * j + 1] = a * s[j] + b * c[j];
        }
    } else {
        #pragma omp simd
        for (int64_t d = 0; d < hd; d++) out[d] = x[d] * r * w[d];
    }
}

inline void put_row(void * dst, const float * src, int64_t n, bool f32) {
    if (f32) memcpy(dst, src, (size_t) n * sizeof(float));
    else     ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) dst, n);
}

// q: dst fp32 [hd, Mi + Mt, H]
void op_q(ggml_tensor * dst, int ith, int nth, void * ud) {
    const AttnP * p = (const AttnP *) ud;
    const int64_t D = p->D, H = p->H, hd = p->hd, Mq = p->Mi + p->Mt;
    const float * qi  = (const float *) dst->src[p->a_qkv_img]->data;
    const float * wi  = (const float *) dst->src[p->a_norm_img]->data;
    const float * cos = (const float *) dst->src[p->a_cos]->data;
    const float * sin = (const float *) dst->src[p->a_sin]->data;
    const float * qt  = p->Mt ? (const float *) dst->src[p->a_qkv_txt]->data : nullptr;
    const float * wt  = p->Mt ? (const float *) dst->src[p->a_norm_txt]->data : nullptr;
    float * out = (float *) dst->data;
    const int64_t m0 = Mq * ith / nth, m1 = Mq * (ith + 1) / nth;
    for (int64_t m = m0; m < m1; m++) {
        const bool img = m < p->Mi;
        const float * src = img ? qi + m * 3 * D : qt + (m - p->Mi) * 3 * D;   // q is the first third
        const float * w = img ? wi : wt;
        const float * c = img ? cos + m * (hd / 2) : nullptr;
        const float * s = img ? sin + m * (hd / 2) : nullptr;
        for (int64_t h = 0; h < H; h++) {
            norm_rope_head(src + h * hd, w, c, s, hd, p->eps, img && p->rope, out + (h * Mq + m) * hd);
        }
    }
}

// k / v: dst [hd, Lc + Mi + Mt, H] in the KV storage type; cached segments first.
void op_kv(ggml_tensor * dst, int ith, int nth, void * ud) {
    const AttnP * p = (const AttnP *) ud;
    const int64_t D = p->D, H = p->H, hd = p->hd;
    const int64_t Mn = p->Mi + p->Mt, Lk = p->Lc + Mn;
    const size_t es = p->f32 ? sizeof(float) : sizeof(ggml_fp16_t);
    uint8_t * out = (uint8_t *) dst->data;

    // cached segments: straight copies, one head at a time
    for (int64_t h = ith; h < H; h += nth) {
        int64_t off = 0;
        for (int sgi = 0; sgi < p->n_seg; sgi++) {
            const uint8_t * seg = (const uint8_t *) dst->src[p->a_seg0 + sgi]->data;
            const int64_t n = p->seg_len[sgi];
            memcpy(out + (size_t) ((h * Lk + off) * hd) * es, seg + (size_t) (h * n * hd) * es, (size_t) (n * hd) * es);
            off += n;
        }
    }

    const float * xi  = (const float *) dst->src[p->a_qkv_img]->data;
    const float * xt  = p->Mt ? (const float *) dst->src[p->a_qkv_txt]->data : nullptr;
    const float * wi  = p->is_k ? (const float *) dst->src[p->a_norm_img]->data : nullptr;
    const float * wt  = (p->is_k && p->Mt) ? (const float *) dst->src[p->a_norm_txt]->data : nullptr;
    const float * cos = (p->is_k && p->rope) ? (const float *) dst->src[p->a_cos]->data : nullptr;
    const float * sin = (p->is_k && p->rope) ? (const float *) dst->src[p->a_sin]->data : nullptr;
    const int64_t part = p->is_k ? 1 : 2;   // qkv = [q | k | v]
    std::vector<float> tmp((size_t) hd);

    const int64_t m0 = Mn * ith / nth, m1 = Mn * (ith + 1) / nth;
    for (int64_t m = m0; m < m1; m++) {
        const bool img = m < p->Mi;
        const float * src = (img ? xi + m * 3 * D : xt + (m - p->Mi) * 3 * D) + part * D;
        for (int64_t h = 0; h < H; h++) {
            void * o = out + (size_t) ((h * Lk + p->Lc + m) * hd) * es;
            if (!p->is_k) {
                put_row(o, src + h * hd, hd, p->f32);
                continue;
            }
            const bool rope = img && p->rope;
            norm_rope_head(src + h * hd, img ? wi : wt,
                           rope ? cos + m * (hd / 2) : nullptr, rope ? sin + m * (hd / 2) : nullptr,
                           hd, p->eps, rope, tmp.data());
            put_row(o, tmp.data(), hd, p->f32);
        }
    }
}

// ---- per-block graph ------------------------------------------------------- //

struct BlockGraph {
    int64_t Mi = 0, Mt = 0;
    bool store = false;
    bool capture = false;     // expose the assembled K/V so the text rows can be cached
    int n_seg = 0;
    int seg_slot[MAX_SEG] = {0, 0, 0, 0};
    int64_t seg_len[MAX_SEG] = {0, 0, 0, 0};
    int64_t Lc = 0;

    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;

    ggml_tensor * x[2] = {nullptr, nullptr};
    ggml_tensor * mod[2] = {nullptr, nullptr};
    ggml_tensor * cos = nullptr, * sin = nullptr;
    ggml_tensor * seg_k[MAX_SEG] = {}, * seg_v[MAX_SEG] = {};
    ggml_tensor * W[2][4] = {}, * B[2][4] = {};
    ggml_tensor * QN[2] = {}, * KN[2] = {};

    ggml_tensor * o[2] = {nullptr, nullptr};
    ggml_tensor * o_kstore = nullptr, * o_vstore = nullptr;
    ggml_tensor * o_k = nullptr, * o_v = nullptr;   // full K/V (capture graphs only)

    std::vector<std::unique_ptr<LnModP>> ln_params;
    std::vector<std::unique_ptr<GateP>>  gate_params;
    std::vector<std::unique_ptr<AttnP>>  attn_params;

    ~BlockGraph() {
        if (ga) ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
    }

    bool matches(int64_t mi, int64_t mt, bool st, bool cap, int ns, const int * slots, const int64_t * lens) const {
        if (mi != Mi || mt != Mt || st != store || cap != capture || ns != n_seg) return false;
        for (int i = 0; i < ns; i++) if (slots[i] != seg_slot[i] || lens[i] != seg_len[i]) return false;
        return true;
    }

    size_t arena_bytes() const { return ga ? ggml_gallocr_get_buffer_size(ga, 0) : 0; }
};

} // namespace

struct joyedge_session {
    joyedge_model * m = nullptr;
    joyedge_session_params p{};

    int64_t N = 0;  // tokens per chunk
    std::vector<float> sigmas, timesteps;
    // mods[ti][pl][stream] = modulate_table + time embedding;  ti == n_steps is t = 0
    std::vector<std::vector<std::vector<std::vector<float>>>> mods;

    std::vector<float> txt_tokens;  // [D, Mt] after the text embedder
    int64_t Mt = 0;

    struct KvSlot {
        int64_t n = 0;
        bool valid = false;
        int chunk = -1;                          // chunk the content was computed on (text slots)
        std::vector<std::vector<uint8_t>> k, v;  // per logical layer, [hd, n, H]
    };
    std::vector<KvSlot> slot;                    // SINK, PREV, REF, TXT0 .. TXT0 + n_steps - 1
    std::vector<std::vector<uint8_t>> stage_k, stage_v;
    int64_t stage_n = 0;
    bool stage_pending = false;    // staged KV of the previous chunk not yet published to PREV
    int  stage_chunk   = -1;

    int chunk_idx = 0;

    std::unique_ptr<BlockGraph> g_denoise, g_denoise_lite, g_store;
    std::vector<float> h_img, h_txt, cos, sin;
    std::vector<float> dummy;

    joyedge_timings tm{};
};

namespace {

size_t kv_es(const joyedge_session * s) { return s->p.kv_f32 ? sizeof(float) : sizeof(ggml_fp16_t); }
ggml_type kv_type(const joyedge_session * s) { return s->p.kv_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16; }

std::unique_ptr<BlockGraph> build_block_graph(joyedge_session * s, int64_t Mi, int64_t Mt, bool store, bool capture,
                                              int n_seg, const int * seg_slot, const int64_t * seg_len) {
    joyedge_model * m = s->m;
    const joyedge_config & c = m->cfg;
    const int64_t D = c.hidden_size, H = c.heads_num, hd = D / H;
    const Block & b0 = m->blocks[0];

    auto g = std::make_unique<BlockGraph>();
    g->Mi = Mi; g->Mt = Mt; g->store = store; g->capture = capture; g->n_seg = n_seg;
    int64_t Lc = 0;
    for (int i = 0; i < n_seg; i++) { g->seg_slot[i] = seg_slot[i]; g->seg_len[i] = seg_len[i]; Lc += seg_len[i]; }
    g->Lc = Lc;

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    g->ctx = ggml_init(ip);
    ggml_context * ctx = g->ctx;

    // The dummy buffer only gives every leaf a non-null data pointer so that the
    // graph allocator treats it as externally owned; real pointers are bound per block.
    size_t dummy_floats = (size_t) std::max<int64_t>({ D * std::max(Mi, Mt), 6 * D, (hd / 2) * Mi, 16 });
    for (int i = 0; i < n_seg; i++) dummy_floats = std::max(dummy_floats, (size_t) (hd * seg_len[i] * H));
    if (s->dummy.size() < dummy_floats) s->dummy.assign(dummy_floats, 0.0f);
    void * dummy = s->dummy.data();

    auto leaf_like = [&](const ggml_tensor * t) {
        ggml_tensor * r = ggml_new_tensor_2d(ctx, t->type, t->ne[0], t->ne[1]);
        r->data = t->data;
        return r;
    };
    auto leaf = [&](ggml_type type, int64_t n0, int64_t n1, int64_t n2 = 1) {
        ggml_tensor * r = ggml_new_tensor_3d(ctx, type, n0, n1, n2);
        r->data = dummy;
        return r;
    };

    const int n_streams = (Mt > 0) ? 2 : 1;
    for (int st = 0; st < n_streams; st++) {
        g->x[st]   = leaf(GGML_TYPE_F32, D, st == IMG ? Mi : Mt);
        g->mod[st] = leaf(GGML_TYPE_F32, D, 6);
        for (int l = 0; l < 4; l++) {
            g->W[st][l] = leaf_like(b0.w[st][l]);
            g->B[st][l] = leaf_like(b0.b[st][l]);
        }
        g->QN[st] = leaf_like(b0.qn[st]);
        g->KN[st] = leaf_like(b0.kn[st]);
    }
    g->cos = leaf(GGML_TYPE_F32, hd / 2, Mi);
    g->sin = leaf(GGML_TYPE_F32, hd / 2, Mi);
    for (int i = 0; i < n_seg; i++) {
        g->seg_k[i] = leaf(kv_type(s), hd, seg_len[i], H);
        g->seg_v[i] = leaf(kv_type(s), hd, seg_len[i], H);
    }

    auto ln_mod = [&](ggml_tensor * x, ggml_tensor * mod, int which) {
        g->ln_params.emplace_back(new LnModP{ which == 0 ? 0 : 3, which == 0 ? 1 : 4, kNormEps });
        ggml_tensor * args[2] = { x, mod };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, args, 2, op_ln_mod,
                              GGML_N_TASKS_MAX, g->ln_params.back().get());
    };
    auto add_gate = [&](ggml_tensor * res, ggml_tensor * x, ggml_tensor * mod, int gate_idx) {
        g->gate_params.emplace_back(new GateP{ gate_idx });
        ggml_tensor * args[3] = { res, x, mod };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, res->ne[0], res->ne[1], 1, 1, args, 3, op_add_gate,
                              GGML_N_TASKS_MAX, g->gate_params.back().get());
    };
    auto linear = [&](ggml_tensor * W, ggml_tensor * B, ggml_tensor * x) {
        return ggml_add(ctx, ggml_mul_mat(ctx, W, x), B);
    };

    // --- attention inputs ---------------------------------------------------- //
    ggml_tensor * qkv[2] = {nullptr, nullptr};
    for (int st = 0; st < n_streams; st++) {
        qkv[st] = linear(g->W[st][QKV], g->B[st][QKV], ln_mod(g->x[st], g->mod[st], 0));
    }

    AttnP base{};
    base.D = D; base.H = H; base.hd = hd; base.Mi = Mi; base.Mt = Mt; base.eps = kNormEps;
    base.rope = true; base.f32 = s->p.kv_f32 != 0;

    // q
    std::vector<ggml_tensor *> a_q = { qkv[IMG], g->QN[IMG], g->cos, g->sin };
    {
        auto p = std::make_unique<AttnP>(base);
        p->a_qkv_img = 0; p->a_norm_img = 1; p->a_cos = 2; p->a_sin = 3;
        if (Mt) { p->a_qkv_txt = (int) a_q.size(); a_q.push_back(qkv[TXT]); p->a_norm_txt = (int) a_q.size(); a_q.push_back(g->QN[TXT]); }
        g->attn_params.push_back(std::move(p));
    }
    ggml_tensor * q = ggml_custom_4d(ctx, GGML_TYPE_F32, hd, Mi + Mt, H, 1, a_q.data(), (int) a_q.size(), op_q,
                                     GGML_N_TASKS_MAX, g->attn_params.back().get());

    // k / v
    auto make_kv = [&](bool is_k) {
        std::vector<ggml_tensor *> a = { qkv[IMG], g->KN[IMG], g->cos, g->sin };
        auto p = std::make_unique<AttnP>(base);
        p->is_k = is_k; p->Lc = Lc; p->n_seg = n_seg;
        p->a_qkv_img = 0; p->a_norm_img = 1; p->a_cos = 2; p->a_sin = 3;
        if (Mt) { p->a_qkv_txt = (int) a.size(); a.push_back(qkv[TXT]); p->a_norm_txt = (int) a.size(); a.push_back(g->KN[TXT]); }
        p->a_seg0 = (int) a.size();
        for (int i = 0; i < n_seg; i++) { p->seg_len[i] = seg_len[i]; a.push_back(is_k ? g->seg_k[i] : g->seg_v[i]); }
        g->attn_params.push_back(std::move(p));
        return ggml_custom_4d(ctx, kv_type(s), hd, Lc + Mi + Mt, H, 1, a.data(), (int) a.size(), op_kv,
                              GGML_N_TASKS_MAX, g->attn_params.back().get());
    };
    ggml_tensor * k = make_kv(true);
    ggml_tensor * v = make_kv(false);
    if (capture) {
        g->o_k = k;
        g->o_v = v;
        ggml_set_output(k);
        ggml_set_output(v);
    }

    if (store) {
        // un-roped (pre-rope) normalized key for the cache; the temporal position is applied at publish time
        ggml_tensor * a[4] = { qkv[IMG], g->KN[IMG], g->cos, g->sin };
        auto p = std::make_unique<AttnP>(base);
        p->is_k = true; p->rope = false; p->Mt = 0; p->Lc = 0; p->n_seg = 0;
        p->a_qkv_img = 0; p->a_norm_img = 1; p->a_cos = 2; p->a_sin = 3; p->a_seg0 = 4;
        g->attn_params.push_back(std::move(p));
        g->o_kstore = ggml_custom_4d(ctx, kv_type(s), hd, Mi, H, 1, a, 4, op_kv,
                                     GGML_N_TASKS_MAX, g->attn_params.back().get());
        g->o_vstore = v;
        ggml_set_output(g->o_kstore);
        ggml_set_output(g->o_vstore);
    }

    // --- joint attention -------------------------------------------------------- //
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) hd), 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, D, Mi + Mt);   // [hd, H, M] -> [D, M]

    for (int st = 0; st < n_streams; st++) {
        const int64_t M   = st == IMG ? Mi : Mt;
        const int64_t off = st == IMG ? 0 : Mi;
        ggml_tensor * a = (Mt == 0) ? attn
                                    : ggml_view_2d(ctx, attn, D, M, attn->nb[1], (size_t) off * attn->nb[1]);
        ggml_tensor * h1 = add_gate(g->x[st], linear(g->W[st][PROJ], g->B[st][PROJ], a), g->mod[st], 2);
        ggml_tensor * up = ggml_gelu_inplace(ctx, linear(g->W[st][UP], g->B[st][UP], ln_mod(h1, g->mod[st], 1)));
        g->o[st] = add_gate(h1, linear(g->W[st][DOWN], g->B[st][DOWN], up), g->mod[st], 5);
        ggml_set_output(g->o[st]);
    }

    g->gf = ggml_new_graph(ctx);
    for (int st = 0; st < n_streams; st++) ggml_build_forward_expand(g->gf, g->o[st]);
    if (store) {
        ggml_build_forward_expand(g->gf, g->o_kstore);
        ggml_build_forward_expand(g->gf, g->o_vstore);
    }
    g->ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(g->ga, g->gf)) return nullptr;
    return g;
}

void bind_block(joyedge_session * s, BlockGraph & g, int layer, int ti) {
    joyedge_model * m = s->m;
    const int pl = layer % (int) m->blocks.size();
    const Block & b = m->blocks[pl];
    const int n_streams = g.Mt > 0 ? 2 : 1;
    for (int st = 0; st < n_streams; st++) {
        g.x[st]->data   = st == IMG ? (void *) s->h_img.data() : (void *) s->h_txt.data();
        g.mod[st]->data = s->mods[ti][pl][st].data();
        for (int l = 0; l < 4; l++) {
            g.W[st][l]->data = b.w[st][l]->data;
            g.B[st][l]->data = b.b[st][l]->data;
        }
        g.QN[st]->data = b.qn[st]->data;
        g.KN[st]->data = b.kn[st]->data;
    }
    g.cos->data = s->cos.data();
    g.sin->data = s->sin.data();
    for (int i = 0; i < g.n_seg; i++) {
        g.seg_k[i]->data = s->slot[g.seg_slot[i]].k[layer].data();
        g.seg_v[i]->data = s->slot[g.seg_slot[i]].v[layer].data();
    }
}

// ---- positional tables ------------------------------------------------------- //

// cos/sin [hd/2, n] for a t-constant grid of h*w tokens, appended at token offset `at`.
void fill_rope(const joyedge_config & c, int pos, int gh, int gw, float source_id,
               std::vector<float> & cos, std::vector<float> & sin, int64_t at) {
    const int hd = c.hidden_size / c.heads_num;
    const int half = hd / 2;
    std::vector<double> freq(half), role(half, 0.0);
    std::vector<int> axis(half);
    int j = 0;
    for (int a = 0; a < 3; a++) {
        const int dim = c.rope_dims[a];
        for (int i = 0; i < dim / 2; i++, j++) {
            freq[j] = 1.0 / std::pow((double) c.rope_theta, (double) (2 * i) / (double) dim);
            axis[j] = a;
        }
    }
    const int role_dim = std::max(0, std::min(c.source_id_rope_dim, hd));
    for (int i = 0; i < role_dim / 2; i++) {
        role[i] = 1.0 / std::pow((double) c.source_id_rope_theta, (double) (2 * i) / (double) role_dim);
    }
    for (int y = 0; y < gh; y++) {
        for (int x = 0; x < gw; x++) {
            const int64_t mtok = at + (int64_t) y * gw + x;
            const double p[3] = { (double) pos, (double) y, (double) x };
            for (int jj = 0; jj < half; jj++) {
                const double ang = p[axis[jj]] * freq[jj] + (double) source_id * role[jj];
                cos[(size_t) (mtok * half + jj)] = (float) std::cos(ang);
                sin[(size_t) (mtok * half + jj)] = (float) std::sin(ang);
            }
        }
    }
}

// ---- forward passes ---------------------------------------------------------- //

// tokens = img_in([C, n] token-major)
void patch_embed(joyedge_session * s, const float * const * latents, int n_lat, int64_t n) {
    joyedge_model * m = s->m;
    const int64_t C = m->cfg.in_channels;
    std::vector<float> X((size_t) (C * n * n_lat));
    for (int li = 0; li < n_lat; li++) {
        const float * lat = latents[li];
        float * xo = X.data() + (size_t) li * C * n;
        for (int64_t c = 0; c < C; c++) {
            for (int64_t t = 0; t < n; t++) xo[t * C + c] = lat[c * n + t];
        }
    }
    s->h_img = run_linear(m, m->img_in_w, m->img_in_b, X.data(), n * n_lat);
}

// txt_slot >= 0: copy the text rows of every block's K/V into that pool slot.
int run_blocks(joyedge_session * s, BlockGraph & g, int ti, int txt_slot = -1) {
    joyedge_model * m = s->m;
    const int depth = m->cfg.depth;
    const size_t es = kv_es(s);
    const int64_t hd = m->cfg.hidden_size / m->cfg.heads_num;
    for (int l = 0; l < depth; l++) {
        const int pl = l % (int) m->blocks.size();
        stream_advise(m, pl);
        bind_block(s, g, l, ti);
        if (ggml_backend_graph_compute(m->backend, g.gf) != GGML_STATUS_SUCCESS) {
            return fail("graph compute failed");
        }
        memcpy(s->h_img.data(), g.o[IMG]->data, s->h_img.size() * sizeof(float));
        if (g.Mt > 0) memcpy(s->h_txt.data(), g.o[TXT]->data, s->h_txt.size() * sizeof(float));
        if (txt_slot >= 0 && g.capture && g.Mt > 0) {
            const int64_t H = m->cfg.heads_num, Lk = g.Lc + g.Mi + g.Mt;
            auto & sl = s->slot[(size_t) txt_slot];
            const size_t row = (size_t) (g.Mt * hd) * es;
            sl.k[l].resize(row * (size_t) H);
            sl.v[l].resize(row * (size_t) H);
            for (int64_t h = 0; h < H; h++) {
                const size_t src = (size_t) ((h * Lk + g.Lc + g.Mi) * hd) * es;
                memcpy(sl.k[l].data() + (size_t) h * row, (const uint8_t *) g.o_k->data + src, row);
                memcpy(sl.v[l].data() + (size_t) h * row, (const uint8_t *) g.o_v->data + src, row);
            }
        }
        if (g.store) {
            const size_t bytes = (size_t) (hd * g.Mi * m->cfg.heads_num) * es;
            s->stage_k[l].resize(bytes);
            s->stage_v[l].resize(bytes);
            memcpy(s->stage_k[l].data(), g.o_kstore->data, bytes);
            memcpy(s->stage_v[l].data(), g.o_vstore->data, bytes);
        }
    }
    return 0;
}

// noise prediction for one denoise step; x/src: [C, N] channel-major
int forward_denoise(joyedge_session * s, const float * x, const float * src, int ti, int pos, float * pred) {
    joyedge_model * m = s->m;
    const joyedge_config & c = m->cfg;
    const int64_t D = c.hidden_size, N = s->N, hd = D / c.heads_num;
    const int64_t Mi = 2 * N;

    const float * lats[2] = { x, src };
    patch_embed(s, lats, 2, N);
    s->h_txt = s->txt_tokens;

    s->cos.resize((size_t) (Mi * hd / 2));
    s->sin.resize((size_t) (Mi * hd / 2));
    fill_rope(c, pos, s->p.lat_h, s->p.lat_w, 0.0f, s->cos, s->sin, 0);   // target tokens
    fill_rope(c, pos, s->p.lat_h, s->p.lat_w, 1.0f, s->cos, s->sin, N);   // source-video condition

    int seg_slot[MAX_SEG]; int64_t seg_len[MAX_SEG]; int n_seg = 0;
    for (int sl : { SINK, PREV, REF }) {
        if (s->slot[sl].valid) { seg_slot[n_seg] = sl; seg_len[n_seg] = s->slot[sl].n; n_seg++; }
    }

    // Optional text-stream cache: every `txt_cache_interval` chunks the text stream is
    // evaluated jointly (exact) and its per-block K/V are kept; in between, image tokens
    // attend to those cached text keys and the text stream is not evaluated at all.
    const int interval = s->p.txt_cache_interval;
    auto & tslot = s->slot[(size_t) (TXT0 + ti)];
    const bool caching = interval > 1;
    const bool reuse = caching && tslot.valid && (s->chunk_idx - tslot.chunk) < interval;
    int64_t Mt = s->Mt;
    if (reuse) {
        seg_slot[n_seg] = TXT0 + ti; seg_len[n_seg] = tslot.n; n_seg++;
        Mt = 0;
    }
    const bool capture = caching && !reuse;

    auto & g = reuse ? s->g_denoise_lite : s->g_denoise;
    if (!g || !g->matches(Mi, Mt, false, capture, n_seg, seg_slot, seg_len)) {
        g.reset();
        g = build_block_graph(s, Mi, Mt, false, capture, n_seg, seg_slot, seg_len);
        if (!g) return fail("failed to allocate the denoise graph");
    }
    if (run_blocks(s, *g, ti, capture ? TXT0 + ti : -1) != 0) return -1;
    if (capture) { tslot.valid = true; tslot.n = s->Mt; tslot.chunk = s->chunk_idx; }

    // final layer on the target tokens only
    std::vector<float> normed((size_t) (D * N));
    parallel_for(m->n_threads, N, [&](int64_t lo, int64_t hi) {
        for (int64_t t = lo; t < hi; t++) {
            const float * xr = s->h_img.data() + t * D;
            float * yr = normed.data() + t * D;
            float sum = 0.0f;
            for (int64_t d = 0; d < D; d++) sum += xr[d];
            const float mean = sum / (float) D;
            float var = 0.0f;
            for (int64_t d = 0; d < D; d++) { const float cc = xr[d] - mean; var += cc * cc; }
            const float r = 1.0f / std::sqrt(var / (float) D + kNormEps);
            for (int64_t d = 0; d < D; d++) yr[d] = (xr[d] - mean) * r;
        }
    });
    const std::vector<float> y = run_linear(m, m->out_w, m->out_b, normed.data(), N);
    const int64_t Co = c.out_channels;
    for (int64_t t = 0; t < N; t++) {
        for (int64_t ch = 0; ch < Co; ch++) pred[ch * N + t] = y[(size_t) (t * Co + ch)];
    }
    return 0;
}

// clean-chunk pass (t = 0, image stream only, self attention only): stages pre-rope K and V.
int forward_store(joyedge_session * s, const float * x, int gh, int gw, int pos, float source_id) {
    joyedge_model * m = s->m;
    const joyedge_config & c = m->cfg;
    const int64_t n = (int64_t) gh * gw, hd = c.hidden_size / c.heads_num;

    const float * lats[1] = { x };
    patch_embed(s, lats, 1, n);
    s->cos.resize((size_t) (n * hd / 2));
    s->sin.resize((size_t) (n * hd / 2));
    fill_rope(c, pos, gh, gw, source_id, s->cos, s->sin, 0);

    if (!s->g_store || !s->g_store->matches(n, 0, true, false, 0, nullptr, nullptr)) {
        s->g_store.reset();
        s->g_store = build_block_graph(s, n, 0, true, false, 0, nullptr, nullptr);
        if (!s->g_store) return fail("failed to allocate the store graph");
    }
    s->stage_n = n;
    return run_blocks(s, *s->g_store, s->p.n_steps);
}

// Publish the staged chunk into a pool slot: rope the pre-rope keys with the
// position the chunk will occupy in future windows, take the values as they are.
void publish(joyedge_session * s, int sl, int pos, int gh, int gw, float source_id) {
    joyedge_model * m = s->m;
    const joyedge_config & c = m->cfg;
    const int64_t H = c.heads_num, hd = c.hidden_size / H, n = s->stage_n, half = hd / 2;
    const bool f32 = s->p.kv_f32 != 0;
    const size_t es = kv_es(s);

    std::vector<float> cos((size_t) (n * half)), sin((size_t) (n * half));
    fill_rope(c, pos, gh, gw, source_id, cos, sin, 0);

    auto & slot = s->slot[sl];
    slot.n = n;
    parallel_for(m->n_threads, c.depth, [&](int64_t lo, int64_t hi) {
        std::vector<float> tmp((size_t) hd), rot((size_t) hd);
        for (int64_t l = lo; l < hi; l++) {
            slot.k[l].resize(s->stage_k[l].size());
            const uint8_t * src = s->stage_k[l].data();
            uint8_t * dst = slot.k[l].data();
            for (int64_t h = 0; h < H; h++) {
                for (int64_t t = 0; t < n; t++) {
                    const size_t off = (size_t) ((h * n + t) * hd) * es;
                    const float * x;
                    if (f32) x = (const float *) (src + off);
                    else { ggml_fp16_to_fp32_row((const ggml_fp16_t *) (src + off), tmp.data(), hd); x = tmp.data(); }
                    const float * cc = cos.data() + t * half;
                    const float * ss = sin.data() + t * half;
                    for (int64_t j = 0; j < half; j++) {
                        rot[2 * j]     = x[2 * j] * cc[j] - x[2 * j + 1] * ss[j];
                        rot[2 * j + 1] = x[2 * j] * ss[j] + x[2 * j + 1] * cc[j];
                    }
                    put_row(dst + off, rot.data(), hd, f32);
                }
            }
            slot.v[l] = s->stage_v[l];
        }
    });
    slot.valid = true;
}

std::vector<float> time_modulation(joyedge_model * m, float t) {
    const int F = m->time_freq_dim, half = F / 2;
    std::vector<float> emb((size_t) F);
    for (int i = 0; i < half; i++) {
        const float f = std::exp(-std::log(10000.0f) * (float) i / (float) half);
        emb[i]        = std::cos(t * f);   // flip_sin_to_cos = True
        emb[half + i] = std::sin(t * f);
    }
    std::vector<float> h = run_linear(m, m->te1_w, m->te1_b, emb.data(), 1, ACT_SILU);
    std::vector<float> temb = run_linear(m, m->te2_w, m->te2_b, h.data(), 1);
    silu_inplace(temb);
    return run_linear(m, m->tp_w, m->tp_b, temb.data(), 1);   // [6 * D]
}

double now_s() { return (double) ggml_time_us() * 1e-6; }

} // namespace

// --------------------------------------------------------------------------- //
// public API
// --------------------------------------------------------------------------- //

const char * joyedge_last_error(void) { return g_err.c_str(); }

joyedge_model_params joyedge_model_default_params(void) {
    joyedge_model_params p{};
    p.n_threads = 0;
    p.stream_window = 0;
    p.verbose = 0;
    return p;
}

joyedge_session_params joyedge_session_default_params(void) {
    joyedge_session_params p{};
    p.lat_h = 20;
    p.lat_w = 35;
    p.n_steps = 2;
    p.max_temporal_ids = 2;
    p.shift = 0.0f;
    p.kv_f32 = 0;
    p.txt_cache_interval = 0;
    return p;
}

void joyedge_model_free(joyedge_model * m) {
    if (!m) return;
    if (m->backend) ggml_backend_free(m->backend);
    if (m->tp) ggml_threadpool_free(m->tp);
    if (m->meta) ggml_free(m->meta);
    if (m->gguf) gguf_free(m->gguf);
#if defined(JOYEDGE_HAVE_MMAP)
    if (m->is_mmap && m->map_base) munmap(m->map_base, m->map_size);
    if (m->fd >= 0) close(m->fd);
#endif
    if (!m->is_mmap) free(m->map_base);
    delete m;
}

joyedge_model * joyedge_model_load(const char * path, joyedge_model_params params) {
    ggml_time_init();
    auto * m = new joyedge_model();
    m->params = params;

    gguf_init_params gp = { /*no_alloc*/ true, &m->meta };
    m->gguf = gguf_init_from_file(path, gp);
    if (!m->gguf) { fail(std::string("cannot read GGUF file: ") + path); delete m; return nullptr; }

    auto u32 = [&](const char * key, uint32_t def, bool required) -> uint32_t {
        const int64_t id = gguf_find_key(m->gguf, key);
        if (id < 0) { if (required) fail(std::string("missing metadata key: ") + key); return def; }
        return gguf_get_val_u32(m->gguf, id);
    };
    auto f32 = [&](const char * key, float def) -> float {
        const int64_t id = gguf_find_key(m->gguf, key);
        return id < 0 ? def : gguf_get_val_f32(m->gguf, id);
    };

    g_err.clear();
    joyedge_config & c = m->cfg;
    c.hidden_size  = (int32_t) u32("joyedge.hidden_size", 0, true);
    c.heads_num    = (int32_t) u32("joyedge.heads_num", 0, true);
    c.depth        = (int32_t) u32("joyedge.depth", 0, true);
    c.in_channels  = (int32_t) u32("joyedge.in_channels", 0, true);
    c.out_channels = (int32_t) u32("joyedge.out_channels", 0, true);
    c.text_dim     = (int32_t) u32("joyedge.text_dim", 0, true);
    c.mlp_hidden   = (int32_t) u32("joyedge.mlp_hidden", 0, true);
    c.source_id_rope_dim   = (int32_t) u32("joyedge.source_id_rope_dim", 128, false);
    c.num_train_timesteps  = (int32_t) u32("joyedge.num_train_timesteps", 1000, false);
    m->time_freq_dim       = (int) u32("joyedge.time_freq_dim", 256, false);
    c.rope_theta           = f32("joyedge.rope_theta", 256.0f);
    c.source_id_rope_theta = f32("joyedge.source_id_rope_theta", 256.0f);
    c.scheduler_shift      = f32("joyedge.scheduler_shift", 5.159f);
    {
        const int64_t id = gguf_find_key(m->gguf, "joyedge.rope_dims");
        if (id < 0 || gguf_get_arr_n(m->gguf, id) != 3 || gguf_get_arr_type(m->gguf, id) != GGUF_TYPE_INT32) {
            fail("missing metadata key: joyedge.rope_dims (int32[3])");
        } else {
            const int32_t * d = (const int32_t *) gguf_get_arr_data(m->gguf, id);
            for (int i = 0; i < 3; i++) c.rope_dims[i] = d[i];
        }
    }
    if (!g_err.empty()) { joyedge_model_free(m); return nullptr; }
    const int hd = c.hidden_size / c.heads_num;
    if (c.rope_dims[0] + c.rope_dims[1] + c.rope_dims[2] != hd || hd % 2 != 0) {
        fail("rope_dims must sum to the attention head dim");
        joyedge_model_free(m);
        return nullptr;
    }

    // ---- map the weights ---------------------------------------------------- //
#if defined(JOYEDGE_HAVE_MMAP)
    m->fd = open(path, O_RDONLY);
    struct stat st{};
    if (m->fd < 0 || fstat(m->fd, &st) != 0) { fail("cannot open model file"); joyedge_model_free(m); return nullptr; }
    m->map_size = (size_t) st.st_size;
    void * base = mmap(nullptr, m->map_size, PROT_READ, MAP_SHARED, m->fd, 0);
    if (base == MAP_FAILED) { fail("mmap failed"); m->map_base = nullptr; joyedge_model_free(m); return nullptr; }
    m->map_base = (uint8_t *) base;
    m->is_mmap = true;
#else
    {
        FILE * f = fopen(path, "rb");
        if (!f) { fail("cannot open model file"); joyedge_model_free(m); return nullptr; }
        fseek(f, 0, SEEK_END);
        m->map_size = (size_t) ftell(f);
        fseek(f, 0, SEEK_SET);
        m->map_base = (uint8_t *) malloc(m->map_size);
        const size_t got = m->map_base ? fread(m->map_base, 1, m->map_size, f) : 0;
        fclose(f);
        if (got != m->map_size) { fail("cannot read model file"); joyedge_model_free(m); return nullptr; }
    }
#endif
    c.file_bytes = m->map_size;

    const size_t data_off = gguf_get_data_offset(m->gguf);
    const int64_t n_tensors = gguf_get_n_tensors(m->gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor * t = ggml_get_tensor(m->meta, name);
        if (!t) continue;
        t->data = m->map_base + data_off + gguf_get_tensor_offset(m->gguf, i);
        c.n_params += (uint64_t) ggml_nelements(t);
    }

    auto get = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(m->meta, name.c_str());
        if (!t && required && g_err.empty()) fail("missing tensor: " + name);
        return t;
    };

    m->img_in_w = get("img_in.weight");
    m->img_in_b = get("img_in.bias");
    m->te1_w = get("condition_embedder.time_embedder.linear_1.weight");
    m->te1_b = get("condition_embedder.time_embedder.linear_1.bias");
    m->te2_w = get("condition_embedder.time_embedder.linear_2.weight");
    m->te2_b = get("condition_embedder.time_embedder.linear_2.bias");
    m->tp_w  = get("condition_embedder.time_proj.weight");
    m->tp_b  = get("condition_embedder.time_proj.bias");
    m->tx1_w = get("condition_embedder.text_embedder.linear_1.weight");
    m->tx1_b = get("condition_embedder.text_embedder.linear_1.bias");
    m->tx2_w = get("condition_embedder.text_embedder.linear_2.weight");
    m->tx2_b = get("condition_embedder.text_embedder.linear_2.bias");
    m->out_w = get("proj_out.weight");
    m->out_b = get("proj_out.bias");

    static const char * stream_name[2] = { "img", "txt" };
    static const char * lin_name[4] = { "_attn_qkv", "_attn_proj", "_mlp.net.0.proj", "_mlp.net.2" };
    for (int l = 0; l < c.depth && g_err.empty(); l++) {
        const std::string pre = "double_blocks." + std::to_string(l) + ".";
        if (!ggml_get_tensor(m->meta, (pre + "img_attn_qkv.weight").c_str())) break;
        Block b;
        for (int st = 0; st < 2; st++) {
            const std::string sp = pre + stream_name[st];
            b.mod[st] = get(sp + "_mod.modulate_table");
            b.qn[st]  = get(sp + "_attn_q_norm.weight");
            b.kn[st]  = get(sp + "_attn_k_norm.weight");
            for (int li = 0; li < 4; li++) {
                b.w[st][li] = get(sp + lin_name[li] + ".weight");
                b.b[st][li] = get(sp + lin_name[li] + ".bias");
            }
        }
        if (!g_err.empty()) break;
        for (int st = 0; st < 2; st++) {
            ggml_tensor * all[] = { b.mod[st], b.qn[st], b.kn[st], b.w[st][0], b.w[st][1], b.w[st][2], b.w[st][3],
                                    b.b[st][0], b.b[st][1], b.b[st][2], b.b[st][3] };
            for (ggml_tensor * t : all) {
                const size_t off = (size_t) ((uint8_t *) t->data - m->map_base);
                b.lo = std::min(b.lo, off);
                b.hi = std::max(b.hi, off + ggml_nbytes(t));
            }
        }
        m->blocks.push_back(b);
    }
    if (g_err.empty() && m->blocks.empty()) fail("no DiT blocks in file");
    if (g_err.empty()) {
        // every block must share block 0's storage types: one graph is re-bound for all blocks
        for (const Block & b : m->blocks) {
            for (int st = 0; st < 2; st++) for (int li = 0; li < 4; li++) {
                if (b.w[st][li]->type != m->blocks[0].w[st][li]->type) fail("blocks use mixed quantization types");
            }
        }
    }
    if (!g_err.empty()) { joyedge_model_free(m); return nullptr; }
    c.depth_in_file = (int32_t) m->blocks.size();

    m->n_threads = params.n_threads > 0 ? params.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    m->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(m->n_threads);
    m->tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(m->backend, m->tp);

    if (params.verbose) {
        fprintf(stderr,
                "joyedge: %s\n  hidden=%d heads=%d depth=%d (stored %d) mlp=%d ch=%d->%d text=%d\n"
                "  params=%.2fM file=%.1f MiB block-weights=%s threads=%d stream_window=%d mmap=%d\n",
                path, c.hidden_size, c.heads_num, c.depth, c.depth_in_file, c.mlp_hidden, c.in_channels,
                c.out_channels, c.text_dim, (double) c.n_params / 1e6, (double) c.file_bytes / (1024.0 * 1024.0),
                ggml_type_name(m->blocks[0].w[IMG][UP]->type), m->n_threads, params.stream_window, (int) m->is_mmap);
    }
    return m;
}

void joyedge_model_get_config(const joyedge_model * model, joyedge_config * out) {
    if (model && out) *out = model->cfg;
}

joyedge_session * joyedge_session_new(joyedge_model * m, joyedge_session_params p) {
    if (!m) { fail("null model"); return nullptr; }
    if (p.lat_h <= 0 || p.lat_w <= 0 || p.n_steps < 1) { fail("invalid session params"); return nullptr; }
    if (p.max_temporal_ids < 1) p.max_temporal_ids = 2;
    if (p.shift <= 0.0f) p.shift = m->cfg.scheduler_shift;

    auto * s = new joyedge_session();
    s->m = m;
    s->p = p;
    s->N = (int64_t) p.lat_h * p.lat_w;

    // flow-match schedule with SD3 time shift
    s->sigmas.resize((size_t) p.n_steps + 1);
    s->timesteps.resize((size_t) p.n_steps);
    for (int i = 0; i <= p.n_steps; i++) {
        const float lin = 1.0f - (float) i / (float) p.n_steps;
        s->sigmas[i] = p.shift * lin / (1.0f + (p.shift - 1.0f) * lin);
    }
    for (int i = 0; i < p.n_steps; i++) s->timesteps[i] = s->sigmas[i] * (float) m->cfg.num_train_timesteps;

    // modulation tables: one per (timestep, block, stream); the last entry is t = 0 (store pass)
    const int64_t D = m->cfg.hidden_size;
    const int P = (int) m->blocks.size();
    s->mods.resize((size_t) p.n_steps + 1);
    for (int ti = 0; ti <= p.n_steps; ti++) {
        const std::vector<float> vec = time_modulation(m, ti < p.n_steps ? s->timesteps[ti] : 0.0f);
        s->mods[ti].resize((size_t) P);
        for (int pl = 0; pl < P; pl++) {
            s->mods[ti][pl].resize(2);
            for (int st = 0; st < 2; st++) {
                const float * table = (const float *) m->blocks[pl].mod[st]->data;
                auto & dst = s->mods[ti][pl][st];
                dst.resize((size_t) (6 * D));
                for (int64_t i = 0; i < 6 * D; i++) dst[(size_t) i] = table[i] + vec[(size_t) i];
            }
        }
    }

    s->slot.resize((size_t) (TXT0 + p.n_steps));
    for (auto & sl : s->slot) { sl.k.resize((size_t) m->cfg.depth); sl.v.resize((size_t) m->cfg.depth); }
    s->stage_k.resize((size_t) m->cfg.depth);
    s->stage_v.resize((size_t) m->cfg.depth);
    return s;
}

void joyedge_session_free(joyedge_session * s) { delete s; }

void joyedge_session_reset(joyedge_session * s) {
    if (!s) return;
    for (auto & sl : s->slot) { sl.valid = false; sl.n = 0; sl.chunk = -1; }
    s->stage_pending = false;
    s->stage_chunk = -1;
    s->chunk_idx = 0;
}

int joyedge_set_prompt(joyedge_session * s, const float * embeds, int32_t n_tokens) {
    if (!s || !embeds || n_tokens <= 0) return fail("invalid prompt");
    joyedge_model * m = s->m;
    std::vector<float> h = run_linear(m, m->tx1_w, m->tx1_b, embeds, n_tokens, ACT_GELU);
    s->txt_tokens = run_linear(m, m->tx2_w, m->tx2_b, h.data(), n_tokens);
    s->Mt = n_tokens;
    for (size_t i = TXT0; i < s->slot.size(); i++) { s->slot[i].valid = false; s->slot[i].chunk = -1; }
    return 0;
}

int joyedge_set_reference_image(joyedge_session * s, const float * latent, int32_t h, int32_t w) {
    if (!s || !latent || h <= 0 || w <= 0) return fail("invalid reference image");
    if (s->chunk_idx != 0) return fail("the reference image must be set before the first chunk");
    // source id 2, temporal position 0; keys are published with that same rotation baked in
    if (forward_store(s, latent, h, w, 0, 2.0f) != 0) return -1;
    publish(s, REF, 0, h, w, 2.0f);
    return 0;
}

int joyedge_process_chunk(joyedge_session * s, const float * noise, const float * src_latent,
                          float * out_latent, int32_t update_history) {
    if (!s || !noise || !src_latent || !out_latent) return fail("null argument");
    if (s->Mt <= 0) return fail("joyedge_set_prompt must be called first");
    const joyedge_model * m = s->m;
    const int64_t CN = (int64_t) m->cfg.in_channels * s->N;
    const int k = s->chunk_idx;
    const int mti = s->p.max_temporal_ids;
    const double t0 = now_s();
    s->tm = joyedge_timings{};

    // roll the window: the chunk staged last time becomes "previous" unless the caller froze it
    double commit_s = 0.0;
    if (s->stage_pending && update_history) {
        const double tc = now_s();
        publish(s, PREV, std::min(s->stage_chunk, mti - 1), s->p.lat_h, s->p.lat_w, 0.0f);
        s->stage_pending = false;
        commit_s += now_s() - tc;
    }

    const int pos = std::min(k, mti);
    std::vector<float> x(noise, noise + CN), pred((size_t) CN);

    const double td = now_s();
    for (int i = 0; i < s->p.n_steps; i++) {
        if (forward_denoise(s, x.data(), src_latent, i, pos, pred.data()) != 0) return -1;
        const float dt = s->sigmas[(size_t) i + 1] - s->sigmas[(size_t) i];
        for (int64_t j = 0; j < CN; j++) x[(size_t) j] += pred[(size_t) j] * dt;
    }
    s->tm.denoise_s = now_s() - td;
    memcpy(out_latent, x.data(), (size_t) CN * sizeof(float));

    const double ts = now_s();
    if (forward_store(s, x.data(), s->p.lat_h, s->p.lat_w, pos, 0.0f) != 0) return -1;
    s->tm.store_s = now_s() - ts;

    const double tc = now_s();
    if (k == 0) {
        publish(s, SINK, 0, s->p.lat_h, s->p.lat_w, 0.0f);   // the first chunk is the global sink
    } else {
        s->stage_pending = true;
        s->stage_chunk = k;
    }
    commit_s += now_s() - tc;
    s->tm.commit_s = commit_s;

    double kv = 0.0;
    for (const auto & sl : s->slot) for (size_t l = 0; l < sl.k.size(); l++) kv += (double) (sl.k[l].size() + sl.v[l].size());
    for (size_t l = 0; l < s->stage_k.size(); l++) kv += (double) (s->stage_k[l].size() + s->stage_v[l].size());
    s->tm.kv_bytes = kv;
    s->tm.graph_bytes = (double) ((s->g_denoise ? s->g_denoise->arena_bytes() : 0) + (s->g_denoise_lite ? s->g_denoise_lite->arena_bytes() : 0) +
                                  (s->g_store ? s->g_store->arena_bytes() : 0));
    s->tm.total_s = now_s() - t0;
    s->chunk_idx++;
    return 0;
}

int32_t joyedge_chunk_index(const joyedge_session * s) { return s ? s->chunk_idx : -1; }

void joyedge_get_timings(const joyedge_session * s, joyedge_timings * out) {
    if (s && out) *out = s->tm;
}
