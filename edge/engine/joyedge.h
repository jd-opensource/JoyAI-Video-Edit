// joyedge - CPU streaming runtime for the JoyAI-Video-Edit autoregressive MMDiT.
//
// A GGUF/ggml engine in the spirit of llama.cpp / Edge0: quantized weights are
// memory-mapped and paged in block by block, so the resident set is bounded by a
// sliding window of DiT blocks instead of the whole model.
//
// The API is plain C so it can be driven from Python (ctypes), Swift, Kotlin/JNI ...
#ifndef JOYEDGE_H
#define JOYEDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define JOYEDGE_API __declspec(dllexport)
#else
#  define JOYEDGE_API __attribute__((visibility("default")))
#endif

typedef struct joyedge_model   joyedge_model;
typedef struct joyedge_session joyedge_session;

typedef struct {
    int32_t n_threads;      // <= 0: std::thread::hardware_concurrency()
    int32_t stream_window;  // > 0: keep at most this many DiT blocks resident and
                            //      read ahead the next one (weights streamed from disk).
                            //   0: plain mmap, the OS page cache decides.
    int32_t verbose;
} joyedge_model_params;

typedef struct {
    int32_t hidden_size;
    int32_t heads_num;
    int32_t depth;            // logical number of blocks
    int32_t depth_in_file;    // physical blocks stored (== depth except in benchmark files)
    int32_t in_channels;
    int32_t out_channels;
    int32_t text_dim;
    int32_t mlp_hidden;
    int32_t rope_dims[3];     // temporal, height, width
    float   rope_theta;
    int32_t source_id_rope_dim;
    float   source_id_rope_theta;
    float   scheduler_shift;
    int32_t num_train_timesteps;
    uint64_t n_params;
    uint64_t file_bytes;
} joyedge_config;

typedef struct {
    int32_t lat_h;             // latent rows of one chunk
    int32_t lat_w;             // latent cols of one chunk
    int32_t n_steps;           // denoise steps per chunk (2 for the released weights)
    int32_t max_temporal_ids;  // temporal position ceiling (2 short rollouts, 8 long-tuned)
    float   shift;             // flow-match shift; <= 0 takes the value stored in the file
    int32_t kv_f32;            // 1: keep the KV cache in fp32 (default fp16)
    int32_t txt_cache_interval; // 0/1: exact (text stream evaluated on every chunk).
                               // R > 1: evaluate it every R chunks and reuse its cached K/V in
                               //        between -- an approximation, validate quality before use.
} joyedge_session_params;

typedef struct {
    double denoise_s;   // all denoise forwards of the last chunk
    double store_s;     // clean-chunk KV store forward
    double commit_s;    // rope + publish into the KV pool
    double total_s;
    double kv_bytes;    // resident KV state
    double graph_bytes; // activation arena of the live graphs
} joyedge_timings;

JOYEDGE_API joyedge_model_params   joyedge_model_default_params(void);
JOYEDGE_API joyedge_session_params joyedge_session_default_params(void);

JOYEDGE_API joyedge_model * joyedge_model_load(const char * path, joyedge_model_params params);
JOYEDGE_API void            joyedge_model_free(joyedge_model * model);
JOYEDGE_API void            joyedge_model_get_config(const joyedge_model * model, joyedge_config * out);

JOYEDGE_API joyedge_session * joyedge_session_new(joyedge_model * model, joyedge_session_params params);
JOYEDGE_API void              joyedge_session_free(joyedge_session * s);

// Drop all KV state and restart at chunk 0 (prompt is kept).
JOYEDGE_API void joyedge_session_reset(joyedge_session * s);

// Condition embeddings from the MLLM encoder: row-major [n_tokens, text_dim] fp32.
JOYEDGE_API int joyedge_set_prompt(joyedge_session * s, const float * embeds, int32_t n_tokens);

// Optional reference image (RV2V): normalized VAE latent [C, h, w] fp32.
// Must be called after joyedge_set_prompt and before the first chunk.
JOYEDGE_API int joyedge_set_reference_image(joyedge_session * s, const float * latent, int32_t h, int32_t w);

// One streaming step.
//   noise      : [C, lat_h, lat_w] fp32, N(0, 1)
//   src_latent : [C, lat_h, lat_w] fp32, normalized VAE latent of the source chunk
//   out_latent : [C, lat_h, lat_w] fp32, edited latent
//   update_history : 0 keeps the previous-chunk KV slot frozen (static-scene anchor),
//                    1 is the normal rolling window.
JOYEDGE_API int joyedge_process_chunk(joyedge_session * s, const float * noise, const float * src_latent,
                                      float * out_latent, int32_t update_history);

JOYEDGE_API int32_t joyedge_chunk_index(const joyedge_session * s);
JOYEDGE_API void    joyedge_get_timings(const joyedge_session * s, joyedge_timings * out);
JOYEDGE_API const char * joyedge_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // JOYEDGE_H
