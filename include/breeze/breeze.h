#ifndef BREEZE_H
#define BREEZE_H

#include <stddef.h>

#if defined(_WIN32)
#ifdef BREEZE_BUILD_SHARED
#define BREEZE_API __declspec(dllexport)
#else
#define BREEZE_API
#endif
#else
#define BREEZE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct breeze_context breeze_context;

// return 0 to continue, non-zero to stop generation
typedef int (*breeze_audio_cb)(const float * samples, int n_samples, void * user);

typedef struct {
    const char * text;
    const char * instruction;   // null for a neutral default
    const char * ref_text;      // null when not cloning
    const float * ref_audio;    // mono 24 kHz samples, null for voice design
    int ref_audio_len;
    float cfg_scale;
    int seed;
    int max_new_tokens;         // 0 uses the model default
    int split_chars;            // 0 uses the default, negative keeps long text in a single pass
    // sampling. zero on any of these keeps whatever the gguf was built with
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
} breeze_request;

BREEZE_API breeze_context * breeze_init(const char * gguf_path, int use_gpu);
BREEZE_API void breeze_free(breeze_context * ctx);
BREEZE_API int breeze_sample_rate(breeze_context * ctx);

// streams audio chunks to the callback; returns 0 on success
BREEZE_API int breeze_generate(breeze_context * ctx, const breeze_request * req,
                               breeze_audio_cb cb, void * user);

// convenience: generate and write a 16-bit PCM WAV; returns 0 on success
BREEZE_API int breeze_generate_wav(breeze_context * ctx, const breeze_request * req,
                                   const char * out_path);

// sets the number of CPU threads used by ctx; ignored on GPU. n <= 0 keeps the default.
// replaces the context's thread pool, so like generation it must not overlap another call on
// the same context. returns 0 on success, non zero on error (e.g. null ctx).
BREEZE_API int breeze_set_threads(breeze_context * ctx, int n_threads);

// pins to a cpu list, e.g. "0-15", or the literal "pcores" to auto detect the performance
// cores. on linux this pins the calling thread and every thread it starts afterwards, and
// breeze_init starts the compute threads, so call it before breeze_init on the thread that
// will call it;
// on windows it pins the whole process. either way it affects the host process, not just
// this library. it also sets the default thread count for contexts breeze_init creates
// afterwards to the number of cpus actually pinned (breeze_set_threads still overrides) --
// this applies to every context created from here on, no matter which thread creates it.
// returns 0 on success.
BREEZE_API int breeze_pin_cpus(const char * spec);

BREEZE_API const char * breeze_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
