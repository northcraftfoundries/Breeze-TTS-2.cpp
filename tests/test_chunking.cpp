// plain-main unit test for the chunk ceiling, the ramp and the flag rule; pure functions, no model needed.
#include "breeze/generation.h"
#include "test_util.h"

#include <cstdio>

using namespace breeze;

int main() {
    CHECK(resolve_chunk_max(0, true) == chunk_max_gpu);
    CHECK(resolve_chunk_max(0, false) == chunk_max_cpu);
    CHECK(resolve_chunk_max(-5, false) == chunk_max_cpu);
    CHECK(resolve_chunk_max(40, true) == 40);
    CHECK(resolve_chunk_max(40, false) == 40);

    // the ramp the docs describe: 4 frames first, a third more each flush, capped at the ceiling
    CHECK(first_chunk(4, chunk_max_gpu) == 4);
    CHECK(first_chunk(0, chunk_max_gpu) == 1);
    CHECK(first_chunk(40, chunk_max_gpu) == chunk_max_gpu);
    CHECK(first_chunk(40, chunk_max_cpu) == 40);
    const int gpu_ramp[] = { 4, 6, 9, 13, 18, 25, 25 };
    int c = first_chunk(4, chunk_max_gpu);
    for (int want : gpu_ramp) { CHECK(c == want); c = next_chunk(c, chunk_max_gpu); }
    const int cpu_ramp[] = { 4, 6, 9, 13, 18, 25, 34, 46, 60, 60 };
    c = first_chunk(4, chunk_max_cpu);
    for (int want : cpu_ramp) { CHECK(c == want); c = next_chunk(c, chunk_max_cpu); }

    CHECK(chunk_flags_error(4, 0) == nullptr);
    CHECK(chunk_flags_error(1, 60) == nullptr);
    CHECK(chunk_flags_error(0, 25) != nullptr);
    CHECK(chunk_flags_error(4, -1) != nullptr);

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
