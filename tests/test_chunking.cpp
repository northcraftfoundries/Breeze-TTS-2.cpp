// plain-main unit test for resolve_chunk_max; pure function, no model needed.
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

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
