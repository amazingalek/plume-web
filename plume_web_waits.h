//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// Whether a timed wait on a page's threads ends late, for code that would otherwise wait on a timer for something
// another thread doesn't wake it for. In Firefox on Windows every one (an Atomics.wait with a timeout) ends on the
// system timer's tick, 15.6 ms unless the browser's own process asks for a finer one (per process since Windows 10
// 2004): a sleep of 100 us from a page's thread took 15.6 ms there on a laptop, and 2 ms in Chrome. A futex wake still
// comes at once.
//

#pragma once

#if defined(__EMSCRIPTEN_PTHREADS__)

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include <emscripten/emscripten.h>
#include <emscripten/threading.h>

namespace plume {
    // Measured once, by the first thread other than the page's to ask: three waits of 100 us, late if the shortest took
    // more than 4 ms. False on the page's thread until then, which can't wait.
    inline bool timedWaitsEndLate() {
        static std::atomic<int> state = 0; // 0 not measured, 1 on time, 2 late.
        int measured = state;
        if ((measured != 0) || emscripten_is_main_browser_thread()) {
            return measured == 2;
        }

        static uint32_t futex = 0;
        double shortest = 1e9;
        for (int i = 0; i < 3; i++) {
            const double start = emscripten_get_now();
            emscripten_futex_wait(&futex, 0, 0.1);
            shortest = std::min(shortest, emscripten_get_now() - start);
        }

        measured = (shortest > 4.0) ? 2 : 1;
        int unmeasured = 0;
        if (state.compare_exchange_strong(unmeasured, measured) && (measured == 2)) {
            fprintf(stderr, "plume: a timed wait of 100 us on a thread of this page takes %.1f ms, so waits off the page's thread are woken or spun, not timed\n", shortest);
        }

        return state == 2;
    }
};

#else

namespace plume {
    // Without threads the page's is the only one, and it can't wait.
    inline bool timedWaitsEndLate() {
        return false;
    }
};

#endif
