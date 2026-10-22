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
// It can change as a page plays: in Firefox 158 on a desktop the same page's waits ended 140 ms past their time in its
// launcher and 0.1 ms past it once its game ran, or stayed late all through, play to play. So waits judged on time as
// the page starts are judged again from the waits in use as it plays (timedWaitEnded()).
//

#pragma once

#if defined(__EMSCRIPTEN_PTHREADS__)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>

#include <emscripten/emscripten.h>
#include <emscripten/threading.h>

namespace plume {
    // The judgement: what the first thread other than the page's to ask measured (three waits of 100 us, late if the
    // shortest took more than 4 ms), until timed waits in use turn it to late (timedWaitEnded()).
    struct WebTimedWaits {
        std::atomic<bool> measuring = false;
        std::atomic<bool> measured = false;
        std::atomic<bool> late = false;

        // For timedWaitEnded(), on one thread: the second being counted, its waits and late ones, and how many seconds in a
        // row most of the waits were late.
        double secondStart = 0.0;
        uint32_t secondWaits = 0;
        uint32_t secondLateWaits = 0;
        uint32_t lateSeconds = 0;
    };

    inline WebTimedWaits &webTimedWaits() {
        static WebTimedWaits waits;
        return waits;
    }

    // A timed wait in use, on the one thread that judges them, that ended overMs past its time. Waits judged on time turn
    // late once most of the waits judged in each of 10 seconds in a row ended more than 4 ms past their time, and stay
    // late. 10 seconds in a row: Pokemon Snap's page's in Firefox 158 ended late for 5-6 s as its game booted and then on
    // time, which is when timed waits are a page's fastest, or late from then to the end of the play. Never back to on
    // time: judged by other waits (woken or spun), timed waits were never seen on time again.
    inline void timedWaitEnded(double overMs) {
        WebTimedWaits &waits = webTimedWaits();
        if (waits.late) {
            return;
        }

        const double now = emscripten_get_now();
        if (waits.secondStart == 0.0) {
            waits.secondStart = now;
        }

        waits.secondWaits++;
        waits.secondLateWaits += (overMs > 4.0) ? 1 : 0;
        if ((now - waits.secondStart) < 1000.0) {
            return;
        }

        waits.lateSeconds = ((waits.secondLateWaits * 2) > waits.secondWaits) ? (waits.lateSeconds + 1) : 0;
        waits.secondStart = now;
        waits.secondWaits = 0;
        waits.secondLateWaits = 0;
        if (waits.lateSeconds >= 10) {
            waits.late = true;
            fprintf(stderr, "plume: most timed waits on a thread of this page ended over 4 ms late for 10 s, so waits off the page's thread are woken or spun from now on, not timed\n");
        }
    }

    inline bool timedWaitsEndLate();

    // emscripten_futex_wait(), judged (timedWaitEnded()) if it ended on its time: a wake that ends it sooner says nothing
    // of the timer. Off the page's thread, on the one that judges waits.
    inline int judgeTimedWait(volatile void *futex, uint32_t value, double maxWaitMs) {
        timedWaitsEndLate(); // The first judgement, measured.
        const double start = emscripten_get_now();
        const int result = emscripten_futex_wait(futex, value, maxWaitMs);
        if (result == -ETIMEDOUT) {
            timedWaitEnded(emscripten_get_now() - start - maxWaitMs);
        }

        return result;
    }

    // False on the page's thread until a thread other than the page's has asked, which the page's thread can't.
    inline bool timedWaitsEndLate() {
        WebTimedWaits &waits = webTimedWaits();
        if (waits.measured || emscripten_is_main_browser_thread()) {
            return waits.late;
        }

        static uint32_t futex = 0;
        double shortest = 1e9;
        for (int i = 0; i < 3; i++) {
            const double start = emscripten_get_now();
            emscripten_futex_wait(&futex, 0, 0.1);
            shortest = std::min(shortest, emscripten_get_now() - start);
        }

        bool notMeasuring = false;
        if (waits.measuring.compare_exchange_strong(notMeasuring, true)) {
            waits.late = (shortest > 4.0);
            waits.measured = true;
            if (shortest > 4.0) {
                fprintf(stderr, "plume: a timed wait of 100 us on a thread of this page takes %.1f ms, so waits off the page's thread are woken or spun, not timed\n", shortest);
            }
        }

        return waits.late;
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
