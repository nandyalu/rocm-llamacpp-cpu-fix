// LD_PRELOAD tracer: count hsa_amd_signal_async_handler registrations by
// caller, and how often the handlers run. Logs a summary to stderr every 5 s.
// Load it together with hip_blocking_sync.so, which loads HIP with
// RTLD_GLOBAL so that HIP's calls bind to this wrapper.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { uint64_t handle; } hsa_signal_t;
typedef int64_t hsa_signal_value_t;
typedef bool (*hsa_amd_signal_handler)(hsa_signal_value_t, void *);
typedef int (*real_fn)(hsa_signal_t, int, hsa_signal_value_t, hsa_amd_signal_handler, void *);

#define MAX_CALLERS 32
static struct { void *ret; unsigned long n; int cond; hsa_signal_value_t value; } callers[MAX_CALLERS];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_ulong registered, fired, kept, last_registered, last_fired;
static double last_print;

struct wrap { hsa_amd_signal_handler fn; void *arg; };

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void report(void) {
    double t = now();
    if (t - last_print < 5)
        return;
    unsigned long r = registered, f = fired;
    if (r == last_registered && f == last_fired)
        return;
    double dt = last_print ? t - last_print : 5;
    fprintf(stderr, "async_trace: %.0f registrations/s, %.0f handler runs/s (%lu kept), total %lu\n",
            (r - last_registered) / dt, (f - last_fired) / dt, (unsigned long)kept, r);
    for (int i = 0; i < MAX_CALLERS && callers[i].ret; i++) {
        Dl_info info;
        const char *lib = "?";
        uintptr_t off = (uintptr_t)callers[i].ret;
        if (dladdr(callers[i].ret, &info) && info.dli_fname) {
            lib = info.dli_fname;
            off -= (uintptr_t)info.dli_fbase;
        }
        fprintf(stderr, "async_trace:   %8lu from %s+0x%lx (cond %d, value %ld)\n",
                callers[i].n, lib, (unsigned long)off, callers[i].cond, (long)callers[i].value);
    }
    last_print = t;
    last_registered = r;
    last_fired = f;
}

static bool trampoline(hsa_signal_value_t value, void *p) {
    struct wrap *w = p;
    fired++;
    if (pthread_mutex_trylock(&lock) == 0) {
        report();
        pthread_mutex_unlock(&lock);
    }
    bool keep = w->fn(value, w->arg);
    if (keep)
        kept++;
    else
        free(w);
    return keep;
}

int hsa_amd_signal_async_handler(hsa_signal_t signal, int cond, hsa_signal_value_t value,
                                 hsa_amd_signal_handler handler, void *arg) {
    static real_fn real;
    if (!real) {
        void *h = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
        real = h ? (real_fn)dlvsym(h, "hsa_amd_signal_async_handler", "ROCR_1") : NULL;
        if (!real) {
            fprintf(stderr, "async_trace: real hsa_amd_signal_async_handler not found\n");
            return 0x1000;  // HSA_STATUS_ERROR
        }
    }
    void *ret = __builtin_return_address(0);
    registered++;
    pthread_mutex_lock(&lock);
    for (int i = 0; i < MAX_CALLERS; i++) {
        if (!callers[i].ret || callers[i].ret == ret) {
            callers[i].ret = ret;
            callers[i].n++;
            callers[i].cond = cond;
            callers[i].value = value;
            break;
        }
    }
    report();
    pthread_mutex_unlock(&lock);
    if (!handler)
        return real(signal, cond, value, handler, arg);
    struct wrap *w = malloc(sizeof *w);
    w->fn = handler;
    w->arg = arg;
    return real(signal, cond, value, trampoline, w);
}
