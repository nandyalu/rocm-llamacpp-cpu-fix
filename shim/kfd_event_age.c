// LD_PRELOAD fix for the ROCm event thread busy loop (ROCm 7.2.1, kernel 6.8).
//
// ROCr passes last_event_age = 1 to AMDKFD_IOC_WAIT_EVENTS on each call. KFD
// compares it with the event's real age; when they differ, the wait returns at
// once and KFD writes the real age back. ROCr drops that value, so the next
// call passes 1 again and the thread loops (~60k ioctls/s per busy GPU).
//
// This wrapper remembers the age that KFD writes back for each event id and
// puts it in place of the stale 1. A real new signal still raises the event's
// age above the remembered one, so the wait still returns at once for it.
// The table is per thread: a thread only re-uses ages that it saw itself,
// before it checked its signal values again. A shared table could give one
// thread an age that it did not see and make it miss a wakeup.
// Acts only in llama-server.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define AMDKFD_IOC_WAIT_EVENTS 0xC0184B0CUL
#define TABLE 64  // per thread; a waiter set has only a few events
// Safety net: no wait sleeps longer than this. If a wakeup is ever missed, the
// thread checks its signals again after at most this time. ROCr's wait loop
// calls the ioctl again after a timeout, so this changes nothing else.
#define MAX_WAIT_MS 100

struct kfd_ioctl_wait_events_args {
    uint64_t events_ptr;
    uint32_t num_events;
    uint32_t wait_for_all;
    uint32_t timeout;
    uint32_t wait_result;
};

struct kfd_event_data {
    union {
        uint64_t last_event_age;  // signal_event_data
        uint8_t raw[32];
    };
    uint64_t kfd_event_data_ext;
    uint32_t event_id;
    uint32_t pad;
};

_Static_assert(sizeof(struct kfd_event_data) == 48, "kfd_event_data size");
_Static_assert(sizeof(struct kfd_ioctl_wait_events_args) == 24, "wait_events_args size");

static __thread struct { uint32_t id; uint64_t age; } ages[TABLE];

static uint64_t *slot(uint32_t id, int add) {
    for (unsigned i = 0, h = id * 2654435761u; i < TABLE; i++) {
        unsigned k = (h + i) % TABLE;
        if (ages[k].age && ages[k].id == id)
            return &ages[k].age;
        if (!ages[k].age) {
            if (!add)
                return NULL;
            ages[k].id = id;
            return &ages[k].age;
        }
    }
    if (!add)
        return NULL;
    unsigned k = id % TABLE;  // full: drop an old entry
    ages[k].id = id;
    return &ages[k].age;
}
static int (*real_ioctl)(int, unsigned long, void *);
static int enabled = -1;

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (!real_ioctl)
        real_ioctl = (int (*)(int, unsigned long, void *))dlsym(RTLD_NEXT, "ioctl");
    if (enabled < 0) {
        enabled = strcmp(program_invocation_short_name, "llama-server") == 0;
        if (enabled)
            fprintf(stderr, "kfd_event_age: active\n");
    }
    if (!enabled || request != AMDKFD_IOC_WAIT_EVENTS || !arg)
        return real_ioctl(fd, request, arg);

    struct kfd_ioctl_wait_events_args *a = arg;
    if (a->timeout > MAX_WAIT_MS)
        a->timeout = MAX_WAIT_MS;
    struct kfd_event_data *d = (struct kfd_event_data *)(uintptr_t)a->events_ptr;
    for (uint32_t i = 0; i < a->num_events; i++) {
        if (d[i].last_event_age == 1) {
            uint64_t *known = slot(d[i].event_id, 0);
            if (known && *known > 1)
                d[i].last_event_age = *known;
        }
    }
    int ret = real_ioctl(fd, request, arg);
    int saved = errno;
    for (uint32_t i = 0; i < a->num_events; i++) {
        if (d[i].last_event_age > 1)
            *slot(d[i].event_id, 1) = d[i].last_event_age;
    }
    errno = saved;
    return ret;
}
