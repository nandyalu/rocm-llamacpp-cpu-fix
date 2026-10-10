// LD_PRELOAD shim: make HIP wait for the GPU with an interrupt (sleep), not a
// busy loop. Acts only in llama-server; other programs only load it.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define hipDeviceScheduleSpin 0x1
#define hipDeviceScheduleYield 0x2
#define hipDeviceScheduleBlockingSync 0x4

__attribute__((constructor)) static void hip_blocking_sync(void) {
    if (strcmp(program_invocation_short_name, "llama-server") != 0)
        return;
    const char *mode = getenv("HIP_SCHEDULE");  // blocking (default), yield or spin
    unsigned flags = hipDeviceScheduleBlockingSync;
    if (mode && !strcmp(mode, "yield"))
        flags = hipDeviceScheduleYield;
    else if (mode && !strcmp(mode, "spin"))
        flags = hipDeviceScheduleSpin;
    // llama-server loads the HIP backend later with dlopen, so load HIP now.
    void *hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
    if (!hip) {
        fprintf(stderr, "hip_blocking_sync: %s\n", dlerror());
        return;
    }
    int (*set_flags)(unsigned) = (int (*)(unsigned))dlsym(hip, "hipSetDeviceFlags");
    if (!set_flags) {
        fprintf(stderr, "hip_blocking_sync: hipSetDeviceFlags not found\n");
        return;
    }
    fprintf(stderr, "hip_blocking_sync: hipSetDeviceFlags(0x%x) = %d\n", flags, set_flags(flags));
}
