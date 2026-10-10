# rocm-llamacpp-cpu-fix

Two `LD_PRELOAD` shims and two settings for `llama-server` on AMD GPUs with ROCm. They stop `llama-server` from using 100% of a CPU thread while the GPU does the work.

**ROCm 7 only.** `hip_blocking_sync.c` loads `libamdhip64.so.7`. On another ROCm major version the shim does nothing and prints an error to stderr.

> Note: This entire investigation and fix has been made with the help of **Claude Opus 5.5**.

## The problem

On the tested system, each busy GPU kept one host CPU thread at 100%. `llama-server` was only waiting for the GPU. With seven GPUs, the host load was 100%.

There were two causes.

1. HIP waited for the GPU in a busy loop. By default it spins.
2. The ROCr event thread looped in `AMDKFD_IOC_WAIT_EVENTS`. ROCr passes `last_event_age = 1` on each call. The kernel compares that value with the real age of the event. When they differ, the wait returns at once, and the kernel writes the real age back. ROCr drops that value, so the next call passes 1 again. The thread then made about 60,000 ioctl calls a second for each busy GPU.

## The fix

The story of how we found these two, with the measurements, is in [INVESTIGATION.md](INVESTIGATION.md).

- **`shim/hip_blocking_sync.c`** sets `hipDeviceScheduleBlockingSync` when the process starts. HIP then sleeps until the GPU signals. Set `HIP_SCHEDULE=yield` or `HIP_SCHEDULE=spin` to test the other modes.
- **`shim/kfd_event_age.c`** wraps `ioctl`. It remembers the age that the kernel writes back for each event, and passes it in place of the stale 1. A real new signal still raises the age, so the wait still returns for it. The table is per thread. A wait never lasts longer than 100 ms, so a missed wakeup costs 100 ms at most.

Each shim acts only in a process named `llama-server`. Any other program that loads it is not changed.

## Two more settings

- **`HSA_ENABLE_SDMA=0`** turns the DMA copy engine off. With it on, a GPU stopped answering for 60 s or more, about three times in each batch of 32 long analyses (8 kills in 2 batches). The GPU showed a stall: busy fixed at 49%, memory clock at 96 MHz, power at 18 to 23 W, no pending fence, no PCIe error. The cause is not proved to be SDMA. The next batch with SDMA off also had `LLAMA_ARG_BACKEND_SAMPLING=1`, and only one clean batch was seen before that setting was added. If a stall comes back, test `HSA_ENABLE_SDMA=1` alone.
- **`LLAMA_ARG_BACKEND_SAMPLING=1`** (the same as `-bs`) runs the token sampling on the GPU. See "Backend sampling" below. In our first test, with a draft model, generation was 26 to 43% faster on one GPU. The repeatable test below, without a draft model, gave 15%.

## Results

Test system: seven RX 6600 cards (`gfx1030`, with `HSA_OVERRIDE_GFX_VERSION=10.3.0`), ROCm 7.2.1 (the one in llama.cpp build b11371), host kernel 6.8, model `gemma4-e4b-it-qat` with a draft model. One `llama-server` for each card. The load was a batch of 32 analyses, each with several long requests.

| | Before | After |
|---|---|---|
| Host CPU load | 100% | 15 to 20% |
| Generation, all cards | about 330 tok/s | about 430 tok/s |
| One batch of 32 | 60 to 70 min | about 40 min |
| Stalls with a kill | 2 to 5 for each batch | 0 in two batches |

The shims, the SDMA setting and `-bs` were applied together or one after the other, so the table does not split the gain between them. In fixed-prompt tests on 2026-10-08, each shim step gave this CPU use for all `llama-server` processes with seven busy cards: no shim 6.47 CPUs, blocking sync only 4.25 CPUs, both shims 0.99 CPUs. See [INVESTIGATION.md](INVESTIGATION.md) for the table and for how we found the causes. Only the `-bs` gain was measured alone, on one GPU.

### Standard `llama-bench`

One card, `llama bench -p 512 -n 128 -r 30 -fa 1 -ngl 999`, with the same model and no draft model. The card is in a direct PCIe slot. The CPU figure is the CPU time of the whole container, measured from its cgroup during each test only (model load and warmup are not in it). Two rounds for each arm.

| | pp512 tok/s | pp512 CPUs | tg128 tok/s | tg128 CPUs |
|---|---|---|---|---|
| No shims, SDMA on | 1164.6 / 1158.4 | 1.48 / 1.48 | 52.92 / 52.94 | 1.92 / 1.92 |
| Both shims, SDMA off | 1106.6 / 1100.9 | 0.12 / 0.13 | 51.48 / 51.39 | 0.11 / 0.11 |

CPU use falls by a factor of about 12 for prompt processing and about 17 for generation. Speed falls by about 5% for prompt processing and about 3% for generation. The shim arm also has `HSA_ENABLE_SDMA=0`, so the small speed loss is not split between the shims and SDMA. `tools/bench_cpu.py` repeats this test.

### Backend sampling (`-bs`)

`llama-bench` does not sample tokens, so it cannot show `-bs`. `tools/bench_sampling.py` starts `llama-server` and sends the same request 5 times: 800 generated tokens, a fixed seed, temperature 1, top-k 64, top-p 0.95, no draft model. The speed comes from the server's own timings. The CPU figure is the container CPU time during each request. Both arms have both shims. Same direct card, two rounds.

| | tg tok/s (mean of 5) | CPUs |
|---|---|---|
| `-bs` off, round 1 / round 2 | 45.75 / 45.86 | 0.20 / 0.20 |
| `-bs` on, round 1 / round 2 | 52.77 / 52.70 | 0.13 / 0.13 |

Generation is 15% faster and the CPU use is a third lower. The spread inside each arm is under 1%. The gain is smaller than in our first test (26 to 43%). That test used a draft model, and the sampling step runs for both models there. The prompt in this test is also very predictable. Your gain can differ with other samplers and models. Backend sampling is marked experimental in `llama-server`.

### Thread count (`-t`)

The same test with `-t 1`, `-t 4` and `-t 8`, 20 repeats, two rounds each. The numbers are the mean of the two rounds.

| | pp512 tok/s | pp512 CPUs | tg128 tok/s | tg128 CPUs |
|---|---|---|---|---|
| No shims, `-t 1` | 1160.9 | 1.27 | 52.99 | 1.92 |
| No shims, `-t 4` | 1157.4 | 1.27 | 52.93 | 1.92 |
| No shims, `-t 8` | 1158.4 | 1.27 | 52.91 | 1.92 |
| Both shims, `-t 1` | 1107.7 | 0.12 | 51.46 | 0.11 |
| Both shims, `-t 4` | 1103.9 | 0.13 | 51.48 | 0.11 |
| Both shims, `-t 8` | 1105.2 | 0.12 | 51.48 | 0.11 |

Thread count changes nothing, with or without the shims. With `-ngl 999` all layers run on the GPU, and the CPU threads have almost no work. The CPU use comes from the wait loops, not from the thread pool, so more threads do not help. Use `-t 1`. The no-shim pp512 figure here (1.27 CPUs) is lower than in the table above (1.48 CPUs). The spin is steady, but the length of the measured window changes the figure a little.

The shims act only in a process named `llama-server`. `llama bench` has another name, so the script starts it with `exec -a llama-server`.

**A cost:** the speed loss above is the CPU core going to sleep while it waits. A deep idle state (C2, 151 µs wake-up) slows each wake-up. In our own fixed-prompt test with the real model settings, one card alone generated 60 tok/s with both shims, against 70 tok/s without. With seven busy cards the pool speed did not change. To get the speed back on a single card, write 0 to `/dev/cpu_dma_latency` and keep the file open. We measured 69.5 tok/s that way. A limit of about 20 µs keeps C1 and blocks C2.

**Not tested:** other GPUs, other ROCm versions, other kernels, and CUDA. The kernel bug is in the ROCr call pattern, so a different ROCm version can change or remove it. If your CPU is not at 100% while the GPU works, you do not need this.

## Use it

Build the image. It adds the shims and the two settings to the official ROCm image of llama.cpp.

```sh
docker build -t llama-server-rocm-fix .
```

Run it as you run the official image.

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -p 8080:8080 -v /path/to/models:/models:ro \
  llama-server-rocm-fix \
  -m /models/model.gguf -ngl 999 --flash-attn on --host 0.0.0.0
```

Add `-e HSA_OVERRIDE_GFX_VERSION=10.3.0` for an RX 6600 or another `gfx1032` card.

Look for these lines in the container log. They show that the shims are active.

```
hip_blocking_sync: hipSetDeviceFlags(0x4) = 0
kfd_event_age: active
```

To use the shims without Docker, build them and set `LD_PRELOAD` for `llama-server`.

```sh
mkdir -p /opt/shim
for f in hip_blocking_sync kfd_event_age; do
  gcc -O2 -shared -fPIC -o /opt/shim/$f.so shim/$f.c -ldl
done
LD_PRELOAD="/opt/shim/hip_blocking_sync.so /opt/shim/kfd_event_age.so" \
HSA_ENABLE_SDMA=0 LLAMA_ARG_BACKEND_SAMPLING=1 llama-server ...
```

`hip_blocking_sync.c` loads `libamdhip64.so.7`. With another ROCm major version, change the library name in the file.

## Optional: the tracing shim

`tools/async_trace.c` is not part of the fix. We used it to find the fix. It wraps `hsa_amd_signal_async_handler`, counts the calls by the code address of the caller, and prints a summary to stderr every 5 s. It showed us that one caller (the HIP runtime) registers all the handlers. See [INVESTIGATION.md](INVESTIGATION.md). Do not use it in production.

Load it after `hip_blocking_sync.so`. That shim loads HIP with `RTLD_GLOBAL`, so the HIP calls bind to the wrapper.

```sh
gcc -O2 -shared -fPIC -o /opt/shim/async_trace.so tools/async_trace.c -ldl -lpthread
LD_PRELOAD="/opt/shim/hip_blocking_sync.so /opt/shim/async_trace.so" llama-server ...
```

Four more tools are in `tools/`: `kfd_wait_trace.sh` (`bpftrace`, needs root, traces each kernel wait), `clock_sample.sh` (one line a second for each card, from sysfs, needs no root) `bench_cpu.py` (the `llama-bench` test with CPU use, see above) and `bench_sampling.py` (the `-bs` test).

## llama-server settings used in the measurements

`-np 1 -t 1 -b 512 -ub 512 -ngl 999 --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16`, with `-c 131072` for one request at a time on each card.

The effect of these settings on the CPU load was not measured one by one. A test of `-c` at 32768, 49152, 65536 and 131072 on 32 analyses showed no change in speed, and no failures.

## Risk

These shims change how a driver library waits for the GPU. Test them on your system before you depend on them. If a request hangs, remove `LD_PRELOAD` first.
