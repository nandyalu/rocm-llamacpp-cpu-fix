# How we found the two fixes

This is the order in which we found the causes, with the tools and the numbers. Dates are from 2026-10-08 and 2026-10-09. The system: seven RX 6600 cards, ROCm 7.2.1, in-tree amdgpu driver of Ubuntu kernel 6.8.0, no amdgpu-dkms, one `llama-server` for each card, `-ngl 999 -t 1`.

## 1. The symptom

Each `llama-server` used about 1.04 CPUs for the whole run, and the seven servers did 91% of all host instructions. The GPUs do the inference, so the CPU should be nearly idle. The IPC was 0.93 to 0.97. That is the pattern of a busy loop, not of real work. A count of "instructions for each token" is wrong here, because it measures the spin.

## 2. First look: where is the CPU time?

`perf record` on one server during an 18.6k-token prompt read:

- The main thread spent 51% of all samples in two ROCr signal wait loops in `libhsa-runtime64` (load a signal, compare, `rdtsc`, repeat). ggml, llama and HIP together used about 1.5%.
- A second thread (the ROCr events thread) used about 85% CPU, mostly in the kernel and amdgpu.

So there are two separate problems: a spin in the main thread, and a loop in the events thread.

## 3. Problem 1: the main thread spins

HIP waits for the GPU with a busy loop by default. The ROCm 7.2.1 build has three wait settings: `ROC_ACTIVE_WAIT_TIMEOUT`, `ROC_CPU_WAIT_FOR_SIGNAL` and `HSA_ENABLE_INTERRUPT`. We set `ROC_ACTIVE_WAIT_TIMEOUT=0`, and nothing changed. There is no environment setting for `hipDeviceScheduleBlockingSync`.

The fix is `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`. `llama-server` loads the HIP backend later with `dlopen`, so a plain call from a shim finds no HIP library through `RTLD_DEFAULT`. The shim has to `dlopen("libamdhip64.so.7")` itself. It does this in a constructor, only in a process named `llama-server`. See `shim/hip_blocking_sync.c`.

Result: the main thread went from 100% to 6 to 18% CPU.

## 4. Problem 2: the events thread loops

With the first shim in place, the events thread still used about 0.9 CPUs for each busy card. It was idle when no request ran. `strace` showed about 55,000 `AMDKFD_IOC_WAIT_EVENTS` calls a second, one every 14 µs. Each call returned at once. The thread did only 26 to 120 voluntary context switches a second, so it was not sleeping between calls.

**Which code registers the waits?** `tools/async_trace.c` wraps `hsa_amd_signal_async_handler` and counts the calls by caller. It showed one caller only: the CLR (`rocvirtual.cpp`, "Set Handler ... timestamp"), with condition "less than 1". It ran 3 to 4 times a second during a prompt read and about 173 times a second during generation, and each handler ran once. So the loop is not one wait for each handler. While one handler is pending, the driver wait returns at once, again and again.

**Why does the wait return at once?** We traced `kfd_wait_on_events` and `test_event_condition` in the kernel with `bpftrace` (`tools/kfd_wait_trace.sh` is a smaller version of that script). The result:

- ROCr passes `last_event_age = 1` for each waited signal event on every call.
- The real age of the event in the kernel was 3.
- The kernel sees that 1 differs from 3, activates the waiter at once, and writes the real age (3) back into the user buffer.
- ROCr throws that value away, so the next call passes 1 again.

This is a bug in the user-space ROCr call pattern, not in the kernel. The fix is `shim/kfd_event_age.c`. It wraps `ioctl`, remembers the age that the kernel writes back for each event id, and passes that age in place of the stale 1. The table is per thread, so no thread ever gets an age that it did not see itself.

Later we added a cap of 100 ms on each wait as a safety net, in case a wakeup is ever missed. We have not seen a missed wakeup.

## 5. Measurements of each step

Fixed-prompt tests on 2026-10-08. "gen" is 800 generated tokens for each request. "mix" is an 18k-token prompt plus 400 generated tokens. CPUs is the CPU time of all `llama-server` processes divided by the wall time. Host busy is the share of all host CPU threads.

| | 1 card, gen | 7 cards, gen | 7 cards, mix |
|---|---|---|---|
| No shim | 70 tok/s, 1.75 CPUs | 304 tok/s, 6.47 CPUs, host 86% | 80.9 tok/s, 6.93 CPUs, host 92% |
| Blocking sync only | 69 tok/s, 0.90 CPUs | 311 tok/s, 4.25 CPUs, host 58% | 85.4 tok/s, 5.35 CPUs, host 73% |
| Blocking sync + age fix | 60 tok/s, 0.30 CPUs | 296 tok/s, 0.99 CPUs, host 17% | 83.7 tok/s, 0.82 CPUs, host 13% |

The age fix is the larger of the two gains for the CPU. The last row was measured before we added the 100 ms cap.

## 6. The cost: one card alone gets slower

With the age fix, one card alone generated 60 tok/s, not 70. The cause is CPU C-state wake-up latency. The host has only three idle states: POLL, C1_ACPI (1 µs) and C2_ACPI (151 µs). Before the fix, the busy loop kept the core awake. After it, the core goes to sleep and takes time to wake for each event.

Proof: we wrote a 0 to `/dev/cpu_dma_latency` and kept the file open. One card then generated 69.5 tok/s, and the CPU use fell to 0.17 CPUs. For a deployment, a limit of about 20 µs keeps C1 and blocks C2, and avoids idle polling.

Under load with seven cards the effect did not show: the pool-wide speed was the same as before (see the table).

### The same cost on a new CPU (2026-10-18)

We moved the host to an i7-11700T (8 cores, 16 threads, `intel_pstate`, governor `powersave`, energy preference `balance_performance`). Its idle states are POLL, C1_ACPI (1 µs) and C2_ACPI (253 µs). The cost came back, and the C-state fix did not remove it.

Test: `llama-server` on one direct card, Gemma 4 E4B QAT with the MTP draft model, 128k context, f16 KV cache, `-t 1`. Each run sends the same 17.5k-token prompt with `cache_prompt: false` and a fixed seed, and generates 800 tokens. All runs gave the same tokens (452 of 1039 drafts accepted). There were three runs for each arm, and the spread was less than 1%. CPUs is the CPU time of the container divided by the request time.

| Arm | Prompt tok/s | Generation tok/s | CPUs |
|---|---|---|---|
| No shims | 922-926 | 68.1-68.4 | 1.93 |
| Both shims | 914 | 58.3 | 0.18 |
| Both shims, `/dev/cpu_dma_latency` = 20 µs | 911 | 59.8 | 0.17 |
| Both shims, energy preference `performance` | 928 | 67.0 | 0.10 |

- **The 20 µs limit blocked C2 and did not help.** We counted 0 C2 entries in 3 s, and the speed went up by only 2.6%.
- **The clock ramp is the cause on this CPU.** The idle cores ran at 800 to 1700 MHz. A thread that sleeps between GPU events wakes on a slow core, and `balance_performance` raises the clock too slowly. Without the shims, the spin loop keeps its core at full clock.
- **The energy preference `performance` gives back almost all of the speed** (67.0 against 68.4 tok/s), at 19 times less CPU than no shims. C2 stays on.
- **`-t 4` changed nothing**, with or without the shims (58.4 and 68.4 tok/s).

We did not measure the idle power of the host with each energy preference, because RAPL needs root.

A later check with the standard `llama-bench` (2026-10-10, one direct card, `-p 512 -n 128 -r 30`, build b11371) gave the same picture. Speed fell by about 5% for pp512 and 3% for tg128 (1161 to 1104 tok/s, 52.9 to 51.4 tok/s). CPU use of the container fell from 1.48 to 0.12 CPUs for pp512 and from 1.92 to 0.11 CPUs for tg128. The shim arm also had `HSA_ENABLE_SDMA=0`. The table and the method are in the README.

We also tried `-t 1`, `-t 4` and `-t 8` on one card, first with the real server and a draft model, then with `llama-bench`. Speed and CPU use did not change in either test, with or without the shims (the table is in the README). With `-ngl 999` the CPU threads have almost no work, so we keep `-t 1`.

## 7. A different problem that looked related: the stalls

During the same weeks, a card sometimes stopped answering for 60 s or more. Our proxy killed the `llama-server` after 60 s without a log line, about three times for each batch of 32 long analyses. Hangs also happened before any shim (2026-10-04 and 2026-10-05), so the shims did not cause them.

What we measured at a stall:

- The host side waits correctly. In one trace, the main thread polled `kfd_wait_on_events` about 195 times a second on one event with 3 ms timeouts. The event age was the same going in and coming out (2825), so the kernel never saw a signal. This is not a missed wakeup, and the age fix is not involved.
- The GPU is not off its link and not in a reset. The link was 16 GT/s x16, runtime status was active, AER counters were zero, and temperatures were 50 to 58 C.
- Fence info showed emitted equal to signaled on every ring, so no work was pending on the kernel rings.
- `tools/clock_sample.sh` showed the signature. A healthy card in generation: busy 95 to 99%, memory clock 875 MHz, 80 to 100 W. A stalled card: busy fixed at 49%, memory clock 96 MHz, power 18 to 23 W, core clock unchanged. In one second the card dropped from about 86% busy to 49%. The low memory clock follows from the stall. The meaning: half the GPU is busy waiting for something that never completes.
- `journalctl -k` showed no amdgpu, KFD or reset line.
- The VBIOS is not the cause. Six cards had one VBIOS and the seventh had another. Both stalled.

We then tried the usual fixes for Navi 2x cards. With `HSA_ENABLE_SDMA=0` the next batch of 32 analyses had 0 kills, and so did the batch after it (which also had `LLAMA_ARG_BACKEND_SAMPLING=1`). Before, there were 2 to 5 kills in each batch. Two clean batches in a row are suggestive. They do not prove the cause, because the second batch changed two settings at once. If your stalls come back, test `HSA_ENABLE_SDMA=1` alone.

## Test of `HSA_ENABLE_INTERRUPT=1` (2026-10-10)

A reader said that `HSA_ENABLE_INTERRUPT=1` can replace the shims. We tested it. It has no effect, because `1` is already the ROCr default.

Test: `llama-bench` (`-p 512 -n 128`, `-t 1`), one card, Gemma 4 E4B QAT, ROCm 7.2.1, 15 repetitions, two rounds. The three env-var arms have no shims. CPUs is the CPU time of the container divided by the test time.

| Arm | pp512 tok/s | tg128 tok/s | CPUs, pp512 / tg128 |
|---|---|---|---|
| No shims | 1160-1169 | 52.8-53.0 | 1.06 / 1.92 |
| No shims, `HSA_ENABLE_INTERRUPT=1` | 1157-1161 | 52.8-52.9 | 1.06 / 1.92 |
| No shims, `HSA_ENABLE_INTERRUPT=0` (control) | 1158-1160 | 52.9 | 3.00 / 3.00 |
| Both shims | 1097-1131 | 51.3-51.4 | 0.09-0.13 / 0.11 |

- **`=1` gives the same result as no setting.** Speed and CPU use are equal.
- **`=0` is the control.** It forces polling and raises CPU use to 3.00 CPUs. So the variable works, and interrupts were already on.
- **Neither spin loop is a polling setting.** The main thread spins in the HIP (CLR) wait loop, and the events thread loops on the stale `last_event_age`. Only the shims fix these.

We did not run this test with seven busy cards, because the result on one card is flat.

## Check of `tools/kfd_wait_trace.sh` (2026-10-10)

We ran the script as root for 25 s on the pool with both shims, while seven cards each generated a long reply. In one 5 s block:

- The ROCr events thread of each server made 48 calls, about 10 a second. Each call waited the full 100 ms (the cap), returned by timeout (result 1), and had the same event age in and out (3). Before the fix this thread made about 55,000 calls a second.
- The main thread of each server made about 400 calls a second with a 3 ms timeout. These return by timeout (result 1), so that is a poll, not a busy loop.
- One more thread of each server made about 500 calls a second and returned with result 0 (signalled), with an event age that rises (11501 in one block). These are real signals.
- All threads of all seven servers together made 33,519 calls in 5 s, about 960 a second for each server.

The same script on an idle pool prints empty blocks, because no process waits.

## Tools

- `tools/async_trace.c`: counts `hsa_amd_signal_async_handler` calls by caller. See the README.
- `tools/kfd_wait_trace.sh`: `bpftrace` script for the KFD waits. Needs root.
- `tools/clock_sample.sh`: one line a second for each card, from sysfs. Needs no root.
- `tools/bench_cpu.py`: `llama-bench` with CPU use for each test, with and without the shims.
- `tools/bench_sampling.py`: `llama-server` with `-bs` off and on, with speed and CPU use.
