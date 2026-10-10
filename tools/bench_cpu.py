#!/usr/bin/env python3
"""Run `llama bench` on one GPU with and without the shims, and print speed and CPU use.

CPU use is the cgroup CPU time of the container during each test only.
Needs Docker with cgroup v2 and the systemd cgroup driver.

  python3 tools/bench_cpu.py --image llama-server-rocm-fix --model /path/to/model.gguf --render renderD128
"""
import argparse, json, os, subprocess, threading, time


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def run_arm(name, env, argv0, threads, a):
    model_dir, model_file = os.path.split(os.path.abspath(a.model))
    bench = f"exec -a {argv0} /app/llama bench -m /models/{model_file} -ngl 999 -fa 1 -p 512 -n 128 -r {a.reps} -t {threads} -o jsonl"
    groups = " ".join(f"--group-add {sh(f'getent group {g} | cut -d: -f3')}" for g in ("video", "render"))
    cid = sh(
        f"docker run -d -w /app --entrypoint bash --device /dev/kfd --device /dev/dri/{a.render} "
        f"{groups} -v {model_dir}:/models:ro {a.docker_args} {env} {a.image} -c '{bench}'"
    )
    cpu_file = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/cpu.stat"
    samples, stop = [], False

    def poll():
        while not stop:
            try:
                for line in open(cpu_file):
                    if line.startswith("usage_usec"):
                        samples.append((time.time(), int(line.split()[1])))
            except OSError:
                pass
            time.sleep(0.1)

    thread = threading.Thread(target=poll)
    thread.start()
    proc = subprocess.Popen(["docker", "logs", "-f", cid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rows = [(time.time(), json.loads(l)) for l in proc.stdout if l.startswith("{")]
    proc.wait()
    stop = True
    thread.join()
    sh(f"docker rm -f {cid}")

    def usage_at(t):
        for (t0, u0), (t1, u1) in zip(samples, samples[1:]):
            if t0 <= t <= t1:
                return u0 + (u1 - u0) * (t - t0) / (t1 - t0)

    for done, row in rows:
        seconds = sum(row["samples_ns"]) / 1e9
        start, end = usage_at(done - seconds), usage_at(done)
        test = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
        cpus = f"{(end - start) / 1e6 / seconds:.2f}" if start and end else "n/a"
        print(f"{name:24s} {test:6s} {row['avg_ts']:8.2f} tok/s  cpus={cpus}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="image built from the Dockerfile in this repo")
    p.add_argument("--model", required=True, help="path to a .gguf file")
    p.add_argument("--render", default="renderD128", help="render node of the card to test")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--threads", default="1", help="comma list of -t values, for example 1,4,8")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--docker-args", default="", help="extra docker run arguments, for example -e HSA_OVERRIDE_GFX_VERSION=10.3.0")
    a = p.parse_args()
    for n in range(1, a.rounds + 1):
        for t in a.threads.split(","):
            run_arm(f"r{n} t={t} no shims", "-e LD_PRELOAD= -e HSA_ENABLE_SDMA=1", "llama-bench", t, a)
            run_arm(f"r{n} t={t} both shims", "", "llama-server", t, a)


main()
