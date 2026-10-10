#!/usr/bin/env python3
"""Measure backend sampling (-bs) on one GPU: generation speed and CPU use, with -bs off and on.

llama-bench does not sample tokens, so it cannot show this. This script starts llama-server,
sends the same request several times, and reads the speed from the server's own timings.
CPU use is the container cgroup CPU time during each request.
Needs Docker with cgroup v2 and the systemd cgroup driver.

  python3 tools/bench_sampling.py --image llama-server-rocm-fix --model /path/to/model.gguf --render renderD128
"""
import argparse, json, os, subprocess, time, urllib.request

PROMPT = "Count from 1 to 4000, one number on each line, and do not stop.\n1\n2\n3\n"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def cpu_usec(cid):
    for line in open(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/cpu.stat"):
        if line.startswith("usage_usec"):
            return int(line.split()[1])


def ask(port, n_predict):
    body = json.dumps({"prompt": PROMPT, "n_predict": n_predict, "seed": 1, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
                       "min_p": 0.0, "ignore_eos": True, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["timings"]


def run_arm(name, bs, a):
    model_dir, model_file = os.path.split(os.path.abspath(a.model))
    groups = " ".join(f"--group-add {sh(f'getent group {g} | cut -d: -f3')}" for g in ("video", "render"))
    cid = sh(
        f"docker run -d -w /app --entrypoint /app/llama-server --device /dev/kfd --device /dev/dri/{a.render} {groups} "
        f"-p {a.port}:8080 -v {model_dir}:/models:ro -e LLAMA_ARG_BACKEND_SAMPLING={'1' if bs else '0'} {a.docker_args} {a.image} "
        f"-m /models/{model_file} --host 0.0.0.0 --port 8080 -t 1 -c 8192 -np 1 -ngl 999 --fit off --flash-attn on"
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except OSError:
                time.sleep(2)
        else:
            raise SystemExit(f"{name}: server did not start\n{sh(f'docker logs {cid} 2>&1 | tail -5')}")
        ask(a.port, 64)
        for i in range(a.requests):
            u0, t0 = cpu_usec(cid), time.time()
            timings = ask(a.port, a.tokens)
            u1, t1 = cpu_usec(cid), time.time()
            print(f"{name:20s} request {i + 1}  {timings['predicted_per_second']:7.2f} tok/s  cpus={(u1 - u0) / 1e6 / (t1 - t0):.2f}", flush=True)
    finally:
        sh(f"docker rm -f {cid}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--render", default="renderD128")
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--tokens", type=int, default=800)
    p.add_argument("--requests", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--docker-args", default="")
    a = p.parse_args()
    for n in range(1, a.rounds + 1):
        run_arm(f"round {n} -bs off", False, a)
        run_arm(f"round {n} -bs on", True, a)


main()
