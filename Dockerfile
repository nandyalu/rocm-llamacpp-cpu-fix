# The official llama.cpp ROCm server image, with the two shims added.
# Build b11371 was measured. Measure again before you move to a newer build.
ARG BASE=ghcr.io/ggml-org/llama.cpp@sha256:e1260406fd0a8f76e8b340904033d0aff6568fa3ce1482ceb73efed35f126b2d
FROM ${BASE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY shim/ /opt/shim-src/
RUN mkdir -p /opt/shim \
 && for f in hip_blocking_sync kfd_event_age; do \
      gcc -O2 -shared -fPIC -o /opt/shim/$f.so /opt/shim-src/$f.c -ldl || exit 1; \
    done
ENV LD_PRELOAD="/opt/shim/hip_blocking_sync.so /opt/shim/kfd_event_age.so"

# Turn the DMA copy engine off. See the README.
ENV HSA_ENABLE_SDMA=0
# Sample tokens on the card, not on the host.
ENV LLAMA_ARG_BACKEND_SAMPLING=1
