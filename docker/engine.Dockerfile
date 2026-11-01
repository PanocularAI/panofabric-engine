# syntax=docker/dockerfile:1.7
#
# panofabric-engine image — bakes the training engine (torch nightly + torchtitan +
# torchft + the recipe registry) into a container so a fresh node skips `make all`:
# no apt, no rustup, no torchft compile, no 2.5 GB torch download. The control
# plane just references it via `--engine-image`; this repo OWNS and publishes it.
#
# It mirrors `make all`, but ONCE at build time, and — crucially — builds torchtitan and
# torchft from the SIBLING CLONES, so the image matches the exact fork SHAs you have
# checked out. The forks arrive as NAMED BUILD CONTEXTS (they are no longer submodules,
# so they are not inside this repo's build context); docker/build-engine-image.sh wires
# them up. To refresh: check the forks out where you want them and rebuild.

ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# builder — toolchain (rust + protoc + python headers) to compile torchft; dropped
# from the runtime stage.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG PROTOC_VERSION=32.0
ARG CUDA_TAG=cu130

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:/root/.cargo/bin:$PATH
ENV VIRTUAL_ENV=/opt/panoengine/.venv
# `uv sync` targets the venv above instead of creating ./.venv in the build context.
ENV UV_PROJECT_ENVIRONMENT=/opt/panoengine/.venv

# Build deps mirror Makefile `setup-env` (minus tailscale, a host concern): build-essential
# + pkg-config + libssl-dev for torchft's maturin build, python headers, git, unzip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev ca-certificates curl wget git unzip \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# protoc (pinned, mirrors Makefile) — compiles torchft's protobufs.
RUN curl -fsSL -o /tmp/protoc.zip \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
    && unzip -o /tmp/protoc.zip -d /usr/local \
    && rm /tmp/protoc.zip

# Rust toolchain (torchft maturin build) + uv.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Venv built against the DISTRO python so it stays valid when copied into the runtime
# stage (same ubuntu base => identical /usr/bin/pythonX.Y at the symlink target).
RUN uv venv --python /usr/bin/python${PYTHON_VERSION} ${VIRTUAL_ENV}

# This repo is the build context; .dockerignore drops .venv/.git/outputs/ so only source
# travels. The forks come in as named contexts (see the header).
WORKDIR /build
COPY . /build

# The Makefile's `install`, once: one `uv sync` of the [train] extra plus the backend extra
# for CUDA_TAG (pyproject.toml pins torch per backend, so no separate torch step), with
# torchtitan + torchft built from the sibling clones -- bind-mounted from their named build
# contexts -- instead of their git pins. `rw` because setuptools and maturin write build
# metadata into the source dir (it lands in a scratch overlay, never in the checkout).
# --no-editable: /build is not the runtime stage's tree. This is where torchft compiles
# from source — paid once.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=torchtitan,target=/src/torchtitan,rw \
    --mount=type=bind,from=torchft,target=/src/torchft,rw \
    set -eu; \
    case "${CUDA_TAG}" in \
      cu130) extra=cu130 ;; rocm*) extra=rocm ;; cpu) extra=cpu ;; \
      *) echo "no backend extra for CUDA_TAG=${CUDA_TAG}" >&2; exit 1 ;; \
    esac; \
    uv sync --locked --no-editable --python /usr/bin/python${PYTHON_VERSION} \
      --extra train --extra "$extra" \
      --no-install-package torchtitan --no-install-package torchft \
    && uv pip install /src/torchtitan /src/torchft

# Park run_train.sh beside the venv so the whole engine travels as one /opt/panoengine tree.
RUN cp /build/run_train.sh /opt/panoengine/run_train.sh && chmod +x /opt/panoengine/run_train.sh

# ---------------------------------------------------------------------------
# runtime — slim: CUDA runtime libs + the baked engine, no toolchain.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS train

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/panoengine/.venv
# venv first on PATH so python/torchrun/torchft_lighthouse resolve straight from it.
ENV PATH=/opt/panoengine/.venv/bin:$PATH

# Runtime deps. python${PYTHON_VERSION} must match the builder so the venv's interpreter
# symlink (-> /usr/bin/pythonX.Y) is valid; libgomp1/libssl3 satisfy torch; iproute2
# provides `ip` for the launcher's NCCL/Gloo socket-interface detection.
#
# build-essential + python-dev are NOT optional here: torchtitan uses FlexAttention /
# torch.compile, so Triton + Inductor JIT-compile C/CUDA helpers ON THE NODE at runtime
# (gcc + libc headers + <Python.h>). Because the venv targets the DISTRO python, the
# headers must come from python${PYTHON_VERSION}-dev (uv's standalone python would bundle
# its own — the distro python does not). libcuda.so.1 comes from the host driver at run.
#
# openssh-server + rsync are SkyPilot's k8s requirement: it runs every task over SSH into
# the pod and rsyncs the workdir. Its pod-init `apt install openssh-server rsync` depends
# on reaching apt mirrors on every launch (flaky); pre-baking them makes that a no-op and
# guarantees /etc/ssh/sshd_config exists (its absence kills the pod at "apt-ssh-setup").
#
# libibverbs1 + ibverbs-providers + librdmacm1t64 are what NCCL needs to use InfiniBand.
# torch's bundled NCCL dlopen()s libibverbs.so.1 at runtime and, when it is missing,
# falls back to TCP sockets WITHOUT an error -- so a multi-node island placed on an
# InfiniBand fabric (network_tier=best, /dev/infiniband passed into the
# container) still synced over Ethernet. ibverbs-providers carries libmlx5, the driver
# for the Mellanox NICs those fabrics use. The cudnn-runtime base ships none of them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev build-essential \
        openssh-server rsync \
        libgomp1 libssl3 ca-certificates iproute2 \
        libibverbs1 ibverbs-providers librdmacm1t64 \
    && rm -rf /var/lib/apt/lists/*

# The whole engine: venv (torch + torchtitan + torchft + panoengine) + run_train.sh.
COPY --from=builder /opt/panoengine /opt/panoengine
WORKDIR /opt/panoengine

# Fail the build early if the baked engine can't even import (catches a broken nightly /
# ABI mismatch before the image ships). `import torch` does not need a GPU.
RUN python -c "import torch, torchtitan, torchft; print('baked torch', torch.__version__)"

# And fail it if NCCL could not open InfiniBand: the fallback is silent at runtime, so
# the build is the only place a missing verbs library is loud.
RUN ldconfig -p | grep -q 'libibverbs.so.1' \
    && ls /usr/lib/*/libibverbs/libmlx5-rdmav*.so >/dev/null \
    && echo 'InfiniBand verbs + mlx5 provider present'

# OCI labels: source -> THIS repo (so GHCR shows the engine's README, not the control
# plane's); the build script passes the resolved source SHAs.
ARG ENGINE_SHA=unknown
ARG TORCHTITAN_SHA=unknown
ARG TORCHFT_SHA=unknown
ARG CUDA_TAG=cu130
LABEL org.opencontainers.image.title="panofabric-engine (train)" \
      org.opencontainers.image.source="https://github.com/PanocularAI/panofabric-engine" \
      org.opencontainers.image.description="Training-only stage: torch nightly + torchtitan + torchft + panoengine. No vllm, no RL runtime, no serving plane — published as :<tag>-train." \
      ai.panocular.cuda="${CUDA_TAG}" \
      ai.panocular.ref.engine="${ENGINE_SHA}" \
      ai.panocular.ref.torchtitan="${TORCHTITAN_SHA}" \
      ai.panocular.ref.torchft="${TORCHFT_SHA}"


# =============================================================================
# STAGE 3 — the full node runtime: training + inference + the serving plane.
#
# Stop at `--target train` for the training-only image (no vLLM, ~6 GB smaller);
# a serving or RL island cannot run on that one.
# =============================================================================
FROM train AS full

ARG CUDA_TAG=cu130
ENV UV_PROJECT_ENVIRONMENT=/opt/panoengine/.venv

# The train stage ships no installer (uv lived only in the builder).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# vLLM, the RL runtime and the serving plane: the `rl` backend extra (cu130 + vllm/
# torchvision/torchmonarch/torchstore/flash-attn-3/...; see pyproject.toml for why each is
# pinned the way it is) plus [serve]. rl pins the same torch nightly as cu130, so torch is
# already satisfied and this layer holds only the additions.
#   --inexact: torchtitan/torchft were installed from the sibling clones in the builder and
#     are not mounted here; an exact sync would uninstall them.
#   --no-install-package: and must not reinstall them from their git pins (no `git`, no
#     Rust toolchain in this stage).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.,target=/src,rw \
    cd /src && uv sync --locked --inexact --no-editable \
      --extra train --extra rl --extra serve \
      --no-install-package torchtitan --no-install-package torchft

# Fail the build if anything can't import (no GPU needed): catches a vllm/torch mismatch
# and a torchft break from the torch bump before the image ships. The RL line is separate
# because it is the one that catches a torchmonarch built against a DIFFERENT torch — its
# rust bindings only resolve libtorch symbols at import, so a mismatched pin sails through
# installation and dies on the node.
RUN python -c "import torch, torchtitan, torchft, vllm, panoengine.serve; print('torch', torch.__version__)" \
 && python -c "import torchstore, renderers, flash_attn_interface, math_verify; \
from monarch.actor import ProcMesh; from monarch.spmd import setup_torch_elastic_env_async; \
import panoengine.train.rl.train, panoengine.train.rl.harness; print('rl stack ok')"

ARG ENGINE_SHA=unknown
ARG TORCHTITAN_SHA=unknown
ARG TORCHFT_SHA=unknown
LABEL org.opencontainers.image.title="panofabric-engine" \
      org.opencontainers.image.source="https://github.com/PanocularAI/panofabric-engine" \
      org.opencontainers.image.description="GPU-node runtime: baked training engine (torch nightly + torchtitan + torchft + panoengine) + vllm + the RL runtime + the serving plane." \
      ai.panocular.cuda="${CUDA_TAG}" \
      ai.panocular.ref.engine="${ENGINE_SHA}" \
      ai.panocular.ref.torchtitan="${TORCHTITAN_SHA}" \
      ai.panocular.ref.torchft="${TORCHFT_SHA}"
