# syntax=docker/dockerfile:1.7
#
# SymphonyLearn engine image — bakes the training engine (torch nightly + torchtitan +
# torchft + the models.* registry) into a container so a fresh node skips `make all`:
# no apt, no rustup, no torchft compile, no 2.5 GB torch download. The symphony control
# plane just references it via `--engine-image`; this repo OWNS and publishes it.
#
# It mirrors `make all`, but ONCE at build time, and — crucially — builds torchtitan and
# torchft from THIS checkout's submodules (uv pip install ./torchtitan ./torchft), so the
# image matches the exact submodule SHAs checked out here. To refresh: bump the submodules
# (git submodule update --remote …) and rebuild with docker/build-engine-image.sh.

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
ARG PYTORCH_BASE_URL=https://download.pytorch.org/whl/nightly

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:/root/.cargo/bin:$PATH
ENV VIRTUAL_ENV=/opt/symphony/.venv

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

# The repo (with submodules) is the build context; .dockerignore drops .venv/.git/outputs/
# torchft's target/ etc. so only source travels.
WORKDIR /build
COPY . /build

# install-torchtt-ft (Makefile): torchtitan deps, then torchtitan + torchft from the local
# submodule source. `uv pip install .` adds symphony-learn for the models.* registry
# (--no-deps: torchtitan/torchft are already installed from source here, not from the
# pyproject git refs). This is where torchft compiles from source — paid once.
# `transformers` is torchtitan's undeclared transformers_modeling_backend dep, which
# models/hf_transformers imports (see the Makefile target for the long version).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r torchtitan/requirements.txt \
    && uv pip install ./torchtitan \
    && uv pip install ./torchft \
    && uv pip install --no-deps . \
    && uv pip install transformers

# install-torch LAST + --force-reinstall so the backend-matched torch NIGHTLY wins over
# whatever stable torch the engine install pulled in (cf. Makefile `install-torch`).
# TORCH_VERSION (e.g. 2.14.0.dev20260804) pins the exact nightly for reproducible
# rebuilds — and for derived images that must layer wheels matched to this exact
# torch build on top without reinstalling it. Unset = newest nightly (old behavior).
ARG TORCH_VERSION=
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --pre "torch${TORCH_VERSION:+==${TORCH_VERSION}+${CUDA_TAG}}" \
    --index-url "${PYTORCH_BASE_URL}/${CUDA_TAG}" --force-reinstall

# Park run_train.sh beside the venv so the whole engine travels as one /opt/symphony tree.
RUN cp /build/run_train.sh /opt/symphony/run_train.sh && chmod +x /opt/symphony/run_train.sh

# ---------------------------------------------------------------------------
# runtime — slim: CUDA runtime libs + the baked engine, no toolchain.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/symphony/.venv
# venv first on PATH so python/torchrun/torchft_lighthouse resolve straight from it.
ENV PATH=/opt/symphony/.venv/bin:$PATH

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
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev build-essential \
        openssh-server rsync \
        libgomp1 libssl3 ca-certificates iproute2 \
    && rm -rf /var/lib/apt/lists/*

# The whole engine: venv (torch + torchtitan + torchft + models.*) + run_train.sh.
COPY --from=builder /opt/symphony /opt/symphony
WORKDIR /opt/symphony

# Fail the build early if the baked engine can't even import (catches a broken nightly /
# ABI mismatch before the image ships). `import torch` does not need a GPU.
RUN python -c "import torch, torchtitan, torchft; print('baked torch', torch.__version__)"

# OCI labels: source -> THIS repo (so GHCR shows symphony-learn's README, not symphony's);
# the build script passes the resolved submodule SHAs.
ARG TORCHTITAN_SHA=unknown
ARG TORCHFT_SHA=unknown
ARG CUDA_TAG=cu130
LABEL org.opencontainers.image.title="symphony-engine" \
      org.opencontainers.image.source="https://github.com/PanocularAI/symphony-learn" \
      org.opencontainers.image.description="Baked training engine (torch nightly + torchtitan + torchft + models.*) for SymphonyLearn." \
      ai.panocular.cuda="${CUDA_TAG}" \
      ai.panocular.ref.torchtitan="${TORCHTITAN_SHA}" \
      ai.panocular.ref.torchft="${TORCHFT_SHA}"
