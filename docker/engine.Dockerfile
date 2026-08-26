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
ARG PYTORCH_BASE_URL=https://download.pytorch.org/whl/nightly

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:/root/.cargo/bin:$PATH
ENV VIRTUAL_ENV=/opt/panoengine/.venv

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

# install-torchtt-ft (Makefile): torchtitan deps, then torchtitan + torchft from the
# sibling clones, bind-mounted from their named build contexts. `rw` because setuptools
# and maturin write build metadata into the source dir (it lands in a scratch overlay,
# never in the checkout). `uv pip install --no-deps .` then adds panofabric-engine for the
# recipe registry (--no-deps: the forks are already installed from source here, not from
# the pyproject git pins). This is where torchft compiles from source — paid once.
# `transformers` is torchtitan's undeclared transformers_modeling_backend dep, which
# panoengine.train.pretrain.hf_transformers imports (see the Makefile for the long version).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=torchtitan,target=/src/torchtitan,rw \
    --mount=type=bind,from=torchft,target=/src/torchft,rw \
    uv pip install -r /src/torchtitan/requirements.txt \
    && uv pip install /src/torchtitan \
    && uv pip install /src/torchft \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev build-essential \
        openssh-server rsync \
        libgomp1 libssl3 ca-certificates iproute2 \
    && rm -rf /var/lib/apt/lists/*

# The whole engine: venv (torch + torchtitan + torchft + panoengine) + run_train.sh.
COPY --from=builder /opt/panoengine /opt/panoengine
WORKDIR /opt/panoengine

# Fail the build early if the baked engine can't even import (catches a broken nightly /
# ABI mismatch before the image ships). `import torch` does not need a GPU.
RUN python -c "import torch, torchtitan, torchft; print('baked torch', torch.__version__)"

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
ARG PYTORCH_BASE_URL=https://download.pytorch.org/whl/nightly

# The train stage ships no installer (uv lived only in the builder).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# vLLM FIRST (the slow, rarely-changing, ~3 GB layer). All three EXACT-pinned to one
# nightly date so the vllm wheel matches the torch it was compiled against (torchtitan
# rl/README §4: build numbers must match; torchvision only because vllm's kernel warmup
# imports it). Do NOT unpin: an unpinned resolve here backtracked past every nightly to
# the vllm 0.2.5 SDIST from PyPI and tried to compile it (no nvcc in this runtime stage).
# unsafe-best-match is still needed so uv consults the nightly index for the pins while
# vllm's ordinary deps resolve from PyPI.
#
# NO --force-reinstall, and TORCH_VERSION must equal the train stage's own pin: torch is
# then already satisfied and this layer holds ONLY vllm+torchvision+deps. If they drift
# apart the pin still forces the matched torch — correct, but the layer silently
# re-fattens by ~5 GB of duplicated torch stack.
#
# The dates are NOT all equal: torchvision's nightly of day D pins the torch of day D-1,
# so it rides one day AHEAD of the torch/vllm pair. Defaults come from the Makefile's
# RL_* vars via the build script — bump them THERE, not here.
ARG TORCH_VERSION=
ARG TORCHVISION_VERSION
ARG VLLM_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    if [ -n "${TORCH_VERSION}" ]; then torch_spec="torch==${TORCH_VERSION}+${CUDA_TAG}"; \
    else torch_spec="torch"; fi; \
    uv pip install --python ${VIRTUAL_ENV}/bin/python --pre \
      "$torch_spec" \
      "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
      "vllm==${VLLM_VERSION}+${CUDA_TAG}" \
      --extra-index-url "${PYTORCH_BASE_URL}/${CUDA_TAG}" \
      --index-strategy unsafe-best-match

# The RL runtime. All four are hand-pinned because none is resolvable the obvious way:
#   - torchstore's PyPI release pins torch==2.9.0 + torchmonarch==0.1.2 and would drag the
#     whole stack backwards, so it is installed --no-deps from a pinned SOURCE TARBALL
#     (a git+https URL needs a `git` binary, which this runtime stage does not ship) with
#     its two real runtime deps added by hand.
#   - torchmonarch ships compiled bindings that load libtorch, so its build date must
#     track TORCH_VERSION's.
#   - flash-attn-3 comes from the pytorch TEST index, not the nightly one (upstream
#     torchtitan rl/README §2). The RL trainer imports `flash_attn_interface`
#     UNCONDITIONALLY on Hopper+ and dies without it; pre-Hopper falls back to FA2.
#   - math-verify is the dapo_math example's rubric dep. It SHOULD ride
#     model.requirements, but torchtitan's setuptools config has no package-data entry so
#     the file is absent from the installed wheel; baking it keeps RL runs unblocked.
# Bump these together with TORCH_VERSION in the Makefile, and re-run the alphabet_sort
# smoke test.
ARG TORCHMONARCH_VERSION
ARG TORCHSTORE_SHA
ARG RENDERERS_VERSION
ARG FLASH_ATTN_3_VERSION
ARG MATH_VERIFY_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VIRTUAL_ENV}/bin/python \
      "torchmonarch==${TORCHMONARCH_VERSION}" \
      "renderers==${RENDERERS_VERSION}" \
      "math-verify==${MATH_VERIFY_VERSION}" \
      pygtrie portpicker \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python --no-deps \
      "torchstore @ https://github.com/meta-pytorch/torchstore/archive/${TORCHSTORE_SHA}.tar.gz" \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python \
      "flash-attn-3==${FLASH_ATTN_3_VERSION}" \
      --extra-index-url "${PYTORCH_BASE_URL%/nightly}/test/${CUDA_TAG}"

# The serving plane's declared deps (aiohttp/safetensors/hf-hub/numpy/transformers).
# They come free with vllm where versions matter, but the extra makes that explicit
# rather than accidental. torch/vllm stay undeclared by [serve] on purpose.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.,target=/src,rw \
    uv pip install --python ${VIRTUAL_ENV}/bin/python "/src[serve]"

# Fail the build if anything can't import (no GPU needed): catches a vllm/torch mismatch
# and a torchft break from the torch bump before the image ships. The RL line is separate
# because it is the one that catches a torchmonarch built against a DIFFERENT torch — its
# rust bindings only resolve libtorch symbols at import, so a mismatched pin sails through
# installation and dies on the node.
RUN python -c "import torch, torchtitan, torchft, vllm, panoengine.serve; print('torch', torch.__version__)" \
 && python -c "import torchstore, renderers, flash_attn_interface, math_verify; \
from monarch.actor import ProcMesh; from monarch.spmd import setup_torch_elastic_env_async; \
import panoengine.train.rl.train; print('rl stack ok')"

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
