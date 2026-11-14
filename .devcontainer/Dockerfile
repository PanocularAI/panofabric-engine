# Thin shell around the Nix flake: this image only
# provides what VS Code's server, git and ssh need from a normal FHS distro.
# uv, protoc, rustup, gcc, the CUDA 13.0 toolkit and every other build
# dependency come from flake.nix's devShells, pinned by flake.lock, exactly as
# on the NixOS dev cluster; Python deps then come from scripts/nix_dev_setup.sh,
# which the devShell's shellHook runs. Nix itself is installed by the
# devcontainer `nix` feature (see devcontainer.json).
#
# Pinned by digest so the base doesn't move under us; bump deliberately with
#   docker buildx imagetools inspect mcr.microsoft.com/devcontainers/base:ubuntu-24.04
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04@sha256:d94c97dd9cacf183d0a6fd12a8e87b526e9e928307674ae9c94139139c0c6eae

# For hosts that pass the GPU through the legacy `nvidia` runtime instead of
# CDI: without `compute`, it only injects nvidia-smi and NVML, not libcuda.
# (nvidia/cuda images set this; we don't use one because the CUDA toolkit
# itself comes from Nix, and only the driver from the host.)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# direnv drives the flake's devShell via the repo's .envrc (nix-direnv itself
# is fetched and hash-pinned by .envrc, since this isn't home-manager).
RUN apt-get update \
 && apt-get install -y --no-install-recommends direnv \
 && rm -rf /var/lib/apt/lists/* \
 && echo 'eval "$(direnv hook bash)"' >> /home/vscode/.bashrc \
 && echo 'eval "$(direnv hook zsh)"' >> /home/vscode/.zshrc

# Mount point for the named volume in devcontainer.json, created here so a
# fresh volume is populated owned by vscode rather than root. It holds
# state/panofabric-engine/... (the per-shell venvs flake.nix puts there),
# share/uv (the uv-managed CPython those venvs link to) and cache/uv (uv's
# wheel cache, via UV_CACHE_DIR).
RUN install -d -o vscode -g vscode /home/vscode/.local

COPY link-nvidia-driver.sh /usr/local/bin/link-nvidia-driver
