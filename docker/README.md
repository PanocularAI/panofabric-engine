# `docker/` — the SymphonyLearn engine image

This bakes the **training engine** into a container so a fresh training node no longer
runs `make all` (apt + rustup + **compile torchft** + download the ~2.5 GB torch nightly)
on every run. Cold provisioning drops from **~10 min to ~1**.

What's baked (mirrors `make all`, once at build time):

- the torch **nightly** for a chosen CUDA build (`cu130` by default),
- **torchtitan** + **torchft**, built from *this repo's submodules* (so the image matches
  the exact SHAs you have checked out; the torchft Rust extension is compiled here, once),
- **symphony-learn** itself, for the `models.*` config registries,

inside a venv at `/opt/symphony/.venv`, on PATH. A container ships its own CUDA userspace,
so the image only needs the host NVIDIA **driver** to be recent enough — it does not have
to match the host CUDA toolkit.


## Build it

```bash
./docker/build-engine-image.sh                 # build locally
docker login ghcr.io && PUSH=1 ./docker/build-engine-image.sh   # build + push
```

The build installs torchtitan/torchft from the **checked-out submodules**, so to refresh
the engine you bump the submodules and rebuild:

```bash
git submodule update --remote torchtitan torchft   # or check out specific SHAs
PUSH=1 ./docker/build-engine-image.sh
```

### Tags

Each build produces two tags:

- `…:cu130-20260630-ab12cd3` — immutable; CUDA build + date + torchft short SHA.
- `…:cu130` — moving "latest for this CUDA" pointer.

## Common overrides (all via env)

| Var | Default | Notes |
|---|---|---|
| `REGISTRY` | `ghcr.io/panocularai` | where to push |
| `IMAGE_NAME` | `symphony-engine` | |
| `CUDA_TAG` | `cu130` | torch nightly index: `cu130`/`cpu`/`rocm7.0`. CUDA 12 (`cu128`/`cu129`) is unsupported — torch>=2.13 is CUDA-13-only and the engine needs it |
| `CUDA_VERSION` | `13.0.3` | `nvidia/cuda` base tag — keep aligned with `CUDA_TAG` |
| `PYTHON_VERSION` | `3.12` | matches the ubuntu 24.04 base |
| `PUSH` | `0` | `1` to push after building |

Extra args pass straight through to `docker build`, e.g. `--no-cache`.

## Requirements

- Docker with BuildKit (the script exports `DOCKER_BUILDKIT=1`).
- Submodules initialized: `git submodule update --init torchtitan torchft`.
- An x86_64 build host.
