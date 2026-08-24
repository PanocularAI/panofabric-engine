# `docker/` — the SymphonyLearn engine image

This bakes the **training engine** into a container so a fresh training node no longer
runs `make all` (apt + rustup + **compile torchft** + download the ~2.5 GB torch nightly)
on every run. Cold provisioning drops from **~10 min to ~1**.

What's baked (mirrors `make all`, once at build time):

- the torch **nightly** for a chosen CUDA build (`cu130` by default),
- **torchtitan** + **torchft**, built from the *sibling fork clones* (so the image matches
  the exact SHAs you have checked out; the torchft Rust extension is compiled here, once),
- **panofabric-engine** itself, for the `panoengine.train.*` recipe registries
  (still reachable at their legacy `models.*` paths),

inside a venv at `/opt/symphony/.venv`, on PATH. A container ships its own CUDA userspace,
so the image only needs the host NVIDIA **driver** to be recent enough — it does not have
to match the host CUDA toolkit.


## Build it

```bash
./docker/build-engine-image.sh                 # build locally
docker login ghcr.io && PUSH=1 ./docker/build-engine-image.sh   # build + push
```

The build installs torchtitan/torchft from the **sibling clones** (`../torchtitan`,
`../torchft` by default — `make forks` creates them), passed in as named build contexts. To
refresh the engine, move those checkouts and rebuild:

```bash
git -C ../torchtitan pull                       # or check out specific SHAs
PUSH=1 ./docker/build-engine-image.sh

# Or build a different checkout without touching the siblings:
TORCHTITAN_DIR=/path/to/torchtitan PUSH=1 ./docker/build-engine-image.sh
```

The script warns if a fork has uncommitted changes: they are baked in, but the image's SHA
label cannot describe them.

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
- The forks checked out as siblings: `make forks` (or set TORCHTITAN_DIR / TORCHFT_DIR).
- An x86_64 build host.
