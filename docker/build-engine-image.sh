#!/usr/bin/env bash
#
# Build (and optionally push) the SymphonyLearn engine image — the baked-deps container
# that replaces `make all` on training nodes (docker/engine.Dockerfile). The symphony
# control plane consumes it via `--engine-image`.
#
# Unlike a git-URL install, this builds torchtitan + torchft from the SIBLING CLONES, so
# the image == the exact fork SHAs you have checked out. To refresh the engine:
#     git -C ../torchtitan pull   # or check out specific SHAs
#     ./docker/build-engine-image.sh
# Override the locations with TORCHTITAN_DIR / TORCHFT_DIR.
#
# Usage:
#   ./docker/build-engine-image.sh                 # build locally; tag <cuda>-<date>-<sha>
#   PUSH=1 ./docker/build-engine-image.sh          # also push (run `docker login` first)
#   CUDA_TAG=rocm7.0 CUDA_VERSION=13.0.3 ./docker/build-engine-image.sh
#   ./docker/build-engine-image.sh --no-cache      # extra args pass through to `docker build`
#
# Override via env: REGISTRY, IMAGE_NAME, CUDA_TAG, CUDA_VERSION, PYTHON_VERSION,
# PROTOC_VERSION, PYTORCH_BASE_URL, PUSH.

set -euo pipefail
export DOCKER_BUILDKIT=1   # required for the `--mount=type=cache` steps in the Dockerfile

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH" >&2; exit 1; }

# ---- config (override via env) --------------------------------------------
REGISTRY="${REGISTRY:-ghcr.io/panocularai}"
IMAGE_NAME="${IMAGE_NAME:-symphony-engine}"
CUDA_TAG="${CUDA_TAG:-cu130}"                 # torch nightly index (cu130/cpu/rocm7.0; CUDA 12 unsupported)
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"        # nvidia/cuda base image tag; keep aligned with CUDA_TAG
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROTOC_VERSION="${PROTOC_VERSION:-32.0}"
PYTORCH_BASE_URL="${PYTORCH_BASE_URL:-https://download.pytorch.org/whl/nightly}"
TORCH_VERSION="${TORCH_VERSION:-}"            # exact nightly pin (e.g. 2.14.0.dev20260804); empty = newest

PUSH="${PUSH:-0}"
EXTRA_BUILD_ARGS=("$@")   # e.g. --no-cache, --progress=plain

# ---- locate the forks (sibling clones; override to build a different checkout) ----
# They are NOT submodules any more, so they are outside this repo's build context and
# have to arrive as named build contexts (see docker/engine.Dockerfile).
TORCHTITAN_DIR="${TORCHTITAN_DIR:-../torchtitan}"
TORCHFT_DIR="${TORCHFT_DIR:-../torchft}"

fork_sha() {  # <dir> -> checked-out commit sha
  local path="$1" sha
  sha="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$sha" ]]; then
    echo "ERROR: no git checkout at '$path'. Run \`make forks\` (clones both as siblings)," >&2
    echo "       or point TORCHTITAN_DIR / TORCHFT_DIR at your checkouts." >&2
    exit 1
  fi
  echo "$sha"
}

echo ">>> reading engine fork SHAs..."
TT_SHA="$(fork_sha "$TORCHTITAN_DIR")"
FT_SHA="$(fork_sha "$TORCHFT_DIR")"
printf '    torchtitan  %s  (%s)\n' "$TT_SHA" "$TORCHTITAN_DIR"
printf '    torchft     %s  (%s)\n' "$FT_SHA" "$TORCHFT_DIR"

# A dirty fork silently produces an image whose SHA label is a lie.
for d in "$TORCHTITAN_DIR" "$TORCHFT_DIR"; do
  if ! git -C "$d" diff --quiet HEAD 2>/dev/null; then
    echo "WARNING: $d has uncommitted changes; they WILL be baked in, but the image's" >&2
    echo "         SHA label will not describe them." >&2
  fi
done

# ---- tags -----------------------------------------------------------------
DATE_TAG="$(date -u +%Y%m%d)"
VERSION_TAG="${CUDA_TAG}-${DATE_TAG}-${FT_SHA:0:7}"   # immutable, records what's inside
IMAGE="${REGISTRY}/${IMAGE_NAME}"

echo ">>> building ${IMAGE}:${VERSION_TAG}"
docker build \
  -f docker/engine.Dockerfile \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
  --build-arg PROTOC_VERSION="$PROTOC_VERSION" \
  --build-arg CUDA_TAG="$CUDA_TAG" \
  --build-arg PYTORCH_BASE_URL="$PYTORCH_BASE_URL" \
  --build-arg TORCH_VERSION="$TORCH_VERSION" \
  --build-arg TORCHTITAN_SHA="$TT_SHA" \
  --build-arg TORCHFT_SHA="$FT_SHA" \
  --build-context torchtitan="$TORCHTITAN_DIR" \
  --build-context torchft="$TORCHFT_DIR" \
  -t "${IMAGE}:${VERSION_TAG}" \
  -t "${IMAGE}:${CUDA_TAG}" \
  "${EXTRA_BUILD_ARGS[@]}" \
  .

echo ">>> built:"
echo "      ${IMAGE}:${VERSION_TAG}   (immutable)"
echo "      ${IMAGE}:${CUDA_TAG}      (moving 'latest for this CUDA' tag)"

if [[ "$PUSH" == "1" ]]; then
  echo ">>> pushing..."
  docker push "${IMAGE}:${VERSION_TAG}"
  docker push "${IMAGE}:${CUDA_TAG}"
  echo ">>> pushed."
else
  echo ">>> PUSH=0 — not pushing. To push: docker login ${REGISTRY%%/*} && PUSH=1 $0"
fi

cat <<EOF
>>> Now you can use the engine-image with ${IMAGE}:${VERSION_TAG}
EOF
