#!/usr/bin/env bash
#
# Build (and optionally push) the SymphonyLearn engine image — the baked-deps container
# that replaces `make all` on training nodes (docker/engine.Dockerfile). The symphony
# control plane consumes it via `--engine-image`.
#
# Unlike a git-URL install, this builds torchtitan + torchft from THIS checkout's
# submodules, so the image == the exact submodule SHAs you have checked out. To refresh
# the engine:
#     git submodule update --remote torchtitan torchft   # or check out specific SHAs
#     ./docker/build-engine-image.sh
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

# ---- read submodule SHAs (provenance + tag) -------------------------------
submodule_sha() {  # <path> -> checked-out commit sha (errors if the submodule is empty)
  local path="$1" sha
  sha="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$sha" ]]; then
    echo "ERROR: submodule '$path' is not initialized. Run: git submodule update --init $path" >&2
    exit 1
  fi
  echo "$sha"
}

echo ">>> reading engine submodule SHAs..."
TT_SHA="$(submodule_sha torchtitan)"
FT_SHA="$(submodule_sha torchft)"
printf '    torchtitan  %s\n' "$TT_SHA"
printf '    torchft     %s\n' "$FT_SHA"

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
