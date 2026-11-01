#!/usr/bin/env bash
#
# Build (and optionally push) the node images — the baked-deps containers that replace
# `make all` on a node (docker/engine.Dockerfile). Two images, two stages, ONE build:
#
#   <image>:<tag>        (--target full)   torch + torchtitan + torchft + panoengine,
#                                          plus vllm, the RL runtime and the serving
#                                          plane. What --engine-image points at.
#   <image>:<tag>-train  (--target train)  the same minus vllm/RL/serving (~6 GB less
#                                          to pull). Pretrain islands only — a serving
#                                          or RL island cannot run on it.
#
# ONE repository, two tags — not two names. --engine-image is a DAEMON-WIDE argument, so
# every node in a deployment gets the same image and the slim one is only reachable by
# standing up a separate pretrain-only daemon. That does not justify a second GHCR
# package (its own visibility setting, its own retention policy) whose name reads like a
# different product.
#
# Unlike a git-URL install, this builds torchtitan + torchft from the SIBLING CLONES, so
# the image == the exact fork SHAs you have checked out. To refresh the engine:
#     git -C ../torchtitan pull   # or check out specific SHAs
#     ./docker/build-engine-image.sh
# Override the locations with TORCHTITAN_DIR / TORCHFT_DIR.
#
# Usage:
#   ./docker/build-engine-image.sh                 # both tags; <cuda>-<date>-<shas>[-train]
#   TARGETS=train ./docker/build-engine-image.sh   # just the training-only tag
#   PUSH=1 ./docker/build-engine-image.sh          # also push (run `docker login` first)
#   CUDA_TAG=rocm7.0 CUDA_VERSION=13.0.3 ./docker/build-engine-image.sh
#   ./docker/build-engine-image.sh --no-cache      # extra args pass through to `docker build`
#
# Override via env: REGISTRY, IMAGE_NAME, TARGETS, CUDA_TAG, CUDA_VERSION,
# PYTHON_VERSION, PROTOC_VERSION, PUSH. Package pins live in pyproject.toml/uv.lock.

set -euo pipefail
export DOCKER_BUILDKIT=1   # required for the `--mount=type=cache` steps in the Dockerfile

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH" >&2; exit 1; }

# ---- config (override via env) --------------------------------------------
REGISTRY="${REGISTRY:-ghcr.io/panocularai}"
IMAGE_NAME="${IMAGE_NAME:-panofabric-engine}"
TARGETS="${TARGETS:-train full}"     # subset to build; `full` carries the unsuffixed tag
CUDA_TAG="${CUDA_TAG:-cu130}"                 # picks the pyproject backend extra (cu130/cpu/rocm*; CUDA 12 unsupported)
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"        # nvidia/cuda base image tag; keep aligned with CUDA_TAG
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROTOC_VERSION="${PROTOC_VERSION:-32.0}"

PUSH="${PUSH:-0}"
EXTRA_BUILD_ARGS=("$@")   # e.g. --no-cache, --progress=plain

# ---- locate the forks (sibling clones; override to build a different checkout) ----
# They are NOT submodules any more, so they are outside this repo's build context and
# have to arrive as named build contexts (see docker/engine.Dockerfile).
TORCHTITAN_DIR="${TORCHTITAN_DIR:-../torchtitan}"
TORCHFT_DIR="${TORCHFT_DIR:-../torchft}"

tree_sha() {  # <dir> -> checked-out commit sha
  local path="$1" sha
  sha="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$sha" ]]; then
    if [[ "$path" == "." ]]; then
      echo "ERROR: '$REPO_ROOT' is not a git checkout; cannot label the image." >&2
    else
      echo "ERROR: no git checkout at '$path'. Run \`make forks\` (clones both as siblings)," >&2
      echo "       or point TORCHTITAN_DIR / TORCHFT_DIR at your checkouts." >&2
    fi
    exit 1
  fi
  echo "$sha"
}

echo ">>> reading source SHAs..."
ENGINE_SHA="$(tree_sha ".")"
TT_SHA="$(tree_sha "$TORCHTITAN_DIR")"
FT_SHA="$(tree_sha "$TORCHFT_DIR")"
printf '    engine      %s  (%s)\n' "$ENGINE_SHA" "$REPO_ROOT"
printf '    torchtitan  %s  (%s)\n' "$TT_SHA" "$TORCHTITAN_DIR"
printf '    torchft     %s  (%s)\n' "$FT_SHA" "$TORCHFT_DIR"

# THREE trees go into this image and all three are copied from the WORKING DIRECTORY,
# not from git — so uncommitted work is baked in whether or not a SHA describes it.
# `git status --porcelain` (not `diff HEAD`) is the right test: untracked files are
# copied by the build too, and .gitignore'd ones do not show up here.
DIRTY_TREES=()
for d in "." "$TORCHTITAN_DIR" "$TORCHFT_DIR"; do
  if [[ -n "$(git -C "$d" status --porcelain 2>/dev/null)" ]]; then
    DIRTY_TREES+=("$d")
    echo "WARNING: $d has uncommitted changes; they WILL be baked in, but no SHA" >&2
    echo "         below describes them." >&2
  fi
done

# ---- tags -----------------------------------------------------------------
DATE_TAG="$(date -u +%Y%m%d)"

# Records what is ACTUALLY inside: this repo's SHA plus both forks'. It used to be
# torchft's SHA alone, which meant an engine-only or torchtitan-only change produced a
# byte-identical tag to the previous build — and torchft is the tree that moves least,
# so that was the common case, not the corner case. Two images claiming one tag, with
# the second push silently overwriting the first.
VERSION_TAG="${CUDA_TAG}-${DATE_TAG}-${ENGINE_SHA:0:7}-tt${TT_SHA:0:7}-ft${FT_SHA:0:7}"

# "Immutable" only holds when all three trees are COMMITTED; otherwise the SHAs above
# describe something other than what was copied in. Mark those builds so they can never
# clobber a real immutable tag — same rule as infra/engine-image/build.sh in panofabric.
if (( ${#DIRTY_TREES[@]} )); then
  VERSION_TAG="${VERSION_TAG}-dirty-$(date +%s)"
  echo "NOTE: uncommitted changes in ${DIRTY_TREES[*]} -> tagging :$VERSION_TAG"
fi

# ---- build -----------------------------------------------------------------
# `full` derives FROM `train`, so building both is one pass plus a cache-hit re-tag.
# `full` gets the bare tags; `train` gets the same with a -train suffix. Every tag is
# collected for the push, because pushing "${IMAGE}:${CUDA_TAG}" once per target would
# have the train build clobber full's moving tag.
IMAGE="${REGISTRY}/${IMAGE_NAME}"
PUSH_REFS=()

for target in $TARGETS; do
  case "$target" in
    full)  sfx="" ;;
    train) sfx="-train" ;;
    *) echo "ERROR: unknown target '$target' (want: train, full)" >&2; exit 1 ;;
  esac
  version_ref="${IMAGE}:${VERSION_TAG}${sfx}"
  moving_ref="${IMAGE}:${CUDA_TAG}${sfx}"

  echo ">>> building ${version_ref}  (--target $target)"
  docker build \
    -f docker/engine.Dockerfile \
    --target "$target" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg PROTOC_VERSION="$PROTOC_VERSION" \
    --build-arg CUDA_TAG="$CUDA_TAG" \
    --build-arg ENGINE_SHA="$ENGINE_SHA" \
    --build-arg TORCHTITAN_SHA="$TT_SHA" \
    --build-arg TORCHFT_SHA="$FT_SHA" \
    --build-context torchtitan="$TORCHTITAN_DIR" \
    --build-context torchft="$TORCHFT_DIR" \
    -t "$version_ref" \
    -t "$moving_ref" \
    "${EXTRA_BUILD_ARGS[@]}" \
    .
  PUSH_REFS+=("$version_ref" "$moving_ref")
done

echo ">>> built:"
for ref in "${PUSH_REFS[@]}"; do echo "      $ref"; done

if [[ "$PUSH" == "1" ]]; then
  echo ">>> pushing..."
  for ref in "${PUSH_REFS[@]}"; do docker push "$ref"; done
  echo ">>> pushed."
else
  echo ">>> PUSH=0 — not pushing. To push: docker login ${REGISTRY%%/*} && PUSH=1 $0"
fi

# Machine-readable handoff: panofabric's 04-deploy-controld.sh reads the tag it just
# built from here instead of scraping the human output above. `full` is the node image,
# so it wins when both were built.
if [[ -n "${TAG_OUT:-}" ]]; then
  if [[ " $TARGETS " == *" full "* ]]; then
    printf '%s\n' "${IMAGE}:${VERSION_TAG}" > "$TAG_OUT"
  else
    printf '%s\n' "${IMAGE}:${VERSION_TAG}-train" > "$TAG_OUT"
  fi
fi

cat <<EOF
>>> point --engine-image / ENGINE_IMAGE at the FULL tag:
      ${IMAGE}:${VERSION_TAG}
EOF
