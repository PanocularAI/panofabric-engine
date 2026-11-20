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
#   ./docker/build-engine-image.sh                 # both tags; <cuda>-<shas>[-dirty]-<image id>[-train]
#   TARGETS=train ./docker/build-engine-image.sh   # just the training-only tag
#   PUSH=1 ./docker/build-engine-image.sh          # also push (run `docker login` first)
#   CUDA_TAG=rocm7.0 CUDA_VERSION=13.0.3 ./docker/build-engine-image.sh
#   ./docker/build-engine-image.sh --no-cache      # extra args pass through to `docker build`
#
# Override via env: REGISTRY, IMAGE_NAME, TARGETS, CUDA_TAG, CUDA_VERSION,
# PYTHON_VERSION, PROTOC_VERSION, PYTORCH_BASE_URL, TORCH_VERSION, PUSH.

set -euo pipefail
export DOCKER_BUILDKIT=1   # required for the `--mount=type=cache` steps in the Dockerfile

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH" >&2; exit 1; }

# ---- config (override via env) --------------------------------------------
REGISTRY="${REGISTRY:-ghcr.io/panocularai}"
IMAGE_NAME="${IMAGE_NAME:-panofabric-engine}"
TARGETS="${TARGETS:-train full}"     # subset to build; `full` carries the unsuffixed tag
CUDA_TAG="${CUDA_TAG:-cu130}"                 # torch nightly index (cu130/cpu/rocm7.0; CUDA 12 unsupported)
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"        # nvidia/cuda base image tag; keep aligned with CUDA_TAG
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROTOC_VERSION="${PROTOC_VERSION:-32.0}"
PYTORCH_BASE_URL="${PYTORCH_BASE_URL:-https://download.pytorch.org/whl/nightly}"
TORCH_VERSION="${TORCH_VERSION:-}"            # exact nightly pin; default: the Makefile's TORCH_VERSION

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
# <cuda>-<engine sha>-tt<sha>-ft<sha>[-dirty]-<image id>[-train]. The SHAs say where the
# sources came from; the image ID (completed after each build below) says what is inside.
# A fully cached rebuild keeps its image ID, so an unchanged image keeps its tag: the
# registry already has it (the push is skipped) and the daemon's --engine-image does not
# change. Any change to the contents, committed or not, gives a new ID and so a new tag
# that cannot overwrite an older one. (The tag used to carry the build date and, for a
# dirty tree, a timestamp, so every build got a new tag even when nothing had changed.)
SOURCE_TAG="${CUDA_TAG}-${ENGINE_SHA:0:7}-tt${TT_SHA:0:7}-ft${FT_SHA:0:7}"
if (( ${#DIRTY_TREES[@]} )); then
  SOURCE_TAG="${SOURCE_TAG}-dirty"
fi
IID_FILE="$(mktemp)"
trap 'rm -f "$IID_FILE"' EXIT

# ---- RL / serving pins: read from the Makefile, which is the single source of truth --
# `make install-rl` (the imageless path) and this image MUST agree, and they used to be
# two hand-synced copies in two repos. Parsed rather than duplicated so a bump lands in
# one place.
mk_var() {  # <NAME> -> the Makefile's `NAME ?= value`
  local v
  v="$(sed -nE "s/^$1[[:space:]]*\\?=[[:space:]]*(.*)\$/\\1/p" Makefile | head -1)"
  [[ -n "$v" ]] || { echo "ERROR: $1 not found in Makefile" >&2; exit 1; }
  echo "$v"
}
# The torch nightly must be the one the vllm/torchvision wheels were built against.
TORCH_VERSION="${TORCH_VERSION:-$(mk_var TORCH_VERSION)}"
RL_ARGS=(
  --build-arg TORCHVISION_VERSION="$(mk_var RL_TORCHVISION_VERSION)"
  --build-arg VLLM_VERSION="$(mk_var RL_VLLM_VERSION)"
  --build-arg TORCHMONARCH_VERSION="$(mk_var RL_TORCHMONARCH_VERSION)"
  --build-arg TORCHSTORE_SHA="$(mk_var RL_TORCHSTORE_SHA)"
  --build-arg RENDERERS_VERSION="$(mk_var RL_RENDERERS_VERSION)"
  --build-arg FLASH_ATTN_3_VERSION="$(mk_var RL_FLASH_ATTN_3_VERSION)"
  --build-arg MATH_VERIFY_VERSION="$(mk_var RL_MATH_VERIFY_VERSION)"
  --build-arg VERIFIERS_VERSION="$(mk_var RL_VERIFIERS_VERSION)"
)

# ---- build -----------------------------------------------------------------
# `train` and `full` share the `runtime` stage, so building both pays for it once.
# `full` gets the bare tags; `train` gets the same with a -train suffix, so the train
# build cannot clobber full's moving tag.
IMAGE="${REGISTRY}/${IMAGE_NAME}"
VERSION_REFS=()
MOVING_REFS=()
NODE_REF=""   # what TAG_OUT hands to the deploy: `full` when built, else `train`

for target in $TARGETS; do
  case "$target" in
    full)  sfx="" ;;
    train) sfx="-train" ;;
    *) echo "ERROR: unknown target '$target' (want: train, full)" >&2; exit 1 ;;
  esac
  moving_ref="${IMAGE}:${CUDA_TAG}${sfx}"

  echo ">>> building ${IMAGE}:${SOURCE_TAG}-<image id>${sfx}  (--target $target)"
  docker build \
    -f docker/engine.Dockerfile \
    --target "$target" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg PROTOC_VERSION="$PROTOC_VERSION" \
    --build-arg CUDA_TAG="$CUDA_TAG" \
    --build-arg PYTORCH_BASE_URL="$PYTORCH_BASE_URL" \
    --build-arg TORCH_VERSION="$TORCH_VERSION" \
    --build-arg ENGINE_SHA="$ENGINE_SHA" \
    --build-arg TORCHTITAN_SHA="$TT_SHA" \
    --build-arg TORCHFT_SHA="$FT_SHA" \
    "${RL_ARGS[@]}" \
    --build-context torchtitan="$TORCHTITAN_DIR" \
    --build-context torchft="$TORCHFT_DIR" \
    --iidfile "$IID_FILE" \
    -t "$moving_ref" \
    "${EXTRA_BUILD_ARGS[@]}" \
    .
  image_id="$(<"$IID_FILE")"
  image_id="${image_id#sha256:}"
  version_ref="${IMAGE}:${SOURCE_TAG}-${image_id:0:12}${sfx}"
  docker tag "$moving_ref" "$version_ref"
  VERSION_REFS+=("$version_ref")
  MOVING_REFS+=("$moving_ref")
  if [[ "$target" == "full" || -z "$NODE_REF" ]]; then NODE_REF="$version_ref"; fi
done

echo ">>> built:"
for ref in "${VERSION_REFS[@]}" "${MOVING_REFS[@]}"; do echo "      $ref"; done

if [[ "$PUSH" == "1" ]]; then
  for i in "${!VERSION_REFS[@]}"; do
    # The version tag names the contents, so if the registry has it, it has this image.
    # A failed or hung lookup (the timeout covers a credential helper waiting on a login)
    # reads as "not there": the push then runs, which is safe, just slower.
    if timeout 60 docker manifest inspect "${VERSION_REFS[$i]}" >/dev/null 2>&1; then
      echo ">>> ${VERSION_REFS[$i]} is already in the registry — not pushing it again."
      continue
    fi
    echo ">>> pushing ${VERSION_REFS[$i]}"
    docker push "${VERSION_REFS[$i]}"
    docker push "${MOVING_REFS[$i]}"
  done
  echo ">>> registry up to date."
else
  echo ">>> PUSH=0 — not pushing. To push: docker login ${REGISTRY%%/*} && PUSH=1 $0"
fi

# Machine-readable handoff: panofabric's 04-deploy-controld.sh reads the tag it just
# built from here instead of scraping the human output above.
if [[ -n "${TAG_OUT:-}" ]]; then
  printf '%s\n' "$NODE_REF" > "$TAG_OUT"
fi

cat <<EOF
>>> point --engine-image / ENGINE_IMAGE at the FULL tag:
      ${NODE_REF}
EOF
