#!/bin/sh
# Sets up the training env WITHOUT going through the Makefile: `uv sync`, backend-matched
# torch, and the sibling torchtitan/torchft forks -- the same outcome as `make all` (plus
# `make dev-forks`), driven by flake.nix's devShell/direnv instead. See flake.nix for why:
# the Makefile's own `setup-env` assumes things (an apt-managed toolchain, a writable
# ~/.bashrc) that don't hold under NixOS/home-manager, and the toolchain itself (protoc,
# rust, uv, gcc) is supplied by the flake's devShell instead of setup-env's curl|sh/apt
# paths. Everything FROM sync-project ON, this script reimplements directly.
#
# Run with no args from inside the devShell (flake.nix's shellHook does this on every
# `nix develop`/direnv entry): it hashes the same install-determining inputs the Makefile's
# own `ensure` cache does and is a fast no-op if nothing changed. Env vars:
#   PF_RL=1                 also install the RL runtime (vllm, torchmonarch, torchstore, ...)
#   PF_DEV_FORKS=0           install the forks normally instead of editable (default: editable,
#                            since this whole path exists for engine hackers)
#   PF_NIX_SETUP_FORCE=1     skip the fingerprint check and reinstall everything
#   FORKS_DIR                where the fork siblings live (default: .., matching the Makefile)
#
# POSIX sh, like scripts/pf_backend.sh and scripts/pf_env_fp.sh, which this script shells
# out to rather than reimplementing.
set -eu

cd "$(dirname "$0")/.."
repo=$(pwd)
# flake.nix points UV_PROJECT_ENVIRONMENT at machine-local state under ~/.local/state;
# `uv sync` below honors the same var, so both agree on where the venv lives.
venv="${UV_PROJECT_ENVIRONMENT:-$repo/.venv}"
forks_dir="${FORKS_DIR:-..}"
torchtitan_dir="$forks_dir/torchtitan"
torchft_dir="$forks_dir/torchft"

log() { printf '[nix-dev-setup] %s\n' "$*"; }
die() {
	printf '[nix-dev-setup] ERROR: %s\n' "$*" >&2
	exit 1
}

# --- pins -------------------------------------------------------------------
# Fork URL+SHA: parsed from pyproject.toml's [train]/[decentralized] extras (the pins the
# Makefile's own TORCHTITAN_REF/TORCHFT_REF comment says to keep in sync with), not
# hand-copied, so this script can never drift from them on its own.
fork_pin() { # $1 = package name (torchtitan|torchft)
	sed -n "s/.*\"$1 @ \(git+[^\"]*\)\".*/\1/p" "$repo/pyproject.toml" | head -1
}
torchtitan_pin=$(fork_pin torchtitan)
torchft_pin=$(fork_pin torchft)
[ -n "$torchtitan_pin" ] && [ -n "$torchft_pin" ] ||
	die "couldn't find the torchtitan/torchft git+ pins in pyproject.toml"
torchtitan_url=${torchtitan_pin#git+}
torchtitan_url=${torchtitan_url%@*}
torchtitan_ref=${torchtitan_pin##*@}
torchft_url=${torchft_pin#git+}
torchft_url=${torchft_url%@*}
torchft_ref=${torchft_pin##*@}

# RL runtime pins: these exist ONLY in the Makefile (pyproject.toml deliberately leaves
# torch/vllm undeclared there -- see its "serve" extra comment), so there is nowhere else
# to read them from. Parsed out of the Makefile's own `RL_* ?= ...`/`TORCH_VERSION ?= ...`
# lines as plain text -- `make` itself is never invoked.
mk_var() { # $1 = variable name
	sed -n "s/^$1[[:space:]]*?=[[:space:]]*//p" "$repo/Makefile" | head -1
}
TORCH_VERSION=$(mk_var TORCH_VERSION)
RL_TORCHMONARCH_VERSION=$(mk_var RL_TORCHMONARCH_VERSION)
RL_TORCHSTORE_SHA=$(mk_var RL_TORCHSTORE_SHA)
RL_RENDERERS_VERSION=$(mk_var RL_RENDERERS_VERSION)
RL_FLASH_ATTN_3_VERSION=$(mk_var RL_FLASH_ATTN_3_VERSION)
RL_VLLM_VERSION=$(mk_var RL_VLLM_VERSION)
RL_TORCHVISION_VERSION=$(mk_var RL_TORCHVISION_VERSION)
RL_MATH_VERIFY_VERSION=$(mk_var RL_MATH_VERIFY_VERSION)
RL_VERIFIERS_VERSION=$(mk_var RL_VERIFIERS_VERSION)

pf_rl="${PF_RL:-0}"
pf_dev_forks="${PF_DEV_FORKS:-1}"

if [ "$pf_rl" = "1" ]; then
	for v in TORCH_VERSION RL_TORCHMONARCH_VERSION RL_TORCHSTORE_SHA RL_RENDERERS_VERSION \
		RL_FLASH_ATTN_3_VERSION RL_VLLM_VERSION RL_TORCHVISION_VERSION \
		RL_MATH_VERIFY_VERSION RL_VERIFIERS_VERSION; do
		eval "val=\${$v}"
		[ -n "$val" ] || die "PF_RL=1 but couldn't parse $v out of the Makefile"
	done
fi

# --- skip if nothing install-determining changed ----------------------------
backend=$(sh "$repo/scripts/pf_backend.sh")
src_fp=$(sh "$repo/scripts/pf_env_fp.sh" --source)
fp=$(printf '%s:%s:rl%s:devforks%s' "$src_fp" "$backend" "$pf_rl" "$pf_dev_forks" | sha256sum | cut -c1-16)
stamp="$venv/.nix-dev-setup-stamp"
verify='import torch,torchtitan,torchft'
[ "$pf_rl" = "1" ] && verify="$verify,vllm"

if [ "${PF_NIX_SETUP_FORCE:-0}" != "1" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$fp" ] &&
	[ -x "$venv/bin/python" ] && "$venv/bin/python" -c "$verify" 2>/dev/null; then
	log "up to date (fp=$fp, backend=$backend). PF_NIX_SETUP_FORCE=1 to redo it anyway."
	exit 0
fi
log "setting up (fp=$fp, backend=$backend, PF_RL=$pf_rl, PF_DEV_FORKS=$pf_dev_forks)"

# --- rust toolchain -----------------------------------------------------------
# flake.nix supplies `rustup` itself (so this never falls back to the Makefile's
# rustup-init curl|sh install), but an actual rustc/cargo still has to be fetched once.
# `rustup default stable` is a no-op if a toolchain is already active. The fetched
# rustc/cargo are real, dynamically-linked upstream binaries -- they run here only
# because nix-ld is enabled cluster-wide (the flake checked $NIX_LD is set).
rustup default stable >/dev/null

# --- project venv ------------------------------------------------------------
log "uv sync"
uv sync

uv_pip() {
	VIRTUAL_ENV="$venv" uv pip install "$@"
}

# --- torch (LAST relative to the project's own deps, matching install-torch's own
# comment: an unconstrained resolve can pull a stable torch over this backend-matched
# nightly). When PF_RL=1 on cu130, pin to the exact nightly date the RL pins below were
# resolved against (cf. docker/engine.Dockerfile's TORCH_VERSION handling) instead of
# whatever the day's unpinned nightly happens to be. --------------------------
case "$backend" in
cu130 | rocm7.2 | cpu) index_url="https://download.pytorch.org/whl/nightly/$backend" ;;
*) die "unknown backend $backend" ;;
esac
if [ "$pf_rl" = "1" ] && [ "$backend" = "cu130" ]; then
	torch_spec="torch==${TORCH_VERSION}+${backend}"
else
	torch_spec="torch"
fi
log "installing $torch_spec from $index_url"
uv_pip --pre "$torch_spec" --index-url "$index_url" --force-reinstall

# --- forks: clone as siblings if absent, same "never touch an existing clone" rule as
# the Makefile's `forks` target (it's the engine hacker's working tree) ------
clone_fork() { # $1=dir $2=url $3=ref
	if [ -d "$1/.git" ]; then
		log "$1 exists at $(git -C "$1" rev-parse --short HEAD) (left untouched)"
	else
		log "cloning $2 -> $1 @ $3"
		git clone --quiet "$2" "$1"
		git -C "$1" checkout --quiet "$3"
	fi
}
clone_fork "$torchtitan_dir" "$torchtitan_url" "$torchtitan_ref"
clone_fork "$torchft_dir" "$torchft_url" "$torchft_ref"

uv_pip -r "$torchtitan_dir/requirements.txt"
if [ "$pf_dev_forks" = "1" ]; then
	uv_pip -e "$torchtitan_dir" -e "$torchft_dir"
else
	uv_pip "$torchtitan_dir"
	uv_pip "$torchft_dir"
fi
# transformers: torchtitan's transformers_modeling_backend imports it without declaring
# it anywhere -- see the Makefile's own install-torchtt-ft comment.
uv_pip transformers

# --- RL runtime (PF_RL=1): vllm, torchmonarch, torchstore, flash-attn-3, ... ----------
if [ "$pf_rl" = "1" ]; then
	[ "$backend" = "cu130" ] || die "RL runtime is CUDA-only; backend=$backend"
	test_index_url="https://download.pytorch.org/whl/test/$backend"
	log "RL runtime: vllm+torchvision from $index_url"
	uv_pip --pre \
		"torchvision==${RL_TORCHVISION_VERSION}+${backend}" \
		"vllm==${RL_VLLM_VERSION}+${backend}" \
		--extra-index-url "$index_url" --index-strategy unsafe-best-match
	uv_pip "torchmonarch==${RL_TORCHMONARCH_VERSION}" \
		"renderers==${RL_RENDERERS_VERSION}" pygtrie portpicker \
		opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
		"math-verify==${RL_MATH_VERIFY_VERSION}"
	printf 'mcp>=1.24,<2\n' >"$venv/rl-overrides.txt"
	uv_pip "verifiers==${RL_VERIFIERS_VERSION}" --override "$venv/rl-overrides.txt"
	uv_pip --no-deps \
		"torchstore @ https://github.com/meta-pytorch/torchstore/archive/${RL_TORCHSTORE_SHA}.tar.gz"
	uv_pip "flash-attn-3==${RL_FLASH_ATTN_3_VERSION}" --extra-index-url "$test_index_url"
fi

"$venv/bin/python" -c "$verify"
printf '%s' "$fp" >"$stamp"
log "done"
