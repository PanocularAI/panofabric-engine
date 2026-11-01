#!/bin/sh
# Sets up the training env WITHOUT going through the Makefile: one `uv sync` with the
# backend extra, plus the sibling torchtitan/torchft forks -- the same outcome as `make all`
# (plus `make dev-forks`), driven by flake.nix's devShell/direnv instead. See flake.nix for why:
# the Makefile's own `setup-env` assumes things (an apt-managed toolchain, a writable
# ~/.bashrc) that don't hold under NixOS/home-manager, and the toolchain itself (protoc,
# rust, uv, gcc) is supplied by the flake's devShell instead of setup-env's curl|sh/apt
# paths. Everything after setup-env (the Makefile's `install` target), this script
# reimplements directly.
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
# Everything that gets installed is pinned in pyproject.toml/uv.lock. The fork URL+SHA are
# parsed out of the [train]/[decentralized] git pins (as the Makefile does) only to clone
# missing siblings.
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

pf_rl="${PF_RL:-0}"
pf_dev_forks="${PF_DEV_FORKS:-1}"

# --- skip if nothing install-determining changed ----------------------------
backend=$(sh "$repo/scripts/pf_backend.sh")
# Editable forks (the default) -> everything this script installs from source is
# editable (`uv sync` installs the engine editable too), so source edits need no
# reinstall and must not trigger one; see pf_env_fp.sh --editable.
if [ "$pf_dev_forks" = "1" ]; then
	src_fp=$(sh "$repo/scripts/pf_env_fp.sh" --editable)
else
	src_fp=$(sh "$repo/scripts/pf_env_fp.sh" --source)
fi
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

# --- project venv: one sync, same extra selection as the Makefile's `install` -------
case "$backend:$pf_rl" in
cu130:1) extra=rl ;;
*:1) die "RL runtime is CUDA-only; backend=$backend" ;;
cu130:*) extra=cu130 ;;
rocm7.2:*) extra=rocm ;;
cpu:*) extra=cpu ;;
*) die "unknown backend $backend" ;;
esac
log "uv sync --extra train --extra $extra (forks from the siblings)"
uv sync --locked --extra train --extra "$extra" \
	--no-install-package torchtitan --no-install-package torchft

# The forks from the sibling clones. Must follow every sync: an exact sync uninstalls the
# packages it was told not to install.
if [ "$pf_dev_forks" = "1" ]; then
	VIRTUAL_ENV="$venv" uv pip install -e "$torchtitan_dir" -e "$torchft_dir"
else
	VIRTUAL_ENV="$venv" uv pip install "$torchtitan_dir" "$torchft_dir"
fi

"$venv/bin/python" -c "$verify"
printf '%s' "$fp" >"$stamp"
log "done"
