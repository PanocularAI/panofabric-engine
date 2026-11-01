#!/bin/sh
# Print the SOURCE fingerprint of the training env `make all` would build — the
# backend-INDEPENDENT half of the env-cache key. The `ensure` Make target combines it with
# the node's torch backend (scripts/pf_backend.sh) into the per-fingerprint cache dir:
# `fp = sha256("<source_fp>:<backend>")[:16]`. See the panofabric env-cache design
# (docs/slurm-setup-cache-plan.md) — this is the panofabric-engine side of it.
#
# POSIX sh; invoked as `sh scripts/pf_env_fp.sh --source`.
#
# `--editable` is the key for an env whose engine AND forks are installed editable
# (scripts/nix_dev_setup.sh's default): a .py edit there is live without reinstalling,
# so only what changes the INSTALL is hashed -- the manifests, the lockfile, the
# Makefile pins, and sources that get COMPILED at install time (torchft's Rust
# extension). Hashing every .py, as `--source` must for a copied-in venv, rebuilt the
# whole dev env (uv sync uninstalls ~260 packages first) on every source edit, and on
# every checkpoint downloaded into a fork (its merges.txt matched *.txt).
set -eu

mode="${1:---source}"
case "$mode" in
--source | --editable) ;;
*) echo "usage: pf_env_fp.sh [--source|--editable]" >&2; exit 2 ;;
esac

py="${PF_PYTHON_VERSION:-3.13}"

# Content hash of the install-determining source: the engine's own `models` and
# `panoengine` trees plus the
# FORKS as they will actually be INSTALLED, restricted to code+manifest files, plus the
# top-level manifests and the Makefile.
#
# The fork roots come from FORKS_DIR (default `..`, matching the Makefile), NOT from
# in-repo `torchtitan/`/`torchft/` directories. Hashing the in-repo copies was wrong in
# both directions and cost a live debug (2026-08-28):
#   - `make install` installs $(FORKS_DIR)/torchft, so an edit to the REAL fork left
#     the key unchanged and `ensure` happily reused a venv built from the old code -- a
#     fix that appeared to do nothing, silently;
#   - a stale leftover copy inside the repo DID change the key, forcing a multi-GB rebuild
#     that installed byte-identical packages.
# When a fork is absent (a fresh node clones it during the build), it contributes nothing
# and the fork VERSION is still covered: pyproject.toml pins both SHAs (installed from
# those git pins when no clone exists) and it is hashed below.
#
# Only file CONTENT is hashed, not paths, so the key does not shift when the forks sit at
# a different FORKS_DIR (an in-tree checkout vs a sibling) -- what matters for a venv is
# what gets installed, not where it was read from.
# Prune DERIVED trees — build/, dist/, torchft/target, __pycache__, .venv, .git, egg-info —
# and bulky binary asset/output dirs, so the hash reflects SOURCE only: a stray `build/` left
# by a local install must not change the fingerprint (and thus must not invalidate a
# perfectly good cache entry). sha256sum reads each file's CONTENT; sort makes it
# order-independent.
forks="${FORKS_DIR:-..}"

# A fork checked out CLEANLY at its pin (PF_FORK_PINS="name=ref ...", from the
# Makefile, parsed from pyproject.toml) adds nothing the hashed pyproject.toml does
# not already pin -- and hashing it made the key depend on whether a clone happened
# to exist: a build cloned the forks next to the workdir, so a relaunch on the same
# node hashed them and
# rebuilt the whole env (run 8e0a2f714cae). So a fork is hashed only when it
# differs from its pin (a dev checkout with edits, or another commit). Without
# PF_FORK_PINS (a standalone call) every fork present is hashed, as before.
fork_dirs="$forks/torchtitan $forks/torchft"
if [ -n "${PF_FORK_PINS:-}" ]; then
	fork_dirs=""
	for pin in $PF_FORK_PINS; do
		dir="$forks/${pin%%=*}"
		[ -d "$dir" ] || continue
		head=$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)
		want=$(git -C "$dir" rev-parse "${pin#*=}^{commit}" 2>/dev/null || true)
		if [ -n "$head" ] && [ "$head" = "$want" ] &&
			[ -z "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
			continue
		fi
		fork_dirs="$fork_dirs $dir"
	done
fi

if [ "$mode" = "--editable" ]; then
	src=$(
		{
			# shellcheck disable=SC2086 # fork_dirs is a space-separated list of paths
			for dir in $fork_dirs; do
				[ -d "$dir" ] || continue
				# Packaging manifests at the fork root. -L: torchtitan's
				# requirements*.txt are symlinks into .ci/.
				find -L "$dir" -maxdepth 1 -type f \( -name pyproject.toml \
					-o -name 'setup.py' -o -name 'setup.cfg' -o -name 'requirements*.txt' \
					-o -name 'Cargo.toml' -o -name 'Cargo.lock' -o -name 'build.rs' \) -print0
				# Compiled-at-install sources: an editable install does not pick these up.
				find "$dir" \( -type d \( -name target -o -name .git -o -name build \
					-o -name dist -o -name node_modules \) -prune \) -o \
					\( -type f \( -name '*.rs' -o -name '*.proto' \) -print0 \)
			done 2>/dev/null
			printf 'pyproject.toml\0uv.lock\0Makefile\0'
		} | xargs -0 sha256sum 2>/dev/null | awk '{ print $1 }' | sort -u |
			sha256sum | cut -c1-64
	)
	printf 'sl-v1-editable|py=%s|src=%s' "$py" "$src" | sha256sum | cut -c1-40 | tr -d '\n'
	exit 0
fi

src=$(
	{
		# shellcheck disable=SC2086 # fork_dirs is a space-separated list of paths
		find models panoengine $fork_dirs \
			\( -type d \( -name target -o -name __pycache__ -o -name .git \
				-o -name node_modules -o -name '*.egg-info' -o -name .venv \
				-o -name build -o -name dist \
				-o -name assets -o -name outputs \) -prune \) -o \
			\( -type f \( -name '*.py' -o -name '*.rs' -o -name '*.toml' \
				-o -name '*.txt' -o -name '*.cfg' -o -name '*.lock' \
				-o -name '*.in' \) -print0 \) 2>/dev/null
		printf 'pyproject.toml\0uv.lock\0Makefile\0'
	} | xargs -0 sha256sum 2>/dev/null | awk '{ print $1 }' | sort |
		sha256sum | cut -c1-64
)
[ -n "$src" ] || exit 0   # nothing hashed (wrong CWD) -> empty; ensure falls back to make all

printf 'sl-v1|py=%s|src=%s' "$py" "$src" | sha256sum | cut -c1-40 | tr -d '\n'
