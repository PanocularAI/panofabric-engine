#!/bin/sh
# Print the SOURCE fingerprint of the training env `make all` would build — the
# backend-INDEPENDENT half of the env-cache key. The `ensure` Make target combines it with
# the node's torch backend (scripts/pf_backend.sh) into the per-fingerprint cache dir:
# `fp = sha256("<source_fp>:<backend>")[:16]`. See the panofabric env-cache design
# (docs/slurm-setup-cache-plan.md) — this is the symphony-learn side of it.
#
# POSIX sh; invoked as `sh scripts/pf_env_fp.sh --source`.
set -eu

py="${PF_PYTHON_VERSION:-3.13}"

# Content hash of the install-determining source: the engine trees (models/torchtitan/
# torchft) restricted to code+manifest files, plus the top-level manifests and the Makefile.
# Prune DERIVED trees — build/, dist/, torchft/target, __pycache__, .venv, .git, egg-info —
# and bulky binary asset/output dirs, so the hash reflects SOURCE only: a stray `build/` left
# by a local install must not change the fingerprint (and thus must not invalidate a
# perfectly good cache entry). sha256sum reads each file's CONTENT; sort makes it
# order-independent.
src=$(
	{
		find models torchtitan torchft \
			\( -type d \( -name target -o -name __pycache__ -o -name .git \
				-o -name node_modules -o -name '*.egg-info' -o -name .venv \
				-o -name build -o -name dist \
				-o -name assets -o -name outputs \) -prune \) -o \
			\( -type f \( -name '*.py' -o -name '*.rs' -o -name '*.toml' \
				-o -name '*.txt' -o -name '*.cfg' -o -name '*.lock' \
				-o -name '*.in' \) -print0 \) 2>/dev/null
		printf 'pyproject.toml\0uv.lock\0Makefile\0'
	} | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -c1-64
)
[ -n "$src" ] || exit 0   # nothing hashed (wrong CWD) -> empty; ensure falls back to make all

printf 'sl-v1|py=%s|src=%s' "$py" "$src" | sha256sum | cut -c1-40 | tr -d '\n'
