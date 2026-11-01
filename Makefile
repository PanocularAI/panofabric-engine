# Makefile for installing the panofabric-engine project.
# Usage:
#   make all         # full bootstrap: toolchain, then the project env (`make install`)
#   make dev-forks   # engine hackers: sibling fork clones, installed editable
#
# WHAT gets installed (torch per backend, the forks, the RL runtime, every pin) lives in
# pyproject.toml + uv.lock. This file only bootstraps the toolchain, picks the backend
# extra, overlays sibling fork clones, and runs the shared env cache (`make ensure`).


# ------------------------------------------------- the forks (no submodules)
# torchtitan/torchft are pinned by SHA in pyproject.toml's [train]/[decentralized] extras;
# `uv sync` installs them from those git pins. Sibling clones (cloned by `make forks`) are
# for engine hackers: when one exists, `make install` installs IT over the pin, so a fork
# edit is picked up without touching pyproject.toml. URL/REF are parsed out of the
# pyproject pins, never copied, so they cannot drift.
FORKS_DIR      ?= ..
TORCHTITAN_DIR ?= $(FORKS_DIR)/torchtitan
TORCHFT_DIR    ?= $(FORKS_DIR)/torchft
fork_pin = $(subst @, ,$(shell sed -n 's/.*"$(1) @ git+\([^"]*\)".*/\1/p' pyproject.toml | head -1))
TORCHTITAN_URL ?= $(firstword $(call fork_pin,torchtitan))
TORCHTITAN_REF ?= $(lastword $(call fork_pin,torchtitan))
TORCHFT_URL    ?= $(firstword $(call fork_pin,torchft))
TORCHFT_REF    ?= $(lastword $(call fork_pin,torchft))

PROTOC_VERSION ?= 32.0
PROTOC_ZIP ?= protoc-$(PROTOC_VERSION)-linux-x86_64.zip
PROTOC_URL ?= https://github.com/protocolbuffers/protobuf/releases/download/v$(PROTOC_VERSION)/$(PROTOC_ZIP)
LOCAL_BIN ?= $(HOME)/.local/bin
EXPORT_LINE ?= export PATH=$$PATH:$(LOCAL_BIN)

UV ?= $(LOCAL_BIN)/uv
# torchft's Rust extension builds via pyo3 (caps at CPython 3.13); pin the built venv so a
# newer default python can't break the maturin build. Matches panofabric's env cache.
PYTHON_VERSION ?= 3.13
# The venv `install` syncs. Default ./.venv (uv's default); `build-into` overrides it with
# an out-of-tree staging venv for the env cache.
VENV ?= .venv
# Extra `uv sync` flags (build-into adds --no-editable).
UV_SYNC_FLAGS ?=

.PHONY: all setup-env install forks dev-forks build-into ensure gc-env-cache show-backend clean-protoc-zip

all: setup-env install

# $HOME/.cargo/bin is on PATH so the torchft maturin build finds cargo/rustc after a
# user-local rustup install (this was missing before — a latent no-root bug).
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

# No-root-friendly: `make all` is ONLY the native-node bootstrap (the container path is
# docker/engine.Dockerfile, which mirrors this at build time). So it must work on a
# no-root Slurm node too: apt + tailscale are gated on passwordless sudo (cloud VMs keep
# the old behavior), and protoc/rust/uv all install user-local under $HOME.
setup-env:
	@echo "Setting up environment..."
	@mkdir -p $(HOME)/.local/bin
	@if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then \
		echo "[setup-env] sudo available -> installing build deps via apt"; \
		sudo apt-get update -y; \
		sudo apt-get install -y unzip curl ca-certificates gnupg wget build-essential pkg-config libssl-dev; \
	else \
		echo "[setup-env] no passwordless sudo (HPC node) -> skipping apt; assuming a toolchain is present"; \
		command -v cc >/dev/null 2>&1 || echo "[setup-env] WARNING: no C compiler on PATH. On HPC run e.g. 'module load gcc' before 'make all'."; \
		command -v unzip >/dev/null 2>&1 || echo "[setup-env] WARNING: 'unzip' not found (needed for protoc). 'module load' it if the build fails."; \
	fi
	# protoc -> $HOME/.local (needed by the torchft build); no root.
	@if ! command -v protoc >/dev/null 2>&1; then \
		wget -q -O $(PROTOC_ZIP) "$(PROTOC_URL)" && unzip -o $(PROTOC_ZIP) -d $(HOME)/.local && rm -f $(PROTOC_ZIP); \
	fi
	@if ! grep -qx '$(EXPORT_LINE)' $(HOME)/.bashrc 2>/dev/null; then echo '$(EXPORT_LINE)' >> $(HOME)/.bashrc; fi
	# Tailscale is a host/overlay concern (docker/engine.Dockerfile drops it too): only
	# with sudo, and never fail the build on it.
	@if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then \
		( curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null && \
		  curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null && \
		  sudo apt-get update -y && sudo apt-get install -y tailscale ) || echo "[setup-env] tailscale install skipped (non-fatal)"; \
	fi
	# Rust (user-local rustup) for the torchft maturin build. Gate on `rustup`, not
	# `cargo`: a shared node often has a rustup/cargo SHIM on PATH with NO default
	# toolchain, so `command -v cargo` passes but the build fails with "no default is
	# configured". Install rustup if absent, then set a default toolchain explicitly
	# (idempotent — a no-op when one is already set).
	@command -v rustup >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable
	@rustup default stable
	# uv (user-local) for installs.
	@[ -x $(LOCAL_BIN)/uv ] || command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# The whole env in one `uv sync`: the project, the [train] extra (torchtitan + torchft from
# their git pins), and ONE backend extra, chosen from scripts/pf_backend.sh -- the SINGLE
# source of truth shared with the env-cache fingerprint (scripts/pf_env_fp.sh), so the
# cache KEY and the actual torch INSTALL can never drift. PF_RL=1 swaps cu130 for `rl`
# (cu130 + vLLM/torchmonarch/torchstore/...; CUDA-only). --locked fails loudly if
# pyproject.toml was edited without re-running `uv lock`.
#
# A fork with a sibling clone is excluded from the sync and installed from the clone
# instead (editable under PF_DEV_FORKS=1). That overlay must follow EVERY sync: an exact
# sync uninstalls packages it was told not to install.
install: export UV_PROJECT_ENVIRONMENT := $(abspath $(VENV))
install: export VIRTUAL_ENV := $(abspath $(VENV))
install:
	@set -e; \
	backend=$$(sh scripts/pf_backend.sh); \
	case "$$backend:$${PF_RL:-0}" in \
		cu130:1) extra=rl ;; \
		*:1) echo "[make install] RL runtime is CUDA-only; backend=$$backend" >&2; exit 1 ;; \
		cu130:*) extra=cu130 ;; \
		rocm7.2:*) extra=rocm ;; \
		cpu:*) extra=cpu ;; \
		*) echo "[make install] Unknown backend $$backend" >&2; exit 1 ;; \
	esac; \
	skip=""; overlay=""; edit=""; \
	[ "$${PF_DEV_FORKS:-0}" != "1" ] || edit="-e"; \
	for spec in "torchtitan|$(TORCHTITAN_DIR)" "torchft|$(TORCHFT_DIR)"; do \
		name=$${spec%%|*}; dir=$${spec#*|}; \
		if [ -d "$$dir/.git" ]; then \
			skip="$$skip --no-install-package $$name"; \
			overlay="$$overlay $$edit $$dir"; \
		fi; \
	done; \
	echo "[make install] backend=$$backend extra=$$extra forks=$${overlay:- (git pins)}"; \
	set -x; \
	$(UV) sync --locked $(UV_SYNC_FLAGS) --python $(PYTHON_VERSION) \
		--extra train --extra "$$extra" $$skip; \
	if [ -n "$$overlay" ]; then $(UV) pip install $$overlay; fi

# Clone the forks as siblings if they are ABSENT (at the pinned ref). An existing
# clone is left exactly as it is — never checked out to the pin, because it is the
# engine hacker's working tree and a detached HEAD would strand their branch.
forks:
	@set -e; \
	for spec in "$(TORCHTITAN_DIR)|$(TORCHTITAN_URL)|$(TORCHTITAN_REF)" \
	            "$(TORCHFT_DIR)|$(TORCHFT_URL)|$(TORCHFT_REF)"; do \
	  dir=$${spec%%|*}; rest=$${spec#*|}; url=$${rest%%|*}; ref=$${rest##*|}; \
	  if [ -d "$$dir/.git" ]; then \
	    echo "[forks] $$dir exists at $$(git -C "$$dir" rev-parse --short HEAD) (left untouched)"; \
	  else \
	    echo "[forks] cloning $$url -> $$dir @ $$ref"; \
	    git clone --quiet "$$url" "$$dir"; \
	    git -C "$$dir" checkout --quiet "$$ref"; \
	  fi; \
	done

# Engine hackers: the forks installed EDITABLE, so a change in ../torchtitan is
# live without a reinstall. This is what replaced `git submodule update --init`.
# Re-run it after any plain `make install`/`uv sync`, which put the forks back to
# regular installs.
dev-forks: forks
	$(MAKE) install PF_DEV_FORKS=1

# ------------------------------------------------- persistent env (`make ensure`)

# build-into: `all` built into a relocatable, SELF-CONTAINED, out-of-tree $(VENV) so the
# cached venv survives the builder's teardown. Non-editable (an -e install records an
# absolute finder into the builder's workdir, rm -rf'd at teardown -> dangling in every
# consumer).
# Copy, not uv's default hardlinks from its cache: the cache is ONE shared dir
# (~/.cache/uv), so every cached env shared each file's inode with it and with every
# other env (link counts of 6-8 seen live). On BeeGFS, opening such files while
# concurrent builds added links to them failed with ENOENT on files that exist --
# killing generator islands at import (transformers, torch, sympy files). Copies cost
# ~7 GB of real disk per cached env, once per fingerprint.
build-into: export UV_LINK_MODE := copy
# The shared interpreter (ensure's UV_PYTHON_INSTALL_DIR) is installed ONCE, under a
# mkdir lock: uv's own lock is flock(), which BeeGFS keeps per node. On a fresh cache,
# gpu08 and gpu09 both installed CPython 3.13.16 at once and one swapped the files out
# from under the other's builds. The venv then may not download one of its own.
build-into: setup-env
	@if [ -n "$${UV_PYTHON_INSTALL_DIR:-}" ]; then \
	  lock="$$UV_PYTHON_INSTALL_DIR.lock"; \
	  until mkdir "$$lock" 2>/dev/null; do \
	    if [ -n "$$(find "$$lock" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then \
	      echo "[build-into] clearing a stale Python-install lock"; rmdir "$$lock" 2>/dev/null; \
	    else sleep 5; fi; \
	  done; \
	  "$(UV)" python install $(PYTHON_VERSION); rc=$$?; rmdir "$$lock"; exit $$rc; \
	fi
	env $${UV_PYTHON_INSTALL_DIR:+UV_PYTHON_DOWNLOADS=never} \
	  $(UV) venv $(VENV) --relocatable --python $(PYTHON_VERSION)
	$(MAKE) install VENV=$(VENV) UV_SYNC_FLAGS=--no-editable

ensure:
	@set -eu; \
	[ "$${PF_ENV_CACHE:-1}" != "0" ] || { echo "[ensure] PF_ENV_CACHE=0; per-run make all"; exec $(MAKE) all; }; \
	root="$${PF_ENV_CACHE_DIR:-}"; \
	if [ -z "$$root" ]; then case "$$HOME" in \
	  */.sky_clusters/*) root="$${HOME%/.sky_clusters/*}/.panofabric-cache";; \
	  *)                 root="$$HOME/.cache/panofabric-env";; esac; fi; \
	mkdir -p "$$root"; \
	fstype=$$(stat -f -c %T "$$root" 2>/dev/null || echo unknown); \
	case " $${PF_ENV_SHARED_FS:-nfs nfs4 lustre gpfs beegfs fhgfs panfs cephfs} " in *" $$fstype "*) : ;; \
	  *) case "$$HOME" in */.sky_clusters/*|/root*) \
	       echo "[ensure] $$root not shared ($$fstype); per-run make all"; exec $(MAKE) all;; \
	     esac;; esac; \
	export UV_PYTHON_INSTALL_DIR="$$root/uv-python"; \
	export UV_CACHE_DIR="$$root/uv-cache/$$(hostname)"; \
	backend=$$(sh scripts/pf_backend.sh); \
	verify='import torch,torchtitan,torchft'; \
	case "$${PF_RL:-0}" in 1) verify="$$verify,vllm,torchstore";; esac; \
	src=$$(PF_FORK_PINS="torchtitan=$(TORCHTITAN_REF) torchft=$(TORCHFT_REF)" \
	  sh scripts/pf_env_fp.sh --source); \
	[ -n "$$src" ] || { echo "[ensure] no fingerprint; per-run make all"; exec $(MAKE) all; }; \
	fp=$$(printf '%s:%s:rl%s' "$$src" "$$backend" "$${PF_RL:-0}" | sha256sum | cut -c1-16); \
	env="$$root/env-$$fp"; \
	restore() { rm -rf .venv; ln -s "$$env/.venv" .venv; \
	            mkdir -p "$(LOCAL_BIN)"; [ -x "$$env/uv" ] && ln -sf "$$env/uv" "$(LOCAL_BIN)/uv" || true; }; \
	if [ -f "$$env/.stamp" ] && "$$env/.venv/bin/python" -c "$$verify" 2>/dev/null; then \
	  echo "[ensure] HIT $$fp backend=$$backend"; restore; exit 0; fi; \
	stale=""; [ -e "$$env" ] && { stale=1; echo "[ensure] existing $$fp is unusable; rebuilding"; }; \
	tmp="$$root/.build.$$fp.$$$$.$$(hostname)"; rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	echo "[ensure] MISS $$fp backend=$$backend; building"; \
	$(MAKE) build-into VENV="$$tmp/.venv"; \
	cp -f "$(UV)" "$$tmp/uv" 2>/dev/null || true; \
	"$(UV)" pip freeze --python "$$tmp/.venv/bin/python" > "$$tmp/freeze.txt" 2>/dev/null || true; \
	"$$tmp/.venv/bin/python" -c "$$verify"; \
	printf '%s\n%s\n%s\n' "$$fp" "$$backend" "$$(date -u +%FT%TZ)" > "$$tmp/.stamp"; \
	if [ -n "$$stale" ]; then mv -T "$$env" "$$env.stale.$$$$" 2>/dev/null || rm -rf "$$env"; fi; \
	rm -rf "$$env.stale.$$$$"; \
	if mv -T "$$tmp" "$$env" 2>/dev/null; then echo "[ensure] published $$fp"; \
	  $(MAKE) gc-env-cache PF_ENV_CACHE_ROOT="$$root" PF_ENV_CACHE_CURRENT="$$env" || true; \
	else echo "[ensure] lost publish race; using existing $$fp"; rm -rf "$$tmp"; fi; \
	restore

# Prune the per-fingerprint cache to the N most-recently PUBLISHED entries, run only by the
# build that just published (PF_ENV_CACHE_CURRENT, which it never prunes). Ranked by each
# entry's .stamp, written once at publish -- not the dir mtime, which an `rm -rf` in progress
# keeps refreshing: four concurrent prunes then ranked half-deleted old envs above the one
# just published and deleted it under all four generators (run 7a2d6145e4dd). Staging dirs
# are `.build.*` (dot-prefixed) so this `env-*` glob never touches an in-flight build.
#
# ALSO sweeps ABANDONED staging dirs. A build that is killed mid-flight (scancel, node
# failure, a run torn down before setup finishes) never reaches its `mv -T`, so it strands
# a multi-GB `.build.*` tree forever — observed live at 7.3GB from one interrupted build.
# Only dirs untouched for PF_ENV_BUILD_STALE_MIN minutes are removed, so a CONCURRENT
# builder on another node (actively writing, hence fresh mtime) is never disturbed.
PF_ENV_CACHE_KEEP ?= 3
PF_ENV_BUILD_STALE_MIN ?= 240
gc-env-cache:
	@root="$(PF_ENV_CACHE_ROOT)"; keep=$(PF_ENV_CACHE_KEEP); \
	[ -n "$$root" ] && [ -d "$$root" ] || exit 0; \
	ls -1t "$$root"/env-*/.stamp 2>/dev/null | tail -n +$$((keep + 1)) | while IFS= read -r s; do \
	  d="$${s%/.stamp}"; [ "$$d" = "$(PF_ENV_CACHE_CURRENT)" ] && continue; \
	  echo "[gc-env-cache] pruning $$d"; rm -rf "$$d"; done; \
	find "$$root" -maxdepth 1 -name 'env-*' -type d -mmin +$(PF_ENV_BUILD_STALE_MIN) \
	  ! -exec test -e {}/.stamp \; -print 2>/dev/null | while IFS= read -r d; do \
	  echo "[gc-env-cache] pruning half-deleted $$d"; rm -rf "$$d"; done; \
	find "$$root" -maxdepth 1 -name '.build.*' -type d \
	  -mmin +$(PF_ENV_BUILD_STALE_MIN) 2>/dev/null | while IFS= read -r d; do \
	  echo "[gc-env-cache] pruning abandoned build $$d"; rm -rf "$$d"; done; \
	find "$$root/uv-cache" -mindepth 1 -maxdepth 1 -type d -mtime +14 \
	  -exec rm -rf {} + 2>/dev/null; \
	exit 0

show-backend:
	@$(MAKE) --no-print-directory -s install UV="echo Would run: uv"

clean-protoc-zip:
	rm -f $(PROTOC_ZIP)