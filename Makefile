# Makefile for installing the panofabric-engine project.
# Usage:
#   make all         # full bootstrap: toolchain, project, torch, the forks
#   make dev-forks   # engine hackers: sibling fork clones, installed editable


# ------------------------------------------------- the forks (no submodules)
# torchtitan/torchft live as SIBLINGS of this repo, cloned on demand. Submodules
# are gone from the public repo: `git clone --recursive` plus a Rust build is
# where a newcomer bounces. Keep these refs in sync with pyproject.toml's
# [train] extra — they are the same pins.
FORKS_DIR      ?= ..
TORCHTITAN_DIR ?= $(FORKS_DIR)/torchtitan
TORCHFT_DIR    ?= $(FORKS_DIR)/torchft
TORCHTITAN_URL ?= https://github.com/PanocularAI/torchtitan.git
TORCHFT_URL    ?= https://github.com/PanocularAI/torchft.git
TORCHTITAN_REF ?= 39f909614862def052998cc21815166641602619
TORCHFT_REF    ?= edad86ca1c8a95195961e555cf0ab3982bb860f7

TORCH_SPEC ?= torch
PYTORCH_BASE_URL ?= https://download.pytorch.org/whl/nightly

PROTOC_VERSION ?= 32.0
PROTOC_ZIP ?= protoc-$(PROTOC_VERSION)-linux-x86_64.zip
PROTOC_URL ?= https://github.com/protocolbuffers/protobuf/releases/download/v$(PROTOC_VERSION)/$(PROTOC_ZIP)
LOCAL_BIN ?= $(HOME)/.local/bin
EXPORT_LINE ?= export PATH=$$PATH:$(LOCAL_BIN)

UV ?= $(LOCAL_BIN)/uv
UV_PIP_CMD ?= $(UV) pip install
# torchft's Rust extension builds via pyo3 (caps at CPython 3.13); pin the built venv so a
# newer default python can't break the maturin build. Matches panofabric's env cache.
PYTHON_VERSION ?= 3.13
# The venv the install targets build into. Default ./.venv (uv's default, so `make all` is
# unchanged); `build-into` overrides it with an out-of-tree staging venv for the env cache.
VENV ?= .venv

.PHONY: all setup-env sync-project install-torch install-torchtt-ft forks dev-forks build-into ensure gc-env-cache show-backend clean-protoc-zip

all: setup-env sync-project install-torch install-torchtt-ft
	@[ "$${PF_RL:-0}" = "1" ] && $(MAKE) install-rl || \
	  echo "[all] PF_RL!=1; skipping the RL runtime"

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

# Resolve + install the project (panofabric-engine; its engine deps are the [train]
# extra's sha-pinned forks) into the in-tree .venv. Split out of setup-env so
# `build-into` (env cache) can reuse the toolchain from setup-env WITHOUT this
# project sync, building into an out-of-tree venv instead.
sync-project:
	$(UV) sync

# Backend detection extracted to scripts/pf_backend.sh — the SINGLE source of truth shared
# with the env-cache fingerprint (scripts/pf_env_fp.sh), so the cache KEY and the actual
# torch INSTALL can never drift. `export VIRTUAL_ENV` (target-scoped, propagated to sub-makes
# via the command-line VENV) targets `make all`'s ./.venv or build-into's staging venv.
install-torch: export VIRTUAL_ENV := $(abspath $(VENV))
install-torch:
	@backend=$$(sh scripts/pf_backend.sh); \
	case "$$backend" in \
		cu130|rocm7.0|cpu) \
			index_url="$(PYTORCH_BASE_URL)/$$backend"; \
			;; \
		*) \
			echo "[make install-torch] Unknown backend $$backend" >&2; \
			exit 1; \
			;; \
	esac; \
	echo "[make install-torch] Backend: $$backend"; \
	echo "[make install-torch] Index URL: $$index_url"; \
	set -x; \
	$(UV_PIP_CMD) --pre $(TORCH_SPEC) --index-url "$$index_url" --force-reinstall; \
	set +x

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
dev-forks: forks
	$(UV_PIP_CMD) -e $(TORCHTITAN_DIR) -e $(TORCHFT_DIR)

install-torchtt-ft: export VIRTUAL_ENV := $(abspath $(VENV))
install-torchtt-ft: forks
	$(UV_PIP_CMD) -r $(TORCHTITAN_DIR)/requirements.txt
	$(UV_PIP_CMD) $(TORCHTITAN_DIR)
	$(UV_PIP_CMD) $(TORCHFT_DIR)
	# transformers: required by panoengine.train.pretrain.hf_transformers (the
	# HF-architecture backend glue) — torchtitan's transformers_modeling_backend
	# imports it but declares it in neither its pyproject nor requirements.txt, so
	# that recipe fails to import on a node without this line. Unpinned deliberately
	# (the backend's own 4.57 pin predates the 5.x we run locally, matched to vLLM).
	$(UV_PIP_CMD) transformers

# ------------------------------------------------- RL runtime (PF_RL=1)
# The decentralized-RL engine needs a stack the training-only env does NOT: vLLM (the
# generator), torchmonarch (the actor mesh), torchstore, renderers, and flash-attn-3.
# These were previously installed ONLY in the panofabric engine IMAGE, so an image-less
# bootstrap (a Slurm site without Pyxis/enroot) produced a venv that dies with
# "ModuleNotFoundError: No module named 'monarch'" the moment an RL island starts.
#
# Pins are the image's, verbatim (panofabric infra/engine-image/Dockerfile) — they must
# move together:
#   - torchmonarch ships compiled bindings that load libtorch, so its build date must
#     track the torch nightly's.
#   - torchstore's PyPI release pins torch==2.9.0 and would drag the stack backwards, so
#     it is installed --no-deps from a pinned source tarball with its real deps by hand.
#   - flash-attn-3 comes from the pytorch TEST index, not nightly. The RL trainer imports
#     flash_attn_interface on Hopper+; pre-Hopper (e.g. L40S/Ada) falls back to FA2.
#   - math-verify is the dapo_math example rubric's dep; cheap and pure-python, so it
#     rides here rather than needing model.requirements (whose file is not packaged).
# Verified resolvable on python 3.13 (this Makefile's PYTHON_VERSION) 2026-08-18.
RL_TORCHMONARCH_VERSION ?= 0.7.0.dev20260805
RL_TORCHSTORE_SHA ?= 5a4d5d3f4d653f2ed7cc913a66e49f822dfd6c1d
RL_RENDERERS_VERSION ?= 0.1.9
RL_FLASH_ATTN_3_VERSION ?= 3.0.0
RL_VLLM_VERSION ?= 1.0.0.dev20260804
RL_TORCHVISION_VERSION ?= 0.29.0.dev20260805
RL_MATH_VERIFY_VERSION ?= 0.9.0

install-rl: export VIRTUAL_ENV := $(abspath $(VENV))
install-rl:
	@backend=$$(sh scripts/pf_backend.sh); \
	case "$$backend" in \
		cu130) : ;; \
		*) echo "[make install-rl] RL runtime is CUDA-only; backend=$$backend" >&2; exit 1;; \
	esac; \
	index_url="$(PYTORCH_BASE_URL)/$$backend"; \
	test_index_url="$(patsubst %/nightly,%,$(PYTORCH_BASE_URL))/test/$$backend"; \
	echo "[make install-rl] vllm+torchvision from $$index_url"; \
	set -x; \
	$(UV_PIP_CMD) --pre \
	  "torchvision==$(RL_TORCHVISION_VERSION)+$$backend" \
	  "vllm==$(RL_VLLM_VERSION)+$$backend" \
	  --extra-index-url "$$index_url" --index-strategy unsafe-best-match; \
	$(UV_PIP_CMD) "torchmonarch==$(RL_TORCHMONARCH_VERSION)" \
	  "renderers==$(RL_RENDERERS_VERSION)" pygtrie portpicker \
	  "math-verify==$(RL_MATH_VERIFY_VERSION)"; \
	$(UV_PIP_CMD) --no-deps \
	  "torchstore @ https://github.com/meta-pytorch/torchstore/archive/$(RL_TORCHSTORE_SHA).tar.gz"; \
	$(UV_PIP_CMD) "flash-attn-3==$(RL_FLASH_ATTN_3_VERSION)" \
	  --extra-index-url "$$test_index_url"; \
	set +x; \
	echo "[make install-rl] done"

# ------------------------------------------------- persistent env (`make ensure`)

# build-into: `all` built into a relocatable, SELF-CONTAINED, out-of-tree $(VENV) so the
# cached venv survives the builder's teardown. Non-editable (an -e install records an
# absolute finder into the builder's workdir, rm -rf'd at teardown -> dangling in every
# consumer). `--no-deps .` skips the redundant git engine pull (panofabric-engine's only
# deps are torchtitan/torchft, installed locally from the sibling clones next).
build-into: export VIRTUAL_ENV := $(abspath $(VENV))
build-into: setup-env
	$(UV) venv $(VENV) --relocatable --python $(PYTHON_VERSION)
	$(UV_PIP_CMD) --no-deps .
	$(MAKE) install-torch VENV=$(VENV)
	$(MAKE) install-torchtt-ft VENV=$(VENV)
	@[ "$${PF_RL:-0}" = "1" ] && $(MAKE) install-rl VENV=$(VENV) || \
	  echo "[build-into] PF_RL!=1; skipping the RL runtime"

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
	backend=$$(sh scripts/pf_backend.sh); \
	src=$$(sh scripts/pf_env_fp.sh --source); \
	[ -n "$$src" ] || { echo "[ensure] no fingerprint; per-run make all"; exec $(MAKE) all; }; \
	fp=$$(printf '%s:%s:rl%s' "$$src" "$$backend" "$${PF_RL:-0}" | sha256sum | cut -c1-16); \
	env="$$root/env-$$fp"; \
	restore() { rm -rf .venv; ln -s "$$env/.venv" .venv; \
	            mkdir -p "$(LOCAL_BIN)"; [ -x "$$env/uv" ] && ln -sf "$$env/uv" "$(LOCAL_BIN)/uv" || true; }; \
	if [ -f "$$env/.stamp" ] && "$$env/.venv/bin/python" -c 'import torch,torchtitan,torchft' 2>/dev/null; then \
	  echo "[ensure] HIT $$fp backend=$$backend"; restore; exit 0; fi; \
	stale=""; [ -e "$$env" ] && { stale=1; echo "[ensure] existing $$fp is unusable; rebuilding"; }; \
	tmp="$$root/.build.$$fp.$$$$.$$(hostname)"; rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	echo "[ensure] MISS $$fp backend=$$backend; building"; \
	$(MAKE) build-into VENV="$$tmp/.venv"; \
	cp -f "$(UV)" "$$tmp/uv" 2>/dev/null || true; \
	"$(UV)" pip freeze --python "$$tmp/.venv/bin/python" > "$$tmp/freeze.txt" 2>/dev/null || true; \
	"$$tmp/.venv/bin/python" -c 'import torch,torchtitan,torchft'; \
	printf '%s\n%s\n%s\n' "$$fp" "$$backend" "$$(date -u +%FT%TZ)" > "$$tmp/.stamp"; \
	if [ -n "$$stale" ]; then mv -T "$$env" "$$env.stale.$$$$" 2>/dev/null || rm -rf "$$env"; fi; \
	if mv -T "$$tmp" "$$env" 2>/dev/null; then echo "[ensure] published $$fp"; \
	else echo "[ensure] lost publish race; using existing $$fp"; rm -rf "$$tmp"; fi; \
	rm -rf "$$env.stale.$$$$"; \
	$(MAKE) gc-env-cache PF_ENV_CACHE_ROOT="$$root" || true; \
	restore

# Prune the per-fingerprint cache to the N most-recent entries (mtime). Staging dirs are
# `.build.*` (dot-prefixed) so this `env-*` glob never touches an in-flight build.
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
	ls -1dt "$$root"/env-* 2>/dev/null | tail -n +$$((keep + 1)) | while IFS= read -r d; do \
	  echo "[gc-env-cache] pruning $$d"; rm -rf "$$d"; done; \
	find "$$root" -maxdepth 1 -name '.build.*' -type d \
	  -mmin +$(PF_ENV_BUILD_STALE_MIN) 2>/dev/null | while IFS= read -r d; do \
	  echo "[gc-env-cache] pruning abandoned build $$d"; rm -rf "$$d"; done; \
	exit 0

show-backend:
	@make --no-print-directory -s install-torch UV_PIP_CMD="echo Would run: uv pip install" TORCH_SPEC="$(TORCH_SPEC)"

clean-protoc-zip:
	rm -f $(PROTOC_ZIP)