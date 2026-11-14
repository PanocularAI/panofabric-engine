{
  description = ''
    Dev shells for panofabric-engine on this NixOS cluster. Does not invoke the
    Makefile at all: it supplies the native toolchain (uv, protoc, rust, gcc, ...)
    as real nix packages, and its shellHook drives scripts/nix_dev_setup.sh --
    a standalone reimplementation of `make all`/`make dev-forks` -- instead of
    `make`. See that script's header for why, and for the pins it reads
    straight out of pyproject.toml/Makefile (as data, never via `make`) so
    there is still only one place those live.

    Three shells share the same toolchain and differ only in which install
    path nix_dev_setup.sh (and, for `serving`, a post-setup step) takes:
      default  -- training / decentralized / relay / serving-glue. No vLLM.
      serving  -- default + a real, unpinned (stable-torch) vllm, for
                  panoengine.serve.engine_stage's actual pipeline stage.
      rl       -- PF_RL=1: the matched RL runtime (vllm/torchmonarch/
                  torchstore/flash-attn-3, pinned together, torch force-
                  reinstalled to match) -- see the Makefile's install-rl.
    Pick one with `nix develop .#<name>` / direnv's `use flake .#<name>`; see
    .envrc / .envrc.local for how this repo selects one automatically.
  '';

  inputs = {
    # Pinned to the same rev the other flakes under /mnt/cluster/dev use
    # (tensorrt_rust_experiments, llama.cpp-gpu's nixpkgs-instances.nix), so the
    # toolchain here shares store paths with them instead of refetching/rebuilding.
    # Bump the rev to update.
    nixpkgs.url = "github:NixOS/nixpkgs/5545adfad2e98de106a5544ca7067e03010410bd";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # CUDA toolkit below.
        };

        # ONE CUDA prefix in the classic FHS `bin/` + `include/` + `lib64/`
        # layout, for flashinfer (pulled in by vllm), which JIT-compiles its
        # kernels with `$CUDA_HOME/bin/nvcc` and links them with a hardcoded
        # `-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs`.
        #
        # nixpkgs ships CUDA as per-component derivations, each keeping its
        # libraries under lib/ with no lib64 at all, so the join gives the
        # single prefix flashinfer assumes and `lib64 -> lib` lands its
        # hardcoded suffix on the real libraries. Pinned here rather than
        # taken from the host's system-wide CUDA_PATH (cuda-merged-12.9 on this
        # cluster) so nvcc's version matches the torch wheels: 13.0 is what
        # scripts/pf_backend.sh's `cu130` backend (and the RL pins built
        # against it) use. Bump alongside that backend.
        #
        # Multi-output components must be listed output by output: a bare
        # `libcurand` contributes only its default output (no headers).
        cudaToolkitRoot = pkgs.symlinkJoin {
          name = "cuda-toolkit-root-13.0";
          paths = with pkgs.cudaPackages_13_0; [
            cuda_nvcc
            # CUDA 13 moved `crt/` (host_config.h etc., included by nvcc's
            # own headers) out of cuda_nvcc into its own component.
            cuda_crt
            cuda_cudart # also carries lib/stubs/libcuda.so for `-lcuda`
            cuda_culibos
            cccl # <cub/...>, <thrust/...>, <cuda/std/...>
            # <curand.h>: flashinfer's sampling kernels (the ones vLLM JITs on
            # its first sampled token) include it.
            libcurand
            libcurand.dev
            libcurand.include
            libcurand.lib
          ];
          # A plain `nvcc foo.cu` must find <cuda_runtime.h> etc. in the joined
          # include/ too -- flashinfer happens to pass `-isystem
          # $CUDA_HOME/include` itself, but nothing else is obliged to. nvcc
          # reads nvcc.profile (and resolves its TOP) relative to its REAL
          # binary, and nixpkgs' profile hardcodes cuda_nvcc's own include dir,
          # so through a symlink it only ever sees cuda_nvcc's headers. Hence:
          # copy the binary in, give it a profile pointing at the joined
          # include/, and re-wrap it like nixpkgs does (gcc on PATH).
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            ln -sfn $out/lib $out/lib64

            nvcc_pkg=${pkgs.cudaPackages_13_0.cuda_nvcc}
            rm $out/bin/nvcc $out/bin/.nvcc-wrapped $out/bin/nvcc.profile
            cp $nvcc_pkg/bin/.nvcc-wrapped $out/bin/.nvcc-wrapped
            sed "s|$nvcc_pkg/include|$out/include|" $nvcc_pkg/bin/nvcc.profile \
              > $out/bin/nvcc.profile
            makeWrapper $out/bin/.nvcc-wrapped $out/bin/nvcc --inherit-argv0 \
              --prefix PATH : ${pkgs.cudaPackages_13_0.backendStdenv.cc}/bin
          '';
        };

        commonPackages = with pkgs; [
          uv # scripts/nix_dev_setup.sh's `uv sync` / `uv pip install`.
          # Already on PATH via home-manager for some users; pinned here so
          # the shell is self-contained for anyone else on the cluster too.

          protobuf # protoc, for torchft's maturin/pyo3 build. README pins
          # protoc 32.0; this nixpkgs rev carries 35.1. protoc's wire format
          # and codegen are stable across this range -- bump the pinned
          # nixpkgs rev above if that ever stops being true.

          rustup # nix_dev_setup.sh runs `rustup default stable` once to
          # fetch a real toolchain. Real upstream rustc/cargo are
          # dynamically-linked, non-nix binaries -- they run here only
          # because nix-ld is enabled cluster-wide ($NIX_LD is set). That's
          # the actual nix/nixos integration point for Rust: nix supplies
          # the launcher, nix-ld lets what it downloads execute.

          gcc # stdenv.cc: links torchft's maturin build and any C-extension
          # wheel that falls back to a source build.
          pkg-config
          openssl

          ninja # flashinfer (pulled in by vllm) JIT-compiles its CUDA sampling
          # kernels on first use via `ninja`, not declared as one of its own
          # deps -- without it on PATH, panoengine.serve.engine_stage crashes
          # the first time it samples a token with
          # `FileNotFoundError: ninja`.

          unzip
          curl
          wget
          git
        ];

        # Shared shellHook body. `pfRl` picks nix_dev_setup.sh's install path
        # (its own header documents PF_RL): "0" is the training/decentralized/
        # serving-glue env this repo has always had; "1" drives the Makefile's
        # install-rl path (pinned vllm/torchmonarch/torchstore/flash-attn-3,
        # with torch force-reinstalled to the exact nightly those pins were
        # built against). `postSetup`, run after nix_dev_setup.sh, is for the
        # `serving` shell's extra unpinned `uv pip install vllm`.
        mkShellHook = { pfRl, label, postSetup ? "" }: ''
          # --- machine-local state ---------------------------------------------
          # The repo lives on a shared NFS mount; venvs (which hold compiled,
          # per-machine native extensions) must NOT. $HOME is on local disk on
          # each node, so this path is implicitly per-machine; the repo-root hash
          # keeps multiple checkouts on one machine from colliding. Same layout
          # as SuperScalar-PipeInfer's .envrc. Keyed on the repo root rather than
          # $PWD so `nix develop` from a subdirectory lands on the same state.
          #
          # One venv PER SHELL (the trailing label): default/serving/rl install
          # mutually incompatible torch builds (nightly vs vllm's stable vs the
          # RL-pinned nightly), so a shared venv would be clobbered and fully
          # reinstalled on every switch between them.
          #
          # If any node ever mounts $HOME over NFS, add "$(hostname -s)" here.
          _pf_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          _pf_key="$(printf '%s' "$_pf_root" | sha1sum | cut -c1-12)"
          export PF_STATE_DIR="$HOME/.local/state/panofabric-engine/$_pf_key/${label}"
          mkdir -p "$PF_STATE_DIR/pycache"
          unset _pf_root _pf_key

          # uv (and scripts/nix_dev_setup.sh, which reads the same var) put the
          # project venv here instead of the in-tree ./.venv.
          export UV_PROJECT_ENVIRONMENT="$PF_STATE_DIR/venv"

          # python bytecode -> local (keeps __pycache__ out of the shared tree).
          export PYTHONPYCACHEPREFIX="$PF_STATE_DIR/pycache"

          # hardware.nvidia's userspace driver lives here on this cluster's NixOS
          # config, not under a standard FHS path. torch/vllm's CUDA wheels (and
          # scripts/pf_backend.sh's own libcuda.so.1 probe) need it on
          # LD_LIBRARY_PATH at runtime.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # Triton's nvidia driver backend locates libcuda.so.1 by shelling out
          # to the hardcoded path /sbin/ldconfig, which doesn't exist on NixOS
          # (ldconfig lives under /nix/store, never at /sbin) -- every
          # vLLM/triton codepath (panoengine.serve.engine_stage included)
          # crashes with `FileNotFoundError: /sbin/ldconfig` at first kernel
          # compile without this. Triton's TRITON_LIBCUDA_PATH knob skips that
          # subprocess entirely and points straight at the real driver libs.
          export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"

          # The flake's own pinned CUDA toolkit (see cudaToolkitRoot), replacing
          # the host's system-wide CUDA_PATH for the rest of this shell.
          export CUDA_HOME="${cudaToolkitRoot}"
          export CUDA_PATH="$CUDA_HOME"

          # `rustup default stable` (run by nix_dev_setup.sh) installs its shims here.
          export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

          # Selects nix_dev_setup.sh's install path for THIS shell. An outer
          # PF_RL from the calling environment is deliberately overridden --
          # the whole point of picking a named shell is to pin this.
          export PF_RL=${pfRl}

          echo "panofabric-engine dev shell [${label}] (nix-ld: ''${NIX_LD:+enabled}, PF_RL=$PF_RL)"
          echo "  uv      $(uv --version 2>/dev/null)"
          echo "  protoc  $(protoc --version 2>/dev/null)"
          echo "  gcc     $(gcc --version 2>/dev/null | head -1)"
          echo "  rustup  $(rustup --version 2>/dev/null | head -1)"
          echo "  venv    $UV_PROJECT_ENVIRONMENT"
          echo

          # Drives the venv/torch/forks setup every time the shell is entered; it is
          # a fast no-op once scripts/pf_env_fp.sh's fingerprint stops changing. Set
          # PF_NIX_SKIP_SETUP=1 for a bare toolchain shell with none of that (then run
          # scripts/nix_dev_setup.sh by hand whenever you do want it).
          if [ "''${PF_NIX_SKIP_SETUP:-0}" != "1" ] && [ -x ./scripts/nix_dev_setup.sh ]; then
            ./scripts/nix_dev_setup.sh
            ${postSetup}
          fi
        '';

        # Light/serving-only path: a plain, unpinned vllm so
        # panoengine.serve.engine_stage's real vLLM engine runs. Deliberately
        # NOT the matched RL pins (Makefile's install-rl) -- it pulls vllm's
        # own STABLE torch and clobbers the backend-matched nightly
        # nix_dev_setup.sh just installed. Fine for a serving-only sandbox;
        # don't also use this venv for torchtitan/torchft training or the RL
        # runtime. Guarded on vllm already importing, so re-entering the
        # shell is a fast no-op.
        servingVllmPostSetup = ''
          if [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] && ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import vllm" >/dev/null 2>&1; then
            echo "[serving-shell] installing vllm (stable, unpinned; replaces the nightly torch)"
            VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install vllm
          fi
        '';
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = commonPackages;
            shellHook = mkShellHook { pfRl = "0"; label = "default"; };
          };

          serving = pkgs.mkShell {
            packages = commonPackages;
            shellHook = mkShellHook {
              pfRl = "0";
              label = "serving";
              postSetup = servingVllmPostSetup;
            };
          };

          rl = pkgs.mkShell {
            packages = commonPackages;
            shellHook = mkShellHook { pfRl = "1"; label = "rl"; };
          };
        };
      }
    );
}
