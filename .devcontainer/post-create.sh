#!/usr/bin/env bash
set -euo pipefail

# The named volume mounted over ~/.local keeps whatever UID first populated
# it; the devcontainer CLI may since have remapped vscode's UID to match the
# host user's.
sudo chown "$(id -u):$(id -g)" "$HOME/.local"

# Make sure the driver is linked before the first shell entry below, which runs
# scripts/pf_backend.sh to pick the torch backend.
link-nvidia-driver

# .envrc picks the devShell (default, or whatever a gitignored .envrc.local
# says -- e.g. `use flake .#rl`). Not chosen automatically here: the workspace
# is a bind mount, so an .envrc.local written here would also switch the shell
# on the host checkout.
direnv allow
# Build the devShell -- and, via its shellHook, the venv/torch/forks setup --
# now rather than on first editor/terminal use, where the extensions would
# time out waiting on it.
direnv exec . true
