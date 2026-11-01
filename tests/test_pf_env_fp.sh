#!/bin/sh
# Regression test for scripts/pf_env_fp.sh (POSIX sh, no framework, no GPU).
#
# The env-cache key MUST track the fork trees that actually get installed
# ($(FORKS_DIR)/torchft, ../torchft by default) and must NOT track stale in-repo copies.
# Getting this backwards cost a live debug on 2026-08-28: a real fix to the fork left the
# key unchanged so `ensure` reused a venv built from the old code, while an edit to a
# leftover in-repo copy forced a pointless multi-GB rebuild.
set -eu
script=$(cd "$(dirname "$0")/.." && pwd)/scripts/pf_env_fp.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

# Minimal engine layout: the repo, plus forks as SIBLINGS (FORKS_DIR defaults to ..).
mkdir -p "$tmp/engine/models" "$tmp/engine/panoengine/decentralized" \
	"$tmp/torchtitan" "$tmp/torchft/torchft"
echo "print('model')"      > "$tmp/engine/models/llama.py"
echo "worker_count"        > "$tmp/engine/panoengine/decentralized/async_diloco.py"
printf 'deps\n'            > "$tmp/engine/pyproject.toml"
printf 'lock\n'            > "$tmp/engine/uv.lock"
printf 'TORCHFT_REF ?= aaa\n' > "$tmp/engine/Makefile"
echo "print('titan')"      > "$tmp/torchtitan/train.py"
echo "AF_INET6"            > "$tmp/torchft/torchft/http.py"

fp() { (cd "$tmp/engine" && sh "$script" --source); }

check_ne() {  # $1 label, $2 before, $3 after
	if [ "$2" != "$3" ]; then echo "ok   $1 (key changed)"; else
		echo "FAIL $1: key did NOT change ($2)"; fails=$((fails + 1)); fi
}
check_eq() {
	if [ "$2" = "$3" ]; then echo "ok   $1 (key stable)"; else
		echo "FAIL $1: key changed ($2 -> $3)"; fails=$((fails + 1)); fi
}

base=$(fp)
[ -n "$base" ] || { echo "FAIL fingerprint is empty"; exit 1; }

# THE regression: editing the fork that gets installed must invalidate the cache.
echo "AF_INET fallback" > "$tmp/torchft/torchft/http.py"
check_ne "sibling fork edit" "$base" "$(fp)"
echo "AF_INET6" > "$tmp/torchft/torchft/http.py"
check_eq "sibling fork restored" "$base" "$(fp)"

# The other direction: a stale in-repo copy is NOT what gets installed, so it must not
# invalidate a perfectly good cache entry.
mkdir -p "$tmp/engine/torchft/torchft"
echo "stale leftover" > "$tmp/engine/torchft/torchft/http.py"
check_eq "stale in-repo copy ignored" "$base" "$(fp)"
rm -rf "$tmp/engine/torchft"

# The pins live in the Makefile, which IS hashed -- that is what covers a fresh node
# where the forks are cloned during the build rather than synced.
printf 'TORCHFT_REF ?= bbb\n' > "$tmp/engine/Makefile"
check_ne "fork pin bump" "$base" "$(fp)"
printf 'TORCHFT_REF ?= aaa\n' > "$tmp/engine/Makefile"

# Absent forks still yield a usable key (the node's pre-clone case), not an empty string.
mv "$tmp/torchft" "$tmp/torchft.away"
noforks=$(fp)
[ -n "$noforks" ] && echo "ok   absent forks still fingerprint" || {
	echo "FAIL absent forks produced an empty fingerprint"; fails=$((fails + 1)); }
mv "$tmp/torchft.away" "$tmp/torchft"

# Derived junk must never move the key (build/ left by a local install, __pycache__, ...).
mkdir -p "$tmp/torchft/build" "$tmp/torchft/torchft/__pycache__"
echo "junk" > "$tmp/torchft/build/artifact.py"
echo "junk" > "$tmp/torchft/torchft/__pycache__/http.cpython-313.pyc"
check_eq "derived trees pruned" "$base" "$(fp)"

# panoengine is INSTALLED into the cached venv, non-editably, so a panoengine-only edit
# that leaves the key unchanged silently runs stale code on every cluster. Live on
# 2026-08-30: a cohort-gate fix in panoengine/decentralized/async_diloco.py reached the
# node via workdir sync, but `ensure` reported `HIT b31bb514f0477f4b` -- the same key as
# before the edit -- and the installed copy had none of the new code. The run deadlocked
# on exactly the bug the fix removed.
prev=$(fp)
echo "worker_count + finished_count" > "$tmp/engine/panoengine/decentralized/async_diloco.py"
check_ne "panoengine edit" "$prev" "$(fp)"
echo "worker_count" > "$tmp/engine/panoengine/decentralized/async_diloco.py"

# --editable (nix_dev_setup.sh's key): engine and forks are installed editable, so a
# source edit is live and must NOT rebuild the env -- it did, uninstalling ~260
# packages per .py edit, and per checkpoint downloaded into a fork (merges.txt).
# What changes the INSTALL still must: manifests, the lock, pins, compiled sources.
printf '[project]\nname = "torchft"\n' > "$tmp/torchft/pyproject.toml"
mkdir -p "$tmp/torchft/src" "$tmp/torchtitan/torchtitan/rl/example_checkpoint/Qwen3-0.6B"
echo "fn main() {}" > "$tmp/torchft/src/lib.rs"
printf 'torch\n' > "$tmp/torchtitan/requirements.txt"
efp() { (cd "$tmp/engine" && sh "$script" --editable); }
ebase=$(efp)
[ -n "$ebase" ] || { echo "FAIL editable fingerprint is empty"; exit 1; }
check_ne "editable key differs from source key" "$(fp)" "$ebase"

echo "edited" > "$tmp/engine/panoengine/decentralized/async_diloco.py"
echo "AF_INET fallback" > "$tmp/torchft/torchft/http.py"
echo "print('titan v2')" > "$tmp/torchtitan/train.py"
echo "merges" > "$tmp/torchtitan/torchtitan/rl/example_checkpoint/Qwen3-0.6B/merges.txt"
check_eq "editable: .py edits and a downloaded checkpoint ignored" "$ebase" "$(efp)"

echo "fn main() { changed() }" > "$tmp/torchft/src/lib.rs"
check_ne "editable: compiled Rust source edit" "$ebase" "$(efp)"
echo "fn main() {}" > "$tmp/torchft/src/lib.rs"
printf '[project]\nname = "torchft"\ndependencies = ["x"]\n' > "$tmp/torchft/pyproject.toml"
check_ne "editable: fork pyproject edit" "$ebase" "$(efp)"
printf '[project]\nname = "torchft"\n' > "$tmp/torchft/pyproject.toml"
printf 'torch\nnumpy\n' > "$tmp/torchtitan/requirements.txt"
check_ne "editable: fork requirements edit" "$ebase" "$(efp)"
printf 'torch\n' > "$tmp/torchtitan/requirements.txt"
printf 'lock2\n' > "$tmp/engine/uv.lock"
check_ne "editable: engine lockfile edit" "$ebase" "$(efp)"
printf 'lock\n' > "$tmp/engine/uv.lock"
printf 'TORCHFT_REF ?= bbb\n' > "$tmp/engine/Makefile"
check_ne "editable: fork pin bump" "$ebase" "$(efp)"
printf 'TORCHFT_REF ?= aaa\n' > "$tmp/engine/Makefile"
check_eq "editable: restored" "$ebase" "$(efp)"

[ "$fails" -eq 0 ] || exit 1
echo "all pf_env_fp checks passed"
