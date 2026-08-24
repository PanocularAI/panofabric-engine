# FORK-DELTA — the forks, and the line between them and this repo

This engine composes two forks:

* [PanocularAI/torchtitan](https://github.com/PanocularAI/torchtitan) — see its own
  [FORK-DELTA.md](https://github.com/PanocularAI/torchtitan/blob/main/FORK-DELTA.md).
  34 files, +8032 / −119 vs upstream.
* [PanocularAI/torchft](https://github.com/PanocularAI/torchft) — see its own
  [FORK-DELTA.md](https://github.com/PanocularAI/torchft/blob/main/FORK-DELTA.md).
  15 files, +7440 / −14 vs upstream.

Both are pinned by SHA from this repo's `[train]` extra. Never by branch: a moving ref in a
published dist is not reproducible, and it breaks outright when the branch is deleted — which
is what happened to the old `@async_rl` and `@async-diloco` refs.

## The line: recipes compose, implementations subclass

| Lives in the forks | Lives in `panoengine/` |
|---|---|
| Code that must **subclass or patch upstream internals**: `experiments/decentralized_rl/` (extends `experiments.rl`'s `PolicyTrainer`), `torchft.heloco`, `torchft.async_diloco`, the `manager.py` / `local_sgd.py` / `parameter_server.py` hooks | Code that only **composes public classes**: every recipe, every preset, the strategy defaults, dataloaders, the serving plane |
| Breaks when upstream moves an *internal* | Breaks only when upstream changes a *public signature* |
| ~15k lines across the two forks | ~1.1k lines, and the place all new work lands |

This is a real line, not a fudge. `panoengine/train/finetune/lora/config_registry.py` already
imports `torchtitan.experiments.torchft.{checkpoint,optimizer,trainer,diloco}` and composes
them from here. Anything that can be built that way belongs here.

## Why the forks are permanent, and why that is cheap

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost is the **119 + 14 deleted lines across 21 modified files** — not the ~15,000
added ones. The new modules are free to keep in-tree forever.

Which means the maintenance lever is not "get out of the fork business". It is "shrink the
diff on modified files". Each fork's FORK-DELTA.md lists its upstreaming candidates and its
target: **21 modified files → 2–3.**

Nothing here blocks on an upstream PR landing. The forks already carry every patch and keep
carrying it; each merge is a free deletion whenever it happens.

## Known friction

torchft builds its Rust extension via maturin, so a source install needs a Rust toolchain,
`protoc` 32.0, and CPython ≤ 3.13 (pyo3 0.24's ceiling — this repo's `requires-python`
declares that window so it fails at resolve time rather than mid-build). That is the single
biggest install-friction item in the stack. Publishing per-(CPython, platform) torchft wheels
on tag would remove Rust from every consumer, including the cloud-node bootstrap, and is
worth more to an outside developer than any of the upstream PRs.
