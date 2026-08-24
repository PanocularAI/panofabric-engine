# FORK-DELTA — the forks, and the line between them and this repo

This engine composes two forks:

* [PanocularAI/torchtitan](https://github.com/PanocularAI/torchtitan) — see its own
  [FORK-DELTA.md](https://github.com/PanocularAI/torchtitan/blob/main/FORK-DELTA.md).
  34 files, +5561 / −119 vs upstream (it was +8032 before the parameter server, relay,
  rollout queue and RL presets moved into this repo).
* [PanocularAI/torchft](https://github.com/PanocularAI/torchft) — see its own
  [FORK-DELTA.md](https://github.com/PanocularAI/torchft/blob/main/FORK-DELTA.md).
  10 files, +2368 / −14 vs upstream (it was 15 files / +7440 before the async DiLoCo and
  HeLoCo algorithms moved into this repo).

Both are pinned by SHA from this repo's `[train]` extra. Never by branch: a moving ref in a
published dist is not reproducible, and it breaks outright when the branch is deleted — which
is what happened to the old `@async_rl` and `@async-diloco` refs.

## The line: recipes compose, implementations subclass

| Lives in the forks | Lives in `panoengine/` |
|---|---|
| Code that must **subclass or patch upstream internals**: the RL trainers in `experiments/decentralized_rl/` (extend `experiments.rl`'s `PolicyTrainer`), `torchft.semi_async_*` (extends the private `_StreamingDiLoCoFragment`), and the additive `Manager` / `local_sgd` patches — you cannot add methods to a class from outside it | Code that stands on its own or only **composes public classes**: **async DiLoCo and HeLoCo** (`panoengine.decentralized`), every recipe, every preset, the strategy defaults, dataloaders, the serving plane |
| Breaks when upstream moves an *internal* | Breaks only when upstream changes a *public signature* |
| ~7.9k lines across the two forks | ~7.2k lines, and the place all new work lands |

This is a real line, not a fudge. `panoengine/train/finetune/lora/config_registry.py` already
imports `torchtitan.experiments.torchft.{checkpoint,optimizer,trainer,diloco}` and composes
them from here. Anything that can be built that way belongs here.

`panoengine.decentralized` is the line applied in the other direction: async DiLoCo and HeLoCo
were leaf modules in the torchft fork — no torchtitan, subclassing nothing, and nothing in
torchft imported them back — so they belong here, and they moved. They depend on torch and
torchft only, which is why the `[decentralized]` extra pulls no torchtitan: the forks' RL
adapter imports this package, and a torchtitan-free extra keeps that from being a cycle.

## Why the forks are permanent, and why that is cheap

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost is the **119 + 14 deleted lines across 21 modified files** — not the ~7,900
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
