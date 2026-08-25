# FORK-DELTA — the forks, and the line between them and this repo

This engine composes two forks:

* [PanocularAI/torchtitan](https://github.com/PanocularAI/torchtitan) — see its own
  [FORK-DELTA.md](https://github.com/PanocularAI/torchtitan/blob/main/FORK-DELTA.md).
  17 files, +756 / −114 vs upstream — 16 of them modified upstream files, the 17th being its
  own FORK-DELTA.md. The fork adds no packages any more: the parameter server, relay, rollout
  queue, the RL presets, all of `decentralized_rl` and the HF-backend RL glue moved into this
  repo (it was +8032 before that).
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
| Code that must **patch upstream internals in place**: `torchft.semi_async_*` (extends the private `_StreamingDiLoCoFragment`) and the additive `Manager` / `local_sgd` patches — you cannot add methods to a class from outside it. Plus entry-point shims for paths stored specs name | Code that stands on its own, **composes public classes, or merely subclasses them**: **async DiLoCo and HeLoCo** (`panoengine.decentralized`), **the RL coordinators** (`panoengine.train.rl`), every recipe, every preset, the strategy defaults, dataloaders, the serving plane |
| Breaks when upstream moves an *internal* | Breaks only when upstream changes a *public signature* |
| ~3.1k lines across the two forks | ~10k lines, and the place all new work lands |

This is a real line, not a fudge. `models/lora/config_registry.py` already
imports `torchtitan.experiments.torchft.{checkpoint,optimizer,trainer,diloco}` and composes
them from here. Anything that can be built that way belongs here.

`panoengine.decentralized` is the line applied in the other direction: async DiLoCo and HeLoCo
were leaf modules in the torchft fork — no torchtitan, subclassing nothing, and nothing in
torchft imported them back — so they belong here, and they moved. They depend on torch and
torchft only, which is why the `[decentralized]` extra pulls no torchtitan: the forks' RL
adapter imports this package, and a torchtitan-free extra keeps that from being a cycle.

## The RL coordinators moved here; `experiments/rl` did NOT

Three more things followed them out of the fork, so that it now adds **no packages at all**:

* Nothing under a `decentralized_rl/` name. Stored run specs still say `--module
  decentralized_rl`, but that is a control-plane value, never an importable module: controld
  maps it to `panoengine.train.rl` when it builds the argv (`spec/runspec.py`'s
  `engine_module()`). The vocabulary the control plane invented stays its problem, and this
  repo ships only real modules.
* `panoengine/train/rl/hf_model_registry.py` — the HF-backend RL `model_registry`, previously
  `torchtitan/experiments/transformers_modeling_backend/rl/`. It only composes that backend's
  public pieces, and nothing in the fork imported it.
* The `FT_ENABLE` launch wrapper in `run_train.sh` — it builds torchtitan's own
  `--fault_tolerance.*` flags, nothing fork-specific, so the fork's `run_train.sh` is now
  byte-identical to upstream.

Because the control plane now launches `panoengine.*` entry points directly rather than through
fork shims, **controld and the engine image must be rolled together**.


The RL **coordinators** — the Monarch trainer actors, the four replica strategies, the
controller mixin and the launch entry points — moved out of the torchtitan fork into
`panoengine/train/rl/` (4,635 lines), flat beside the presets that name them. The RL **implementation** they build on
stays in the fork, at `torchtitan/experiments/rl/`, and is imported from there.

That implementation was briefly vendored into this repo as an editable copy. The copy was
deleted: an exact duplicate of a directory the fork already carries is pure maintenance debt —
every upstream rebase lands improvements in one copy and not the other, and the fork's own
modifications to `experiments/rl` (six files, +280/−80) would have to be kept in sync by hand.

Subclassing across the package boundary is fine, and is all these classes need. Measured on the
current code, `RLTrainer`, `DiLoCoManagerTrainer`, `HeLoCoPolicyTrainer` and
`SnapshotPolicyTrainer` override **zero** base methods between them — every method they define
is an addition. They want `PolicyTrainer` and `Controller` as bases, not as things to patch,
and an ordinary cross-package subclass gives them that. What genuinely cannot leave a fork is
patching a class *in place*, which is what `torchft.semi_async_*` and the `Manager` /
`local_sgd` patches do.

**The coupling this creates, stated plainly.** `panoengine.train.rl` depends on
`torchtitan.experiments.rl` — upstream's own *experimental* directory, the least stable API
surface in the stack. When it drifts, the breakage surfaces here rather than in the fork, where
upstream could also be patched in place. It is bounded by the SHA pin in this repo's `[train]`
extra: nothing moves until someone bumps it. If that drift gets expensive, moving these
coordinators back into the fork is the escape hatch.

The fork keeps no shims for any of this. controld names the engine's own paths —
`panoengine.train.rl.{train,worker}` and `panoengine.decentralized.{parameter_server,relay,
rollout_queue}` — and translates the legacy `--module decentralized_rl` spec value itself. The
consequence is a deploy ordering constraint: **controld and the engine image must roll
together**, since a controld naming those paths cannot launch against an older image.

`panoengine.train.rl` must never be imported by the CPU-only processes (parameter server,
relay, rollout queue): importing any coordinator pulls the RL stack and vLLM with it, which on
a CPU node is seconds of import cost and a crash risk. Those three live in
`panoengine.decentralized`, which needs neither torchtitan nor vLLM.
`test_package_import_stays_cpu_light` guards that.

## Why the forks are permanent, and why that is cheap

A rebase conflicts only on files **both sides touched**. Added files never conflict. So the
carrying cost is the **114 + 14 deleted lines across 19 modified files** — not the ~3,100
added ones. The new modules are free to keep in-tree forever.

Which means the maintenance lever is not "get out of the fork business". It is "shrink the
diff on modified files". Each fork's FORK-DELTA.md lists its upstreaming candidates and its
target: **19 modified files → 2–3.**

Nothing here blocks on an upstream PR landing. The forks already carry every patch and keep
carrying it; each merge is a free deletion whenever it happens.

## Known friction

torchft builds its Rust extension via maturin, so a source install needs a Rust toolchain,
`protoc` 32.0, and CPython ≤ 3.13 (pyo3 0.24's ceiling — this repo's `requires-python`
declares that window so it fails at resolve time rather than mid-build). That is the single
biggest install-friction item in the stack. Publishing per-(CPython, platform) torchft wheels
on tag would remove Rust from every consumer, including the cloud-node bootstrap, and is
worth more to an outside developer than any of the upstream PRs.
