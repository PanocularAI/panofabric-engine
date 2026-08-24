# What DiLoCo, HeLoCo and async actually do

[`docs/distributed.md`](distributed.md) covers *how to launch* decentralized training. This
page is about *which algorithm you are launching* and when each one is the right choice.

## The shared idea

All of these are answers to one problem: gradient synchronization assumes a fast, uniform
interconnect. Across datacenters — or across two HPC sites, or two boxes on home internet —
you have neither. Synchronizing every step would spend all its time waiting on the network.

So instead of exchanging **gradients every step**, islands train independently for `H` steps
and then exchange **pseudo-gradients**: the difference between where an island started the
window and where it ended up. One exchange per `H` steps instead of per step, which is
roughly an `H`-fold reduction in synchronization traffic.

An **island** here is one replica: a group of GPUs, possibly spanning several nodes, that
trains together with ordinary FSDP/TP inside. Islands talk to each other over the WAN via
Gloo on CPU; NCCL stays inside an island.

## Choosing one

| | What it does | Use it when |
|---|---|---|
| **solo** | No inter-island sync. Ordinary single-replica training. | One island. Baselines and debugging. |
| **DiLoCo** | All islands sync as a group at each window boundary, all-reduce style. | Islands are **similar in size and speed**. |
| **HeLoCo** | Islands push pseudo-gradients to a **parameter server** and pull back the global model, independently of one another. | Islands **differ** — 4×H100 alongside 8×A100 — or membership changes. |
| **async** | Like HeLoCo, but islands never block on a window boundary at all; the parameter server applies updates as they arrive. | Stragglers or unreliable links dominate, and you will trade some convergence quality for throughput. |

The practical dividing line is **DiLoCo vs HeLoCo**: DiLoCo is a collective, so the slowest
island sets the pace for everyone and every island must be present. HeLoCo replaces the
collective with a server, so a small island and a large one each proceed at their own rate.
That is what makes genuinely heterogeneous training work, and it is also why HeLoCo needs one
extra process (the parameter server) that DiLoCo does not.

## Fragments

A model can be split into **fragments** that sync on a staggered schedule instead of all at
once, so communication for one fragment overlaps computation on the next (the "streaming" in
Streaming DiLoCo). Two rules:

* `sync_steps` must be divisible by `num_fragments` — `tests/test_config_registry.py`
  enforces this for every preset.
* `num_fragments` must be **1** for a model whose spec has no `fragment_fn` (nothing can split
  it, so the whole model syncs as one unit). `hf_transformers` and `resnet` are in this case.

Fragments also cut the parameter server's peak memory, since it holds one fragment in flight
rather than a whole model.

## Choosing at launch, not in the recipe

The strategy is **not** a property of a recipe. Every recipe in `panoengine/train/` carries the
same default (see [`panoengine/train/strategies.py`](../panoengine/train/strategies.py)) and the
launcher overrides it:

```bash
uv run ./run_train.sh ... --fault_tolerance.semi_sync_method=heloco
```

This is why pretraining, LoRA fine-tuning and RL all get every strategy without any of them
implementing it: they are all `FaultTolerantTrainer.Config`, and the strategy is an argv value.

Useful knobs on the same config node:

| Flag | Meaning |
|---|---|
| `--fault_tolerance.sync_steps` | Window length `H`: local steps between exchanges. Larger = less traffic, more drift. |
| `--fault_tolerance.num_fragments` | How many pieces to stagger (see above). |
| `--fault_tolerance.fragment_sync_delay` | Inner steps to wait before blocking on a fragment's sync (the "tau" of the Streaming DiLoCo paper). Improves overlap, costs quality. |
| `--fault_tolerance.fragment_update_alpha` | Mixes local and global parameters after a sync. 0.0 = take the global parameters, keeping replicas identical. |
| `--fault_tolerance.rank0_synchronization_only` | Only rank 0 of a replica joins the exchange. Required for islands of differing size. |

## Where the implementations live

These algorithms are **not** in this repo. They subclass upstream internals, so they live in
the forks:

* [`torchft/heloco.py`](https://github.com/PanocularAI/torchft/blob/main/torchft/heloco.py),
  `async_diloco.py`, `semi_async_*` — plus the `manager.py` / `local_sgd.py` /
  `parameter_server.py` hooks they depend on.
* DiLoCo itself is upstream torchft, wired in through
  `torchtitan.experiments.torchft`.

What lives here is everything that only *composes* those public classes. See
[FORK-DELTA.md](../FORK-DELTA.md) for the full line.
