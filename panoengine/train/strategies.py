"""Decentralized-training strategy defaults, shared by every recipe.

This is why "pretraining, fine-tuning and RL all get decentralized training for
free" is a fact about the code rather than a claim in the README: every recipe
returns a ``FaultTolerantTrainer.Config``, and the fault-tolerance block below is
the same for all of them.

WHERE THE STRATEGY IS ACTUALLY CHOSEN — read this before adding a
``heloco()`` / ``async_diloco()`` sibling to ``diloco()``:

    The strategy is NOT fixed when a preset is constructed. Presets carry a
    default, and the launcher overrides it at runtime with a tyro flag:

        --fault_tolerance.semi_sync_method=heloco

    (panofabric ``spec/launch.py`` emits that from ``workload.strategy``.) So
    solo / diloco / heloco / async are argv values, not constructors, and a set
    of preset-time factories here would imply a selection mechanism that does
    not exist. One shared default is the honest shape.

The algorithms themselves live in the forks, because they subclass upstream
internals: ``torchft.heloco``, ``torchft.async_diloco``, and the ``manager.py`` /
``local_sgd.py`` hooks they depend on.
"""

from torchtitan.experiments.torchft.config.job_config import FaultTolerance

__all__ = ["diloco"]


def diloco(*, num_fragments: int = 2, **overrides) -> FaultTolerance:
    """The FT block every recipe in this repo shares.

    ``num_fragments`` is the one field recipes genuinely differ on: it must be 1
    for a model whose spec has no ``fragment_fn`` (nothing can split it, so
    DiLoCo syncs the whole model), and the DiLoCo invariant is that
    ``sync_steps`` stays divisible by it — tests/test_config_registry.py
    enforces that.

    ``process_group="gloo"`` because the pseudo-gradient exchange runs over the
    WAN between islands on CPU, not over NCCL inside one.
    """
    return FaultTolerance(
        enable=True,
        sync_steps=10,
        num_fragments=num_fragments,
        semi_sync_method="diloco",
        process_group="gloo",
        process_group_timeout_ms=10000,
        **overrides,
    )
