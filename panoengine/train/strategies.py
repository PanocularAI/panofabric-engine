"""Decentralized-training strategy defaults, shared by the pretrain and finetune recipes.

Every recipe under ``models/`` returns a ``FaultTolerantTrainer.Config`` whose
fault-tolerance block is the one built below — so a recipe gets decentralized
training by writing ``fault_tolerance=semi_sync()`` and nothing else.

NOT used by RL. ``panoengine.train.rl`` reaches decentralized training through a
different mechanism entirely: its four replica classes in
``panoengine.train.rl.replicas`` (DiLoCo / HeLoCo / async-inference / both),
selected by preset rather than by a ``semi_sync_method`` flag. Nothing under
``panoengine/train/rl/`` imports this module.

WHERE THE STRATEGY IS ACTUALLY CHOSEN — two answers, depending on who launches:

    Standalone (``./run_train.sh``, ``python -m torchtitan.train``): whatever
    the preset built here. ``semi_sync(method="heloco")`` really does run
    HeLoCo — but see the function docstring, that path also needs a parameter
    server process and its URLs in the environment.

    Under controld: the preset's value is only a default. ``spec/launch.py``
    always emits an explicit

        --fault_tolerance.semi_sync_method=<diloco | local_sgd | heloco>

    derived from ``workload.strategy``, and tyro applies it on top. controld
    also stands up whatever plane that strategy needs — for heloco that is
    ``HeLoCoPretrainHub``: a parameter server AND a lighthouse (the trainer
    derives its dataloader shard from the FT manager, so with fault tolerance
    off every island would train identical data).

    That is why every preset in this repo calls a bare ``semi_sync()``. There is
    ONE default here, not one factory per strategy, because a factory per
    strategy would suggest the preset decides — and in the deployment that
    matters, it does not.

WHERE EACH VALUE ACTUALLY RESOLVES — the torchtitan fork's
``experiments/torchft/manager.py`` branches on the final ``semi_sync_method``:

    "diloco"     -> ``torchft.local_sgd.DiLoCo``    (upstream torchft)
    "local_sgd"  -> ``torchft.local_sgd.LocalSGD``  (upstream torchft)
    "heloco"     -> ``panoengine.decentralized.async_diloco.AsyncDiLoCo``

That last one is not a typo. On the TRAINER side both parameter-server strategies
are the same class: ``AsyncDiLoCo`` POSTs its pseudo-gradient to the PS and pulls
back look-ahead global params. What makes a run HeLoCo rather than plain async
DiLoCo is SERVER-side — ``panoengine.decentralized.heloco``'s ``HeLoCoOptimizer``
(look-ahead init, tensor-block directional correction), which
``panoengine.decentralized.parameter_server`` runs.

So the algorithms live HERE, in ``panoengine.decentralized``, not in the forks.
What stays in the torchft fork is only what patches upstream internals in place:
``torchft.semi_async_diloco`` / ``torchft.semi_async_heloco`` (they extend the
private ``_StreamingDiLoCoFragment``) and the additive ``manager.py`` /
``local_sgd.py`` hooks. See FORK-DELTA.md.
"""

import sys

from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.optimizer import default_ft_adamw

__all__ = ["semi_sync", "adamw"]

#: The flag controld emits for a CENTRALIZED run. It emits the positive
#: `--fault_tolerance.enable` otherwise -- always one or the other, never
#: omission, because every preset here hardcodes enable=True and an omitted flag
#: would silently keep fault tolerance on. See ResolvedIsland.ft_flags().
_FT_DISABLED_FLAG = "--fault_tolerance.no_enable"

#: The values torchtitan's `maybe_semi_sync_training` dispatches on. "solo" is
#: NOT one of them — controld's strategy axis has it, and it means "leave
#: fault_tolerance off entirely", not "pass semi_sync_method=solo".
SEMI_SYNC_METHODS = ("diloco", "local_sgd", "heloco")


def semi_sync(
    *,
    method: str = "diloco",
    num_fragments: int = 2,
    **overrides,
) -> FaultTolerance:
    """The FT block every recipe under ``models/`` shares.

    ``method`` picks the algorithm. It is a real parameter — a preset CAN pin a
    strategy, and a standalone launch (``./run_train.sh``, ``python -m
    torchtitan.train``) will run whatever it says. Under controld it is only a
    default: the launcher always emits an explicit
    ``--fault_tolerance.semi_sync_method=...`` derived from ``workload.strategy``,
    and tyro applies that on top. Which is why every preset in this repo just
    calls ``semi_sync()`` and lets the launcher decide.

    Picking ``"heloco"`` here is not sufficient on its own: that path POSTs
    pseudo-gradients to a parameter server, so it also needs
    ``panoengine.decentralized.parameter_server`` running and its URLs exported
    as ``$DILOCO_SERVER_ADDR`` / ``$DILOCO_HB_ADDR``. controld's
    ``HeLoCoPretrainHub`` does that for you; standalone, you do it yourself. The
    trainer fails loudly with the missing env var if you forget.

    ``num_fragments`` is the one field recipes genuinely differ on: it must be 1
    for a model whose spec has no ``fragment_fn`` (nothing can split it, so the
    whole model syncs), and the DiLoCo invariant is that ``sync_steps`` stays
    divisible by it — tests/test_config_registry.py enforces that. It must also
    match the parameter server's ``--num_fragments`` on the heloco path.

    ``process_group="gloo"`` because the pseudo-gradient exchange runs over the
    WAN between islands on CPU, not over NCCL inside one.

    Anything else in ``FaultTolerance`` can be set through ``**overrides``.
    """
    if method not in SEMI_SYNC_METHODS:
        raise ValueError(
            f"semi_sync_method must be one of {SEMI_SYNC_METHODS}, got {method!r}"
        )
    # The algorithm has exactly one spelling here: the `method` parameter. Its
    # FaultTolerance field name would otherwise slip through **overrides and win
    # the dict update below, and a cluster quietly running the other algorithm is
    # an expensive way to discover that. (`num_fragments` needs no such guard —
    # it is a named parameter, so Python binds it before **overrides sees it.)
    if "semi_sync_method" in overrides:
        raise TypeError(
            "pass the strategy as the named parameter `method=`, not as "
            "semi_sync_method= in **overrides"
        )
    fields = dict(
        enable=True,
        sync_steps=10,
        num_fragments=num_fragments,
        semi_sync_method=method,
        process_group="gloo",
        process_group_timeout_ms=10000,
    )
    fields.update(overrides)   # every default above is overridable
    return FaultTolerance(**fields)


def adamw(lr: float = 8e-4, **kwargs) -> OptimizersContainer.Config:
    """The optimizer every recipe under ``models/`` shares -- FT-aware.

    ``default_ft_adamw`` builds a ``TorchFTOptimizersContainer``, and the FT trainer
    picks that container purely from the config TYPE
    (``experiments/torchft/trainer.py``: ``isinstance(config.optimizer,
    TorchFTOptimizersContainer.Config)``). Its ``__init__`` then reads
    ``ft_manager.manager``, which ASSERTS when fault tolerance is off -- so a preset
    that hardcodes the FT optimizer cannot run centralized at all. A centralized
    island should be plain torchtitan: no manager, no torchft optimizer wrapper.

    The preset cannot be handed that decision as a parameter: ``ConfigManager``
    strips only ``--module``/``--config``, calls this registry function with no
    arguments, and lets tyro apply the remaining CLI on top of what it returns. So
    the flag is read here from the very argv tyro is about to parse. That is a real
    contract, not a guess -- controld always emits one of the two spellings.

    Standalone (``./run_train.sh`` with no FT flags) keeps the FT optimizer, which
    is what the presets' ``semi_sync()`` block expects.
    """
    if _FT_DISABLED_FLAG in sys.argv:
        return default_adamw(lr, **kwargs)
    return default_ft_adamw(lr, **kwargs)
