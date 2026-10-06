# Copyright (c) Panocular AI.
#
# Config entry points for the decentralized_rl coordination strategies, discoverable
# by torchtitan's ConfigManager via ``--module decentralized_rl --config
# <function_name>`` (or the fully-qualified ``--module
# panoengine.train.rl``).
#
# ``base_rl_config`` + ``wrap_replica`` are the shared building blocks: a plain
# RLTrainer.Config with the common loss/data choices, and a helper that
# copies it into one of the coordinator Configs (each a strict superset of
# RLTrainer.Config) plus the coordinator-specific extras. The ``rl_*`` entry
# points are generated at the bottom of this file, one per strategy x model
# (x task) -- see "Presets" there.
#
# Adding a new RL model: add it to ``_MODEL_REGISTRY_BY_MODEL`` and
# ``_RENDERER_BY_MODEL`` below, and its checkpoint to ``_DEFAULT_HF_ASSETS_PATH``
# (which is also what gives it named presets). GPU count is not fixed either:
# trainer/generator tensor_parallel_degree are real parameters here (default
# 1), and num_replicas / GPUS_PER_REPLICA (launch script arg) flow through
# independently.
#
# ConfigManager calls the ``--config`` function with NO arguments (CLI flags
# then overlay onto the resulting dataclass's fields), so a size/flavor/model
# switch needs its own named entry point to be reachable from the CLI.

import dataclasses
import functools
import inspect
import os

from renderers import DefaultRendererConfig, Qwen35RendererConfig, Qwen3RendererConfig

from torchtitan import rl as _rl_pkg
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import (
    AdamW,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import CompileConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from panoengine.train.rl.controller import RLTrainer
from panoengine.train.rl.replicas import (
    AsyncInferenceReplica,
    DiLoCoRLReplica,
    HeLoCoAsyncInferenceReplica,
    HeLoCoRLReplica,
    SoloRLReplica,
)
from panoengine.train.rl.worker import AsyncInferenceWorker
from torchtitan.rl.generator import SamplingConfig, VLLMGenerator
from torchtitan.rl.trainer import Trainer
from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.controller import AsyncLoopConfig, ValidationConfig
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.losses import DAPOLoss, GRPOLoss
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import model_registry as _llama3_model_registry
from torchtitan.models.qwen3 import model_registry as _qwen3_model_registry


def _qwen3_5_model_registry(flavor: str, *, seq_len: int, attn_backend: str):
    """Qwen3.5 through the engine's own ``models.qwen3_5``, not torchtitan's
    registry: that wrapper drops the vision tower (nothing to train or sync for
    a text task) and loads through the adapter that reads the published
    checkpoints' ``model.language_model.*`` keys -- torchtitan's adapter keys a
    text-only model as ``model.*`` and would load nothing. The lm_head runs in
    fp32 as in upstream's Qwen3.5 RL presets: GRPO's ratio compares trainer and
    generator logprobs. Lazy import: the wrapper pulls in the FT trainer stack."""
    from models.qwen3_5 import model_registry

    return model_registry(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=[LMHeadCastConverter.Config()],
    )


#: Entries share one signature: (flavor, *, seq_len, attn_backend).
_MODEL_REGISTRY_BY_MODEL = {
    "qwen3": _qwen3_model_registry,
    "llama3": _llama3_model_registry,
    "qwen3_5": _qwen3_5_model_registry,
}

#: Chat template per model (the `renderers` package). llama3 keeps the generic
#: "default" template it has always used here. Qwen3.5 needs its own: it emits
#: XML tool calls (``<function=f><parameter=k>v</parameter></function>``), which
#: the qwen3 renderer's JSON parser reads as ``invalid_json`` -- every correct
#: tool call would score as an error. (Upstream's Qwen3.5 RL presets use the
#: qwen3 renderer, which only works because their task has no tools.)
_RENDERER_BY_MODEL = {
    "qwen3": lambda: Qwen3RendererConfig(enable_thinking=False),
    "llama3": lambda: DefaultRendererConfig(),
    "qwen3_5": lambda: Qwen35RendererConfig(enable_thinking=False),
}

_EXAMPLE_CHECKPOINT_DIR = os.path.join(_rl_pkg.__path__[0], "example_checkpoint")

#: Default hf_assets_path per (model, flavor). "qwen3"/"0.6B" ships inside
#: torchtitan's rl package (no download needed); the rest are conventional
#: paths under the same example_checkpoint dir that the caller downloads a
#: checkpoint into, or pass hf_assets_path explicitly.
_DEFAULT_HF_ASSETS_PATH = {
    ("qwen3", "0.6B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-0.6B"),
    ("qwen3", "1.7B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-1.7B"),
    ("qwen3", "4B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-4B-Base"),
    ("llama3", "8B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Llama-3.1-8B"),
    ("qwen3_5", "0.8B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3.5-0.8B"),
    ("qwen3_5", "9B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3.5-9B"),
}

#: Packed training width (tokens per trainer microbatch = 2 x this) and the
#: model's context bound; alphabet-sort's prompt + 700-token response fits.
_SEQ_LEN = 2048


def _alphabet_sort_rollouter() -> Rollouter.Config:
    """The alphabet-sort example task (torchtitan's own rollouter wiring)."""
    from torchtitan.rl.examples.alphabet_sort.data import AlphabetSortDataset
    from torchtitan.rl.examples.alphabet_sort.env import AlphabetSortEnv
    from torchtitan.rl.examples.alphabet_sort.rubric import RewardAlphabetSort

    return Rollouter.Config(
        train_dataset=AlphabetSortDataset.Config(seed=42),
        validation_dataset=AlphabetSortDataset.Config(seed=99),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardAlphabetSort.Config(weight=1.0)]),
            message_env=AlphabetSortEnv.Config(),
        ),
    )


def _adamw(lr: float, **kwargs) -> OptimizersContainer.Config:
    return OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=lr, **kwargs)]
    )


def base_rl_config(
    hf_assets_path: str | None = None,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
    rollouter=None,
    seq_len: int = _SEQ_LEN,
) -> RLTrainer.Config:
    """Shared base RLTrainer config: stock GRPO loss, wandb off.

    ``model``/``flavor`` select the model spec (via ``_MODEL_REGISTRY_BY_MODEL``)
    and the default checkpoint path (via ``_DEFAULT_HF_ASSETS_PATH``);
    hf_assets_path overrides the default (e.g. for a fine-tuned checkpoint of
    the same flavor, or any (model, flavor) with no default).

    vLLM's KV-cache budget is deliberately NOT pinned here: it stays at
    VLLMGenerator.Config.gpu_memory_limit's own default. These presets used to
    force 0.35 so a generator would coexist with other tenants of a shared dev
    box, but every generator actor already owns a disjoint GPU slice, so on a
    provisioned island that only threw the rest of the card away -- measured on
    a decoupled H100 generator: 9.4 GiB used of 79, and vLLM capped at ~3.2
    concurrent full-length requests while ~68 GiB sat idle. It was also a
    preset KWARG, which ConfigManager cannot reach from argv, so no spec could
    raise it. Tune it per run instead: `--generator.gpu_memory_limit=0.35`
    (spec `overrides`), which is how a shared box should ask for less.
    trainer/generator tensor_parallel_degree default to 1; raise either for a
    model too large to fit one GPU per role -- the GPU mesh spawned by
    panoengine.train.rl.train (and therefore the GPU count a
    launch script needs to provision) scales with these automatically.
    ``rollouter`` is the task bundle (dataset + reward rubric + environment, a
    ``Rollouter.Config`` subclass instance); it defaults to the alphabet-sort
    example task and flows through ``wrap_replica`` unchanged, so a different
    task/reward needs no coordinator changes.
    """
    if model not in _MODEL_REGISTRY_BY_MODEL:
        raise ValueError(
            f"unknown RL model {model!r} (known: "
            f"{sorted(_MODEL_REGISTRY_BY_MODEL)}); add it to "
            "_MODEL_REGISTRY_BY_MODEL/_RENDERER_BY_MODEL in this file"
        )
    resolved_hf_assets_path = hf_assets_path or _DEFAULT_HF_ASSETS_PATH.get(
        (model, flavor)
    )
    if resolved_hf_assets_path is None:
        raise ValueError(
            f"no default hf_assets_path for {model} {flavor!r}; add one to "
            "_DEFAULT_HF_ASSETS_PATH in this file or pass hf_assets_path "
            "explicitly"
        )
    model_config = _MODEL_REGISTRY_BY_MODEL[model](
        flavor, seq_len=seq_len, attn_backend="varlen"
    )
    compile_config = CompileConfig(backend="aot_eager")
    loss = GRPOLoss.Config(global_vocab_size=decoder_vocab_size(model_config))
    if model == "qwen3_5":
        # Upstream's Qwen3.5 RL recipe: eager, and the loss chunked over the
        # sequence -- the fp32 lm_head over a 248K vocab is ~1 GB of logits per
        # 1K tokens otherwise.
        compile_config = None
        loss = ChunkedLossWrapper.Config(num_chunks=8, loss_fn=loss)
    return RLTrainer.Config(
        model=model_config,
        hf_assets_path=resolved_hf_assets_path,
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=8,
            # decentralized_rl replicas drive a SYNCHRONOUS windowed loop
            # (RLControllerMixin), so no off-policy buffering: target 0 sizes the
            # active buffer to exactly one step's groups, whose slots free only
            # after the post-step weight pull, so every group is generated under
            # the just-published policy. windowed_fifo_batches=1 is FIFO by batch
            # (None would be greedy, taking any finished group).
            target_offpolicy_steps=0,
            windowed_fifo_batches=1,
            batcher=Batcher.Config(),
            validation=ValidationConfig(num_samples=20),
        ),
        compile=compile_config,
        rollouter=rollouter if rollouter is not None else _alphabet_sort_rollouter(),
        renderer=from_renderers(_RENDERER_BY_MODEL[model]()),
        generator_router=InterGeneratorRouter.Config(
            strategy=RoundRobinRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=Trainer.Config(
            # Structured-logging JSONL traces are a debugging aid that costs
            # real disk (hundreds of MB/hour per actor); off by default here
            # since decentralized_rl runs are typically many-hour, many-actor swarms.
            debug=DebugConfig(enable_structured_logging=False),
            optimizer=_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                # Packed RL batches change their varlen metadata every step, which
                # CUDA-graph capture cannot replay (upstream's RL presets all
                # disable it too).
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=trainer_tensor_parallel_degree,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
                # A full checkpoint (fp32 params + AdamW state) is ~12 bytes per
                # parameter: ~108 GB at 9B. torchtitan's default keeps 10 of
                # them, which filled a 2 TB node within one long run. The newest
                # plus one fallback (in case a write is interrupted) is enough
                # to resume.
                keep_latest_k=2,
            ),
            loss=loss,
        ),
        generator=VLLMGenerator.Config(
            debug=DebugConfig(enable_structured_logging=False),
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=generator_tensor_parallel_degree,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def wrap_replica(cls, base: RLTrainer.Config, **kwargs):
    """Copy every field of a plain RLTrainer.Config ``base`` into ``cls.Config``
    (one of the coordinator Configs, each a strict superset of RLTrainer.Config),
    plus the coordinator-specific extras in ``kwargs`` (sync_every,
    num_replicas, should_quantize, ...). Avoids repeating the full field list
    in every config function.

    A coordinator Config may redeclare a base field with a different default
    (PureLearnerReplica: ``num_generators = 0``). For a field the caller left
    at the base default and did not pass in kwargs, the coordinator's
    redeclared default wins -- blindly copying the base's untouched value
    would silently defeat the redeclaration. A value the caller explicitly
    set on ``base`` still copies through (and still trips the coordinator's
    validation if incompatible)."""
    base_fields = {}
    base_defaults = {}
    for f in dataclasses.fields(base):
        base_fields[f.name] = getattr(base, f.name)
        base_defaults[f.name] = f.default
    for f in dataclasses.fields(cls.Config):
        if (
            f.name not in kwargs
            and f.name in base_fields
            and f.default is not dataclasses.MISSING
            and base_defaults[f.name] is not dataclasses.MISSING
            and f.default != base_defaults[f.name]
            and base_fields[f.name] == base_defaults[f.name]
        ):
            base_fields[f.name] = f.default
    base_fields.update(kwargs)
    return cls.Config(**base_fields)


# === Presets =================================================================
#
# A preset is named ``<strategy>[_dapo_math]_<model>_<size>``, e.g.
# rl_heloco_qwen3_5_9b or rl_heloco_async_inference_worker_dapo_math_qwen3_4b.
# The names are the API (specs pin them, controld getattr()s them, derives the
# generator preset by inserting "_worker" after the strategy segment, and reads
# the size off the suffix), so they are all defined -- but generated from the
# tables below rather than written out per strategy x model x task. One preset
# exists per strategy and per _DEFAULT_HF_ASSETS_PATH entry.
#
# A preset takes keyword overrides from Python (ConfigManager passes none): any
# base_rl_config argument (seq_len, hf_assets_path, flavor, ...) or any field
# of the strategy's Config (sync_every, num_outer_steps, group_size, ...).

#: Strategy segment -> (Config class, role, values that differ from that
#: Config's own defaults). The classes' docstrings describe each strategy.
#: "colocated": every island runs its own trainer + generators (2 GPUs at the
#: default TP=1). "learner": a pure-learner trainer (no local vLLM) fed by a
#: remote worker pool through the rollout queue. "worker": that pool's
#: generator process, no trainer actor at all.
_STRATEGIES = {
    "rl_solo": (SoloRLReplica, "colocated", {}),
    "rl_diloco": (DiLoCoRLReplica, "colocated", {}),
    "rl_heloco": (HeLoCoRLReplica, "colocated", {"should_quantize": True}),
    "rl_async_inference": (AsyncInferenceReplica, "learner", {}),
    "rl_heloco_async_inference": (
        HeLoCoAsyncInferenceReplica,
        "learner",
        {"should_quantize": True},
    ),
    "rl_async_inference_worker": (AsyncInferenceWorker, "worker", {}),
    # Bigger rounds, so a small pool fills several trainers' windows quickly.
    "rl_heloco_async_inference_worker": (
        AsyncInferenceWorker,
        "worker",
        {"groups_per_round": 8},
    ),
}

#: Per-(model, flavor) layout. Qwen3.5-9B is sized for an H100:8 island: the
#: fp32 trainer state alone is ~144 GB, so the trainer runs TP=4 (~36 GB/GPU;
#: its 4 KV heads cap TP at 4) next to four TP=1 generators. num_generators
#: only applies to colocated islands; a decoupled launch sets the trainer's TP
#: from the island's GPU count (controld), so this is only its default.
_LAYOUT = {
    ("qwen3_5", "9B"): {"trainer_tensor_parallel_degree": 4, "num_generators": 4},
}

#: DAPO-Math budgets: 8K response inside a 10K packed context.
_DAPO_MATH_SEQ_LEN = 10240
_DAPO_MATH_MAX_RESPONSE_TOKENS = 8192


def _dapo_math_rollouter(max_total_tokens: int) -> Rollouter.Config:
    """Upstream's DAPO-Math task wiring (its ``_dapo_math_rollouter_config``):
    DAPO-Math-17k train, all 30 AIME 2025 problems for validation, a binary
    Math-Verify reward, single-turn rollouts within ``max_total_tokens``."""
    # Lazy: the rubric imports math_verify at module level. A top-level import
    # here would take the WHOLE registry down without it -- every preset,
    # alphabet-sort included -- and ConfigManager reports that as the baffling
    # "config function not found".
    from torchtitan.rl.examples.dapo_math.data import AIME2025Dataset, DapoMathDataset
    from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
    from torchtitan.rl.examples.dapo_math.rubric import RewardMathVerify
    from torchtitan.rl.rollout.advantage import AdvantageEstimator

    return Rollouter.Config(
        train_dataset=DapoMathDataset.Config(),
        validation_dataset=AIME2025Dataset.Config(num_samples=30),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardMathVerify.Config(weight=1.0)],
                error_reward=0.0,
            ),
            message_env=DapoMathEnv.Config(),
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens, max_num_turns=1
            ),
            advantage=AdvantageEstimator.Config(should_std_normalize=False),
        ),
    )


def _apply_dapo_math(cfg: RLTrainer.Config, *, colocated: bool) -> None:
    """Overlay the single-node DAPO-Math reference recipe
    (torchtitan/experiments/rl/examples/dapo_math/config_registry.py, the
    ``rl_dapo_qwen3_*_math_*`` presets) onto a base_rl_config, in place, so
    ``rl_heloco_dapo_math_qwen3_4b`` is upstream's ``rl_dapo_qwen3_4b_math_8k``
    run on N replicas with no override pile.

    Everything the reference sets that is not a single-node topology choice is
    here: the task (DAPO-Math-17k train / AIME 2025 validation, math-verify
    reward), thinking on, temperature/top-p 1.0, the 8K response / 10K packing
    budgets, DAPO clip [0.2, 0.28] (arXiv:2503.14476) under a chunked loss,
    the 8 prompts x 16 samples train step at up to 4 steps of lag, and the 1e-6
    CONSTANT LR with betas (0.9, 0.98).

    Not carried over: ``num_generators`` and the trainer's tensor-parallel
    degree (topology, owned by the launcher and the island's GPU count), and
    the reference's fp32-lm-head converter / vLLM cudagraph capture (memory and
    warmup tradeoffs that are not the recipe).

    Needs math-verify on every island that scores rollouts (and the hub).
    """
    seq_len = cfg.trainer.training.max_context_length
    cfg.rollouter = _dapo_math_rollouter(seq_len)
    cfg.renderer = from_renderers(Qwen3RendererConfig(enable_thinking=True))
    cfg.generator.sampling.temperature = 1.0
    cfg.generator.sampling.top_p = 1.0
    cfg.generator.sampling.max_tokens = _DAPO_MATH_MAX_RESPONSE_TOKENS
    # The packed width is ONE sequence per rank, and at this length that is not
    # a tuning preference: measured on an H200, a 4B trainer peaks at 112 GiB at
    # one 10K sequence and OOMs a 140 GiB card at two (there is no activation
    # checkpointing in this config). base_rl_config's 2x is sized for
    # alphabet-sort's 2048-token budget, not a 10K one.
    cfg.trainer.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    cfg.async_loop.validation.num_samples = 30  # all of AIME 2025
    # The reference train step: 8 prompt groups x 16 samples = 128 rollouts.
    # A worker preset takes this as its group_size (see _build).
    cfg.async_loop.num_prompts_per_train_step = 8
    cfg.async_loop.num_samples_per_prompt = 16
    cfg.trainer.loss = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=DAPOLoss.Config(
            ratio_clip_low=0.2,
            ratio_clip_high=0.28,
            global_vocab_size=decoder_vocab_size(cfg.model),
        ),
    )
    # 1e-6 held CONSTANT (warmup 0, floor factor 1.0) -- base_rl_config's
    # 2e-6 warmup+linear-decay is alphabet-sort's schedule, not this recipe's.
    cfg.trainer.optimizer = _adamw(lr=1e-6, betas=(0.9, 0.98), weight_decay=0.1)
    cfg.trainer.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0, min_lr_factor=1.0
    )
    if not colocated:
        # A pure learner consumes the shared queue, where lag is bounded by
        # max_staleness, and has no local generator pool to route between.
        return
    # The reference's rollout lag. base_rl_config pins 0 -- fully on-policy,
    # which caps in-flight work at ONE train step's worth (128 sequences) and
    # leaves a replica's generators idle waiting for each publish. At 4 the cap
    # is 640; the loss is importance-sampling corrected (_batch_staleness), so
    # the lag is priced in rather than silently biasing the update. Keep it
    # below sync_every so the deepest stale sample still sits INSIDE a window
    # instead of straddling a parameter-server merge.
    cfg.async_loop.target_offpolicy_steps = 4
    # base_rl_config pins FIFO-by-batch for the on-policy default; the
    # reference recipe uses upstream's greedy default, so restore it here.
    cfg.async_loop.windowed_fifo_batches = None
    # Generators finish at different times under an 8K budget; round-robin
    # hands work to a generator that is still busy.
    cfg.generator_router.strategy = LeastLoadedRoutingStrategy.Config()


_BASE_RL_CONFIG_ARGS = frozenset(inspect.signature(base_rl_config).parameters)


def _build(strategy: str, model: str, flavor: str, task: str | None, **kwargs):
    """Build one preset: base_rl_config, the task overlay, then the strategy's
    Config. ``kwargs`` win over every table default."""
    cls, role, overrides = _STRATEGIES[strategy]
    kwargs = {"model": model, "flavor": flavor, **overrides, **kwargs}
    kwargs = {**_LAYOUT.get((kwargs["model"], kwargs["flavor"]), {}), **kwargs}
    if task == "dapo_math":
        kwargs.setdefault("seq_len", _DAPO_MATH_SEQ_LEN)
    if role != "colocated":
        kwargs.pop("num_generators", None)  # no local generators in this role
    base = base_rl_config(
        **{k: kwargs.pop(k) for k in _BASE_RL_CONFIG_ARGS & kwargs.keys()}
    )
    if task == "dapo_math":
        _apply_dapo_math(base, colocated=role == "colocated")
    if role == "worker":
        # A worker emits whole GRPO groups; a trainer expecting N siblings
        # cannot use groups of any other size.
        kwargs.setdefault("group_size", base.async_loop.num_samples_per_prompt)
        return AsyncInferenceWorker.Config(
            model=base.model,
            hf_assets_path=base.hf_assets_path,
            generator=base.generator,
            rollouter=base.rollouter,
            renderer=base.renderer,
            **kwargs,
        )
    # Run-bound by wall clock unless a step count is given (exactly one is set).
    kwargs.setdefault("train_seconds", 0.0 if kwargs.get("num_outer_steps") else 3600.0)
    return wrap_replica(cls, base, **kwargs)


def _register(namespace: dict, strategy, model, flavor, task, build, apply=None):
    """Define the preset ``<strategy>[_<task>]_<model>_<size>`` in
    ``namespace``: ``build(**kwargs)``, then ``apply`` if given."""
    size = flavor.lower().replace(".", "_")
    name = "_".join(filter(None, (strategy, task, model, size)))

    def preset(**kwargs):
        cfg = build(**kwargs)
        return apply(cfg) if apply else cfg

    preset.__name__ = preset.__qualname__ = name
    preset.__module__ = namespace["__name__"]
    preset.__doc__ = f"{strategy} on {model} {flavor}" + (f" ({task})" if task else "")
    namespace[name] = preset


def register_task(namespace: dict, task: str, apply, *, model: str, **kwargs) -> None:
    """For a code overlay: define ``<strategy>_<task>_<model>_<size>`` in
    ``namespace`` (the overlay's ``globals()``) for every strategy and every
    checkpoint of ``model`` -- the engine preset built with ``kwargs`` (e.g.
    ``seq_len``), then passed through ``apply``, which returns the config.
    ``apply`` gets trainer and worker Configs alike, so it should only set what
    both carry (rollouter, renderer, generator)."""
    for strategy in _STRATEGIES:
        for m, flavor in _DEFAULT_HF_ASSETS_PATH:
            if m == model:
                build = functools.partial(_build, strategy, m, flavor, None, **kwargs)
                _register(namespace, strategy, m, flavor, task, build, apply)


for _strategy in _STRATEGIES:
    for _model, _flavor in _DEFAULT_HF_ASSETS_PATH:
        _args = (_strategy, _model, _flavor)
        _register(globals(), *_args, None, functools.partial(_build, *_args, None))
        if _model == "qwen3":  # the recipe pins the Qwen3 (thinking) renderer
            _build_dapo = functools.partial(_build, *_args, "dapo_math")
            _register(globals(), *_args, "dapo_math", _build_dapo)
