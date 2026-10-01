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
# RLTrainer.Config) plus the coordinator-specific extras. The ``rl_*``
# functions below are the full-size entry points, one per strategy x model
# (each with a "_0_6b" and, where a checkpoint is available, a larger preset).
#
# Adding a new RL model: add it to ``_MODEL_REGISTRY_BY_MODEL`` and
# ``_RENDERER_NAME_BY_MODEL`` below (and, if it should have a default
# checkpoint, ``_DEFAULT_HF_ASSETS_PATH``) -- no changes needed anywhere else
# in this package. GPU count is not fixed either: trainer/generator
# tensor_parallel_degree are real parameters here (default 1), and
# num_replicas / GPUS_PER_REPLICA (launch script arg) flow through
# independently -- nothing below assumes a specific machine's GPU count, only
# its own defaults do.
#
# ConfigManager calls the ``--config`` function with NO arguments (CLI flags
# then overlay onto the resulting dataclass's fields), so a size/flavor/model
# switch needs its own named entry point to be reachable from the CLI --
# passing ``model="llama3"`` to ``rl_heloco_qwen3_0_6b`` only works from
# Python (as the ``rl_heloco_llama3_8b`` wrapper below does).

import dataclasses
import os

from renderers import DefaultRendererConfig, Qwen3RendererConfig

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
from panoengine.train.rl.controller import RLTrainer
from panoengine.train.rl.replicas import (
    AsyncInferenceReplica,
    DiLoCoRLReplica,
    HeLoCoAsyncInferenceReplica,
    HeLoCoRLReplica,
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


#: Entries share one signature: (flavor, *, seq_len, attn_backend).
_MODEL_REGISTRY_BY_MODEL = {
    "qwen3": _qwen3_model_registry,
    "llama3": _llama3_model_registry,
}

#: Chat template per model (the `renderers` package). llama3 keeps the generic
#: "default" template it has always used here.
_RENDERER_BY_MODEL = {
    "qwen3": lambda: Qwen3RendererConfig(enable_thinking=False),
    "llama3": lambda: DefaultRendererConfig(),
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
        compile=CompileConfig(backend="aot_eager"),
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
            ),
            loss=GRPOLoss.Config(global_vocab_size=decoder_vocab_size(model_config)),
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


def rl_heloco_qwen3_0_6b(
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    should_quantize: bool = True,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
    seq_len: int = _SEQ_LEN,
) -> HeLoCoRLReplica.Config:
    """Async multi-worker GRPO (2 GPUs/worker at the default TP=1: 1 generator
    + 1 trainer GPU each, so two workers fit on a 4-GPU node alongside the CPU
    parameter server -- raise the tensor_parallel_degree params for a model
    that needs more than one GPU per role; the launch script's
    GPUS_PER_REPLICA must then grow to match).

    N workers each run a full RL loop and sync pseudo-gradients through the CPU
    parameter server (HeLoCo outer optimizer) with no barrier. Loss is stock GRPO.
    """
    return wrap_replica(
        HeLoCoRLReplica,
        base_rl_config(
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
            seq_len=seq_len,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        should_quantize=should_quantize,
    )


def rl_heloco_qwen3_1_7b(**kwargs) -> HeLoCoRLReplica.Config:
    """1.7B preset; see rl_heloco_qwen3_0_6b for the strategy docstring
    (same 2-GPUs/worker layout -- 1.7B still fits comfortably at TP=1)."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_qwen3_0_6b(**kwargs)


def rl_heloco_llama3_8b(**kwargs) -> HeLoCoRLReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. No RL checkpoint ships for llama3; download one to
    example_checkpoint/Llama-3.1-8B first, or pass hf_assets_path explicitly.
    See rl_heloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_heloco_qwen3_0_6b(**kwargs)


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


def _apply_dapo_math(cfg, *, max_response_tokens: int, max_total_tokens: int):
    """Overlay the single-node DAPO-Math reference recipe
    (torchtitan/experiments/rl/examples/dapo_math/config_registry.py, the
    ``rl_dapo_qwen3_*_math_*`` presets) onto a decentralized_rl replica config,
    in place.

    Everything the reference sets that is not a single-node topology choice is
    here: the task (DAPO-Math-17k train / AIME 2025 validation, math-verify
    reward), thinking on, temperature/top-p 1.0, the 8K response / 10K packing
    budgets, DAPO clip [0.2, 0.28] under a chunked loss, the 8 prompts x 16
    samples train step at up to 4 steps of lag, and the 1e-6 CONSTANT LR with
    betas (0.9, 0.98). A caller that wants a different loop overrides these
    fields from the CLI as usual -- the point is that the DEFAULTS are
    upstream's, so "run the reference recipe" needs no override pile.

    Not carried over: ``num_generators`` and the trainer's tensor-parallel
    degree (topology, owned by the launcher and the island's GPU count), and
    the reference's fp32-lm-head converter / vLLM cudagraph capture (memory and
    warmup tradeoffs that are not the recipe).
    """
    cfg.rollouter = _dapo_math_rollouter(max_total_tokens)
    cfg.renderer = from_renderers(Qwen3RendererConfig(enable_thinking=True))
    cfg.generator.sampling.temperature = 1.0
    cfg.generator.sampling.top_p = 1.0
    cfg.generator.sampling.max_tokens = max_response_tokens
    # The caller built the model with seq_len=max_total_tokens (its context
    # bound); the packed width is ONE sequence per rank, and at this length that
    # is not a tuning preference: measured on an H200, a 4B trainer peaks at
    # 112 GiB at one 10K sequence and OOMs a 140 GiB card at two (there is no
    # activation checkpointing in this config). base_rl_config's 2x is sized for
    # alphabet-sort's 2048-token budget, not a 10K one.
    cfg.trainer.training.num_tokens_per_microbatch_per_dp_rank = max_total_tokens
    cfg.trainer.training.max_context_length = max_total_tokens
    cfg.async_loop.validation.num_samples = 30  # all of AIME 2025
    # The reference train step: 8 prompt groups x 16 samples = 128 rollouts.
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
    return cfg


def rl_heloco_dapo_math_qwen3_0_6b(
    max_response_tokens: int = 8192,
    max_total_tokens: int = 10240,
    **kwargs,
) -> HeLoCoRLReplica.Config:
    """DAPO-Math on the heloco stack: single-turn math prompts (DAPO-Math-17k
    filtered for training, all 30 AIME 2025 problems for validation) scored by
    a binary Math-Verify reward, trained with the DAPO clip-higher loss
    (arXiv:2503.14476) instead of stock GRPO. Everything else -- the async
    multi-worker loop, the CPU parameter server, quantized pushes -- is
    rl_heloco_qwen3_0_6b (see its docstring for the strategy layout).

    Defaults ARE the single-node reference recipe (see _apply_dapo_math), so
    this is upstream's ``rl_dapo_qwen3_4b_math_8k`` run on N replicas rather
    than one node -- no override pile needed to reproduce it.

    Extra dependency: math-verify (examples/dapo_math/requirements.txt) --
    NOT baked into the engine image; declare it on the run so the islands and
    the parameter-server hub install it at provisioning.
    """
    cfg = _apply_dapo_math(
        rl_heloco_qwen3_0_6b(seq_len=max_total_tokens, **kwargs),
        max_response_tokens=max_response_tokens,
        max_total_tokens=max_total_tokens,
    )
    # Local generation only (the pure-learner variant has no generators, and
    # no router to point at them).
    #
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
    return cfg


def rl_heloco_dapo_math_qwen3_4b(**kwargs) -> HeLoCoRLReplica.Config:
    """The reference DAPO-Math model (Qwen3-4B-Base); needs the checkpoint
    downloaded into _DEFAULT_HF_ASSETS_PATH's dir (or hf_assets_path passed).
    See rl_heloco_dapo_math_qwen3_0_6b."""
    kwargs.setdefault("flavor", "4B")
    return rl_heloco_dapo_math_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_qwen3_0_6b(
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    should_quantize: bool = True,
    max_staleness: int = 4,
    rollout_queue_address: str = "",
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
    seq_len: int = _SEQ_LEN,
) -> HeLoCoAsyncInferenceReplica.Config:
    """Decoupled generation (arXiv:2505.07291) scaled to
    MULTIPLE trainers: N PURE-LEARNER HeLoCo trainer replicas (1 GPU each at
    the default TP=1 -- no local generation, no vLLM on the trainer) plus a
    separate pool of rl_heloco_async_inference_worker_* generator processes on
    their own machines that free-run rollouts into a hub-hosted shared queue.
    Each trainer pops rollouts from that queue, trains, and pushes its
    pseudo-gradient to the HeLoCo parameter server (no barrier); any trainer
    may consume any worker's rollouts. The hub
    (panoengine.decentralized.parameter_server)
    publishes the CURRENT global theta (the consensus weights, not any one
    trainer's copy) to a relay process for the generator pool to pull. Start
    the coordination plane first: the relay,
    then the rollout_queue, then the server (with --relay_addr).
    rollout_queue_address is required (usually $ROLLOUT_QUEUE_ADDR, the same
    queue workers' rollout_queue_address points at). Loss is stock GRPO.
    """
    return wrap_replica(
        HeLoCoAsyncInferenceReplica,
        base_rl_config(
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
            seq_len=seq_len,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        should_quantize=should_quantize,
        max_staleness=max_staleness,
        rollout_queue_address=rollout_queue_address,
        # Pure learner: no local vLLM (Controller.Config defaults to 1, but
        # generation is fully decoupled onto the remote worker pool).
        num_generators=0,
    )


def rl_heloco_async_inference_qwen3_1_7b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """1.7B preset; see rl_heloco_async_inference_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_llama3_8b(**kwargs) -> HeLoCoAsyncInferenceReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. See rl_heloco_async_inference_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_dapo_math_qwen3_0_6b(
    max_response_tokens: int = 8192,
    max_total_tokens: int = 10240,
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """DAPO-Math on the decoupled heloco stack: the task and recipe of
    rl_heloco_dapo_math_qwen3_0_6b (see _apply_dapo_math) on the pure-learner
    topology of rl_heloco_async_inference_qwen3_0_6b (see its docstring for the
    relay / rollout-queue / parameter-server layout).

    The generator side of the recipe reaches the swarm through the matching
    rl_heloco_async_inference_worker_dapo_math_* preset, not this one: a pure
    learner runs no local vLLM, so the sampling knobs set here are inert and
    only the trainer-side ones (batch width, loss, optimizer, group size)
    take effect. Both must move together -- the worker's ``group_size`` is
    this config's ``async_loop.num_samples_per_prompt``.

    Extra dependency: math-verify, on the trainers AND the worker islands.
    """
    # No max_offpolicy_steps / router overlay here (unlike the non-decoupled
    # preset): a pure learner consumes the shared queue, where lag is bounded
    # by max_staleness, and it has no local generator pool to route between.
    return _apply_dapo_math(
        rl_heloco_async_inference_qwen3_0_6b(seq_len=max_total_tokens, **kwargs),
        max_response_tokens=max_response_tokens,
        max_total_tokens=max_total_tokens,
    )


def rl_heloco_async_inference_dapo_math_qwen3_4b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """The reference DAPO-Math model (Qwen3-4B-Base) on the decoupled stack;
    see rl_heloco_dapo_math_qwen3_4b for the checkpoint requirement."""
    kwargs.setdefault("flavor", "4B")
    return rl_heloco_async_inference_dapo_math_qwen3_0_6b(**kwargs)


def rl_diloco_qwen3_0_6b(
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    num_replicas: int = 2,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> DiLoCoRLReplica.Config:
    """Synchronous DiLoCo GRPO (2 GPUs/worker at the default TP=1: 1 generator
    + 1 trainer GPU each, so two workers fit on a 4-GPU node -- raise the
    tensor_parallel_degree params for a model that needs more than one GPU per
    role; the launch script's GPUS_PER_REPLICA must then grow to match).

    N workers coordinate through a torchft Lighthouse/Manager quorum and sync
    averaged pseudo-gradients + an outer Nesterov-SGD step every sync_every
    steps (stock DiLoCo, no parameter server). Loss is stock GRPO.
    """
    return wrap_replica(
        DiLoCoRLReplica,
        base_rl_config(
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        num_replicas=num_replicas,
    )


def rl_diloco_qwen3_1_7b(**kwargs) -> DiLoCoRLReplica.Config:
    """1.7B preset; see rl_diloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_diloco_llama3_8b(**kwargs) -> DiLoCoRLReplica.Config:
    """Llama3-8B preset; see rl_heloco_llama3_8b for the extension-point
    note and rl_diloco_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_async_inference_qwen3_0_6b(
    hf_assets_path: str | None = None,
    sync_every: int = 4,
    train_seconds: float = 3600.0,
    num_outer_steps: int = 0,
    max_staleness: int = 4,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    num_shards: int = 4,
    publish_every: int = 1,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
) -> AsyncInferenceReplica.Config:
    """Trainer role: ONE pure-learner trainer (1 GPU, no local vLLM) fed 
    entirely by a pool of remote generator workers
    (rl_async_inference_worker_*) on their own machines. The trainer pops
    rollouts from the standalone queue process at rollout_queue_address under
    a max_staleness bound, trains, and shards + publishes its weights to
    relay_addresses every publish_every windows (SHARDCAST-style) -- plus an
    initial publish at startup so the workers can bootstrap. Both addresses
    are required -- start the servers first:
    ``python -m panoengine.train.rl.relay`` and
    ``python -m panoengine.train.rl.rollout_queue``.
    Workers push rollouts to the same queue
    ($ASYNC_INFERENCE_ROLLOUT_QUEUE_ADDR) and pull weights from the relay
    ($ASYNC_INFERENCE_RELAY_ADDRS).
    """
    return wrap_replica(
        AsyncInferenceReplica,
        base_rl_config(
            hf_assets_path,
            model=model,
            flavor=flavor,
            trainer_tensor_parallel_degree=trainer_tensor_parallel_degree,
            generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        ),
        sync_every=sync_every,
        train_seconds=0.0 if num_outer_steps else train_seconds,
        num_outer_steps=num_outer_steps,
        max_staleness=max_staleness,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        num_shards=num_shards,
        publish_every=publish_every,
        # Pure learner: no local vLLM (Controller.Config defaults to 1, but
        # generation is fully decoupled onto the remote worker pool).
        num_generators=0,
    )


def rl_async_inference_qwen3_1_7b(**kwargs) -> AsyncInferenceReplica.Config:
    """1.7B preset; see rl_async_inference_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_llama3_8b(**kwargs) -> AsyncInferenceReplica.Config:
    """Llama3-8B preset -- a second model family, proving the
    _MODEL_REGISTRY_BY_MODEL extension point works end to end with no other
    changes. See rl_async_inference_qwen3_0_6b for the strategy docstring."""
    kwargs.setdefault("model", "llama3")
    kwargs.setdefault("flavor", "8B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_qwen3_0_6b(
    hf_assets_path: str | None = None,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    worker_id: int = 0,
    group_size: int = 8,
    groups_per_round: int = 2,
    poll_interval_s: float = 2.0,
    num_rounds: int = 0,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    generator_tensor_parallel_degree: int = 1,
) -> AsyncInferenceWorker.Config:
    """Inference-worker role of the async-inference relay swarm: no trainer
    fields apply here (this role has no trainer actor -- see
    async_inference/worker.py), so this copies only what a generator needs
    out of base_rl_config() rather than going through wrap_replica (which
    assumes an RLTrainer.Config-shaped target). relay_addresses (weights in)
    and rollout_queue_address (rollouts out, the standalone queue process both
    this worker and the trainer talk to) are both required.
    """
    base = base_rl_config(
        hf_assets_path=hf_assets_path,
        model=model,
        flavor=flavor,
        generator_tensor_parallel_degree=generator_tensor_parallel_degree,
    )
    return AsyncInferenceWorker.Config(
        model=base.model,
        hf_assets_path=hf_assets_path or base.hf_assets_path,
        generator=base.generator,
        rollouter=base.rollouter,
        renderer=base.renderer,
        group_size=group_size,
        groups_per_round=groups_per_round,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        worker_id=worker_id,
        poll_interval_s=poll_interval_s,
        num_rounds=num_rounds,
    )


def rl_async_inference_worker_qwen3_1_7b(**kwargs) -> AsyncInferenceWorker.Config:
    """1.7B preset; see rl_async_inference_worker_qwen3_0_6b for the strategy
    docstring."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_qwen3_0_6b(
    hf_assets_path: str | None = None,
    relay_addresses: str = "",
    rollout_queue_address: str = "",
    worker_id: int = 0,
    group_size: int = 8,
    groups_per_round: int = 8,
    poll_interval_s: float = 2.0,
    num_rounds: int = 0,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    generator_tensor_parallel_degree: int = 1,
    seq_len: int = _SEQ_LEN,
) -> AsyncInferenceWorker.Config:
    """Inference-worker (generator) role of the heloco_async_inference swarm:
    the exact same AsyncInferenceWorker process as rl_async_inference_worker_*
    (all workers free-run -- generate continuously at their current weights,
    upgrading opportunistically -- which is the definition of a decoupled
    async generator). These workers are the trainers' SOLE rollout source (no
    trainer here runs any local generation), and there can be many of them
    feeding many trainers through the one hub. Point relay_addresses
    (weights in) at the relay process and rollout_queue_address (rollouts out)
    at the shared queue process ($ROLLOUT_QUEUE_ADDR). ``groups_per_round`` defaults
    higher than the base preset so a small generator pool fills a trainer's
    per-window token target in a few rounds.
    """
    base = base_rl_config(
        hf_assets_path=hf_assets_path,
        model=model,
        flavor=flavor,
        generator_tensor_parallel_degree=generator_tensor_parallel_degree,
        seq_len=seq_len,
    )
    return AsyncInferenceWorker.Config(
        model=base.model,
        hf_assets_path=hf_assets_path or base.hf_assets_path,
        generator=base.generator,
        rollouter=base.rollouter,
        renderer=base.renderer,
        group_size=group_size,
        groups_per_round=groups_per_round,
        relay_addresses=relay_addresses,
        rollout_queue_address=rollout_queue_address,
        worker_id=worker_id,
        poll_interval_s=poll_interval_s,
        num_rounds=num_rounds,
    )


def rl_heloco_async_inference_worker_qwen3_1_7b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """1.7B preset; see rl_heloco_async_inference_worker_qwen3_0_6b."""
    kwargs.setdefault("flavor", "1.7B")
    return rl_heloco_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b(
    max_response_tokens: int = 8192,
    max_total_tokens: int = 10240,
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """Generator role for the DAPO-Math swarm: the worker of
    rl_heloco_async_inference_worker_qwen3_0_6b (see it for the role) running
    the DAPO-Math task with the reference sampling settings -- thinking on,
    temperature/top-p 1.0, 8K response budget.

    ``group_size`` is 16 to match the trainer preset's
    ``async_loop.num_samples_per_prompt``: a worker emits whole GRPO groups and
    a trainer that expects 16 siblings cannot use groups of 8.

    Extra dependency: math-verify.
    """
    kwargs.setdefault("group_size", 16)
    cfg = rl_heloco_async_inference_worker_qwen3_0_6b(seq_len=max_total_tokens, **kwargs)
    cfg.rollouter = _dapo_math_rollouter(max_total_tokens)
    cfg.renderer = from_renderers(Qwen3RendererConfig(enable_thinking=True))
    cfg.generator.sampling.temperature = 1.0
    cfg.generator.sampling.top_p = 1.0
    cfg.generator.sampling.max_tokens = max_response_tokens
    return cfg


def rl_heloco_async_inference_worker_dapo_math_qwen3_4b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """4B DAPO-Math worker; see
    rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b."""
    kwargs.setdefault("flavor", "4B")
    return rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b(**kwargs)
