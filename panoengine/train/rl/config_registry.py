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

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments import rl as _rl_pkg
from panoengine.train.rl.controller import RLTrainer
from panoengine.train.rl.replicas import (
    AsyncInferenceReplica,
    DiLoCoRLReplica,
    HeLoCoAsyncInferenceReplica,
    HeLoCoRLReplica,
)
from panoengine.train.rl.worker import AsyncInferenceWorker
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import AsyncLoopConfig, ValidationConfig
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
from torchtitan.experiments.rl.losses import DAPOLoss, GRPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.routing import (
    InterGeneratorRouter,
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
)
from torchtitan.models.llama3 import model_registry as _llama3_model_registry
from torchtitan.models.qwen3 import model_registry as _qwen3_model_registry


def _hf_model_registry(flavor, *, attn_backend, hf_assets_path):
    # Lazy: pulls transformers, only needed when the "hf" model is selected.
    from panoengine.train.rl.hf_model_registry import model_registry

    return model_registry(
        flavor, attn_backend=attn_backend, hf_assets_path=hf_assets_path
    )


#: Entries share one signature: (flavor, *, attn_backend, hf_assets_path).
#: Native registries ignore hf_assets_path (they load weights via their own
#: state-dict adapters); the "hf" backend resolves its ARCHITECTURE from it.
_MODEL_REGISTRY_BY_MODEL = {
    "qwen3": lambda flavor, *, attn_backend, hf_assets_path: _qwen3_model_registry(
        flavor, attn_backend=attn_backend
    ),
    "llama3": lambda flavor, *, attn_backend, hf_assets_path: _llama3_model_registry(
        flavor, attn_backend=attn_backend
    ),
    "hf": _hf_model_registry,
}

#: Renderer (chat-template) name per model -- see
#: torchtitan.experiments.rl.renderer._RENDERER_BY_MODEL. llama3 has no
#: dedicated entry there, so it resolves via the "default" key.
_RENDERER_NAME_BY_MODEL = {
    "qwen3": "qwen3",
    "llama3": "default",
    # Arbitrary HF architectures: resolve the chat template from the
    # checkpoint's own tokenizer (renderers "auto" path).
    "hf": "auto",
}

_EXAMPLE_CHECKPOINT_DIR = os.path.join(_rl_pkg.__path__[0], "example_checkpoint")

#: Default hf_assets_path per (model, flavor). "qwen3"/"0.6B" ships inside
#: torchtitan's rl experiment (no download needed); the rest are conventional
#: paths under the same example_checkpoint dir (matching
#: torchtitan.experiments.rl.config_registry's own qwen3_1_7b/qwen3_14b
#: presets) that the caller downloads a checkpoint into, or pass
#: hf_assets_path explicitly.
_DEFAULT_HF_ASSETS_PATH = {
    ("qwen3", "0.6B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-0.6B"),
    ("qwen3", "1.7B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-1.7B"),
    ("qwen3", "4B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-4B-Base"),
    ("llama3", "8B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Llama-3.1-8B"),
    # HF backend A/B baselines: same checkpoints as the native qwen3 presets.
    ("hf", "Qwen3-0.6B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-0.6B"),
    ("hf", "Qwen3-1.7B"): os.path.join(_EXAMPLE_CHECKPOINT_DIR, "Qwen3-1.7B"),
}


def base_rl_config(
    hf_assets_path: str | None = None,
    *,
    model: str = "qwen3",
    flavor: str = "0.6B",
    trainer_tensor_parallel_degree: int = 1,
    generator_tensor_parallel_degree: int = 1,
    rollouter=None,
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
            "_MODEL_REGISTRY_BY_MODEL/_RENDERER_NAME_BY_MODEL in this file"
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
    model_registry = _MODEL_REGISTRY_BY_MODEL[model]
    return RLTrainer.Config(
        model_spec=model_registry(
            flavor,
            attn_backend="varlen",
            hf_assets_path=resolved_hf_assets_path,
        ),
        hf_assets_path=resolved_hf_assets_path,
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=8,
            # decentralized_rl replicas drive a SYNCHRONOUS windowed loop
            # (RLControllerMixin), so no off-policy buffering: each step trains
            # on rollouts generated under the just-published policy.
            # BOTH knobs are required for that: upstream's windowed FIFO
            # (window_fraction, default 0.3) lets observed offpolicy steps
            # EXCEED the target, so target=0 alone derives a bound of 1, not 0.
            # window_fraction=None is strict FIFO (window_size 1) => bound 0.
            target_offpolicy_steps=0,
            window_fraction=None,
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=rollouter
        if rollouter is not None
        else AlphabetSortRollouter.Config(),
        renderer=RendererConfig(
            name=_RENDERER_NAME_BY_MODEL[model], enable_thinking=False
        ),
        generator_router=InterGeneratorRouter.Config(
            strategy=RoundRobinRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=PolicyTrainer.Config(
            # Structured-logging JSONL traces are a debugging aid that costs
            # real disk (hundreds of MB/hour per actor); off by default here
            # since decentralized_rl runs are typically many-hour, many-actor swarms.
            debug=DebugConfig(enable_structured_logging=False),
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=trainer_tensor_parallel_degree,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            debug=DebugConfig(enable_structured_logging=False),
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=generator_tensor_parallel_degree,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    # Lazy: the rollouter's rubric imports math_verify at module level. A
    # top-level import here would take the WHOLE registry down without it --
    # every preset, alphabet-sort included -- and ConfigManager reports that
    # as the baffling "config function not found".
    from torchtitan.experiments.rl.examples.dapo_math import DapoMathRollouter

    cfg.rollouter = DapoMathRollouter.Config(
        token_env=TokenEnv.Config(
            max_rollout_tokens=max_total_tokens, max_num_turns=1
        ),
    )
    cfg.renderer.enable_thinking = True
    cfg.generator.sampling.temperature = 1.0
    cfg.generator.sampling.top_p = 1.0
    cfg.generator.sampling.max_tokens = max_response_tokens
    cfg.async_loop.batcher.batch.seq_len = max_total_tokens
    # Upstream's per-rank batch is ONE packed sequence, and at this seq_len that is
    # not a tuning preference: measured on an H200, a 4B trainer peaks at 112 GiB at
    # local_batch_size=1 and OOMs a 140 GiB card at 2 (there is no activation
    # checkpointing in this config). base_rl_config's 2 is sized for alphabet-sort's
    # 2048-token budget, not a 10K one.
    cfg.async_loop.batcher.batch.local_batch_size = 1
    cfg.async_loop.validation.num_samples = 30  # all of AIME 2025
    # The reference train step: 8 prompt groups x 16 samples = 128 rollouts.
    cfg.async_loop.num_prompts_per_train_step = 8
    cfg.async_loop.num_samples_per_prompt = 16
    cfg.trainer.loss = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
    )
    # 1e-6 held CONSTANT (warmup 0, floor factor 1.0) -- base_rl_config's
    # 2e-6 warmup+linear-decay is alphabet-sort's schedule, not this recipe's.
    cfg.trainer.optimizer = default_adamw(
        lr=1e-6, betas=(0.9, 0.98), weight_decay=0.1
    )
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
        rl_heloco_qwen3_0_6b(**kwargs),
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
    # base_rl_config pins strict FIFO for the on-policy default; the reference
    # recipe uses upstream's windowed default, so restore it here.
    cfg.async_loop.window_fraction = 0.3
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


def rl_heloco_hf_qwen3_0_6b(**kwargs) -> HeLoCoRLReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes; rl_heloco_qwen3_0_6b for
    the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_qwen3_0_6b(**kwargs)


def rl_heloco_hf_qwen3_1_7b(**kwargs) -> HeLoCoRLReplica.Config:
    """1.7B HF-backend preset; see rl_heloco_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_hf_qwen3_0_6b(**kwargs)


def rl_heloco_hf(**kwargs) -> HeLoCoRLReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_hf_qwen3_0_6b(**kwargs)


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


def rl_heloco_async_inference_hf_qwen3_0_6b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes;
    rl_heloco_async_inference_qwen3_0_6b for the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_async_inference_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_hf_qwen3_1_7b(
    **kwargs,
) -> HeLoCoAsyncInferenceReplica.Config:
    """1.7B HF-backend preset; see rl_heloco_async_inference_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_hf(**kwargs) -> HeLoCoAsyncInferenceReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_async_inference_hf_qwen3_0_6b(**kwargs)


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
        rl_heloco_async_inference_qwen3_0_6b(**kwargs),
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


def rl_diloco_hf_qwen3_0_6b(**kwargs) -> DiLoCoRLReplica.Config:
    """Qwen3-0.6B via the HF transformers modeling backend — the architecture
    comes from the checkpoint's config.json, the modules ARE transformers
    modules, and weights load through the near-identity HF adapter. Same
    GRPO/DiLoCo recipe as rl_diloco_qwen3_0_6b (its A/B baseline), including
    trainer-side per-layer compile (aot_eager) and generator-side CUDA-graph
    capture. Generator-side per-layer compile is auto-disabled by the vLLM
    wrapper for HF backends: the transformers AttentionInterface dict dispatch
    (attention_instances[layer_idx]) specializes dynamo per layer x shape and
    exceeds any fullgraph recompile budget."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_diloco_qwen3_0_6b(**kwargs)


def rl_diloco_hf_qwen3_1_7b(**kwargs) -> DiLoCoRLReplica.Config:
    """1.7B HF-backend preset; see rl_diloco_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_diloco_hf_qwen3_0_6b(**kwargs)


def rl_diloco_hf(**kwargs) -> DiLoCoRLReplica.Config:
    """Model-agnostic HF-backend preset: the checkpoint (and therefore the
    architecture) comes entirely from --hf_assets_path — pass it explicitly
    (launchers that fetch a HF repo always do). The flavor default is only the
    fallback checkpoint for bare CLI runs. Same for the rl_heloco_hf /
    rl_async_inference_hf / rl_heloco_async_inference_hf (+_worker_) family."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_diloco_hf_qwen3_0_6b(**kwargs)


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


def rl_async_inference_hf_qwen3_0_6b(**kwargs) -> AsyncInferenceReplica.Config:
    """Qwen3-0.6B on the HF transformers modeling backend (see
    rl_diloco_hf_qwen3_0_6b for the backend notes;
    rl_async_inference_qwen3_0_6b for the strategy docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_async_inference_qwen3_0_6b(**kwargs)


def rl_async_inference_hf_qwen3_1_7b(**kwargs) -> AsyncInferenceReplica.Config:
    """1.7B HF-backend preset; see rl_async_inference_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_async_inference_hf_qwen3_0_6b(**kwargs)


def rl_async_inference_hf(**kwargs) -> AsyncInferenceReplica.Config:
    """Model-agnostic HF-backend preset; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_async_inference_hf_qwen3_0_6b(**kwargs)


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
        model_spec=base.model_spec,
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


def rl_async_inference_worker_hf_qwen3_0_6b(**kwargs) -> AsyncInferenceWorker.Config:
    """Qwen3-0.6B HF-backend worker (see rl_diloco_hf_qwen3_0_6b for the
    backend notes; rl_async_inference_worker_qwen3_0_6b for the role
    docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_hf_qwen3_1_7b(**kwargs) -> AsyncInferenceWorker.Config:
    """1.7B HF-backend worker; see rl_async_inference_worker_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_async_inference_worker_hf_qwen3_0_6b(**kwargs)


def rl_async_inference_worker_hf(**kwargs) -> AsyncInferenceWorker.Config:
    """Model-agnostic HF-backend worker; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_async_inference_worker_hf_qwen3_0_6b(**kwargs)


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
    )
    return AsyncInferenceWorker.Config(
        model_spec=base.model_spec,
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


def rl_heloco_async_inference_worker_hf_qwen3_0_6b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """Qwen3-0.6B HF-backend worker (see rl_diloco_hf_qwen3_0_6b for the
    backend notes; rl_heloco_async_inference_worker_qwen3_0_6b for the role
    docstring)."""
    kwargs.setdefault("model", "hf")
    kwargs.setdefault("flavor", "Qwen3-0.6B")
    return rl_heloco_async_inference_worker_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_hf_qwen3_1_7b(
    **kwargs,
) -> AsyncInferenceWorker.Config:
    """1.7B HF-backend worker; see
    rl_heloco_async_inference_worker_hf_qwen3_0_6b."""
    kwargs.setdefault("flavor", "Qwen3-1.7B")
    return rl_heloco_async_inference_worker_hf_qwen3_0_6b(**kwargs)


def rl_heloco_async_inference_worker_hf(**kwargs) -> AsyncInferenceWorker.Config:
    """Model-agnostic HF-backend worker; see rl_diloco_hf."""
    if os.environ.get("RL_HF_ASSETS_PATH"):
        # ConfigManager calls config fns with NO args and only overlays CLI
        # flags onto the returned dataclass's FIELDS — too late for the
        # model_spec built here. Launchers that fetch a HF repo export the
        # checkpoint dir so the architecture resolves from it at call time.
        kwargs.setdefault("hf_assets_path", os.environ["RL_HF_ASSETS_PATH"])
    return rl_heloco_async_inference_worker_hf_qwen3_0_6b(**kwargs)


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
    # Lazy import: see _apply_dapo_math.
    from torchtitan.experiments.rl.examples.dapo_math import DapoMathRollouter

    kwargs.setdefault("group_size", 16)
    cfg = rl_heloco_async_inference_worker_qwen3_0_6b(**kwargs)
    cfg.rollouter = DapoMathRollouter.Config(
        token_env=TokenEnv.Config(
            max_rollout_tokens=max_total_tokens, max_num_turns=1
        ),
    )
    cfg.renderer.enable_thinking = True
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
