"""Presets: the engine's Qwen3.5 RL presets with the parcel task swapped in.

controld picks the strategy segment from the spec (one island with nothing to
sync -> ``rl_solo_*``, colocated heloco -> ``rl_heloco_*``, decoupled heloco ->
``rl_heloco_async_inference_*`` plus the ``_worker_`` preset on the generator
islands), so define every variant your specs use.
"""

from panoengine.train.rl import config_registry as engine
from panoengine.train.rl.harness import verifiers_rollouter

from .parcels import ParcelTasksetConfig

# The tool schemas, ten lookups and their results fit in 4K tokens; the preset's seq_len and
# the rollouter's prompt bound must agree.
SEQ_LEN = 4096


def _with_task(cfg):
    cfg.rollouter = verifiers_rollouter(ParcelTasksetConfig(), max_rollout_tokens=SEQ_LEN)
    # One reply: a batch of tool calls (up to ten lookups) or a short sentence.
    cfg.generator.sampling.max_tokens = 512
    return cfg


def rl_solo_tool_agent_qwen3_5_0_8b():
    return _with_task(engine.rl_solo_qwen3_5_0_8b(seq_len=SEQ_LEN))


def rl_solo_tool_agent_qwen3_5_9b():
    return _with_task(engine.rl_solo_qwen3_5_9b(seq_len=SEQ_LEN))


def rl_heloco_tool_agent_qwen3_5_0_8b():
    return _with_task(engine.rl_heloco_qwen3_5_0_8b(seq_len=SEQ_LEN))


def rl_heloco_tool_agent_qwen3_5_9b():
    return _with_task(engine.rl_heloco_qwen3_5_9b(seq_len=SEQ_LEN))


def rl_heloco_async_inference_tool_agent_qwen3_5_0_8b():
    return _with_task(engine.rl_heloco_async_inference_qwen3_5_0_8b(seq_len=SEQ_LEN))


def rl_heloco_async_inference_tool_agent_qwen3_5_9b():
    return _with_task(engine.rl_heloco_async_inference_qwen3_5_9b(seq_len=SEQ_LEN))


def rl_heloco_async_inference_worker_tool_agent_qwen3_5_0_8b():
    return _with_task(engine.rl_heloco_async_inference_worker_qwen3_5_0_8b(seq_len=SEQ_LEN))


def rl_heloco_async_inference_worker_tool_agent_qwen3_5_9b():
    return _with_task(engine.rl_heloco_async_inference_worker_qwen3_5_9b(seq_len=SEQ_LEN))
