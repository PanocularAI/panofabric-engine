"""Presets: the engine's Qwen3.5 RL presets with the parcel task swapped in.

controld picks the strategy segment from the spec (one island with nothing to
sync -> ``rl_solo_*``, colocated heloco -> ``rl_heloco_*``, decoupled heloco ->
``rl_heloco_async_inference_*`` plus the ``_worker_`` preset on the generator
islands). ``register_task`` at the bottom defines every variant, from
``rl_solo_tool_agent_qwen3_5_0_8b`` to
``rl_heloco_async_inference_worker_tool_agent_qwen3_5_9b``.
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


engine.register_task(globals(), "tool_agent", _with_task, model="qwen3_5", seq_len=SEQ_LEN)
