# Copyright (c) Panocular AI.
#
# Bring-your-own agent harness for kind: rl, on torchtitan's Verifiers bridge
# (torchtitan/rl/examples/verifiers).
#
# The default rollout path drives a TitanRL MessageEnv one turn at a time. Here
# YOUR agent loop runs the episode instead: it talks to the live policy through
# an ordinary OpenAI client, calls its tools, and keeps going until it is done.
# Verifiers' interception server sits behind that client, records every turn
# token-exact, and hands the finished trace back to the trainer -- so an
# existing OpenAI-client harness plugs in unchanged, pointed at a different
# base URL. Tool calls in the replies are parsed by the run's renderer
# (Qwen3.5's XML format via the qwen3_5 presets), not by a vLLM parser.
#
# What a user writes (a workspace code overlay, models/<pkg>/):
#   - one module exporting, via __all__, a vf.Taskset (the tasks, with
#     @vf.reward methods scoring each finished trace) AND an AgentHarness
#     subclass (the agent loop). Verifiers loads a taskset module's own harness
#     by default, which is what lets both cross into its env-server process.
#   - config_registry.py: an rl_* preset that takes one of the engine's presets
#     and swaps in ``verifiers_rollouter(MyTasksetConfig(), ...)``.
# examples/rl_tool_agent is a complete one.

from __future__ import annotations

import sys
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire

from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.rubric import Rubric


class AgentHarness(vf.Harness[vf.HarnessConfig]):
    """A Verifiers harness that runs your agent loop in-process.

    Implement ``run_agent``. Make EVERY model call through the ``client`` it is
    given: that client's base URL is the interception server, and a call that
    bypasses it is invisible to training. Stash whatever your rewards need
    (tool results, failed calls) in ``trace.info``; the taskset's ``@vf.reward``
    methods read it from the same trace. An exception fails the rollout, which
    then scores the rubric's ``error_reward``.
    """

    APPENDS_SYSTEM_PROMPT = True
    EXECUTES_CODE = False   # the loop runs here; the model gets no shell
    NEEDS_CONTAINER = False

    async def run_agent(
        self,
        client: Any,
        model: str,
        messages: list[dict],
        data: vf.TaskData,
        trace: vf.Trace,
    ) -> None:
        """Run one episode. ``client`` is an ``openai.AsyncOpenAI``; ``messages``
        is the task's opening conversation in OpenAI wire format."""
        raise NotImplementedError

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        from openai import AsyncOpenAI

        system_prompt, prompt = self.resolve_prompt(data)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages += [message_to_wire(m) for m in prompt]
        async with AsyncOpenAI(base_url=endpoint, api_key=secret) as client:
            await self.run_agent(client, ctx.model, messages, data, trace)
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


def _local(taskset: vf.TasksetConfig) -> vf.TasksetConfig:
    """``taskset`` with its id set to the importable alias of its module.

    Verifiers resolves a taskset (and the harness it exports) by a plain module
    name; an overlay's ``models.<pkg>.<mod>`` is dotted, which its lookup
    rejects. torchtitan registers the alias here and again in the env-server
    process it spawns.

    The module must also export a Harness: Verifiers falls back to its built-in
    ``bash`` harness for a taskset that ships none, which hands the model a
    shell on the island -- never the intent here, so refuse instead."""
    module = sys.modules[type(taskset).__module__]
    exported = [getattr(module, name, None) for name in getattr(module, "__all__", ())]
    if not any(isinstance(o, type) and issubclass(o, vf.Harness) for o in exported):
        raise ValueError(
            f"{module.__name__} must export its AgentHarness subclass in __all__ "
            "next to the Taskset (Verifiers would otherwise run its bash harness)"
        )
    return taskset.model_copy(
        update={"id": register_local_taskset_alias(type(taskset).__module__)}
    )


def verifiers_rollouter(
    taskset: vf.TasksetConfig,
    *,
    max_rollout_tokens: int,
    validation_taskset: vf.TasksetConfig | None = None,
    max_turns: int | None = None,
    num_env_workers: int = 1,
    max_concurrent: int | None = None,
    error_reward: float = 0.0,
) -> VerifiersRollouter.Config:
    """A rollouter whose episodes run in your taskset's harness.

    ``taskset`` is your ``vf.TasksetConfig`` subclass instance; its module must
    export the Taskset and the AgentHarness. ``max_rollout_tokens`` bounds a
    prompt (the whole conversation so far) -- keep it within the preset's
    ``seq_len``. ``validation_taskset`` defaults to ``taskset`` in fixed order.
    ``num_env_workers`` / ``max_concurrent`` size Verifiers' env-server pool:
    episodes run concurrently on asyncio inside each worker, so raise them only
    when the agent loop itself is CPU-bound.
    """
    train = _local(taskset)
    validation = _local(validation_taskset or taskset)
    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(verifiers_taskset=train, seed=42),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=validation, seed=99, shuffle=False
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=vf.SingleAgentEnvConfig(
                agent=vf.AgentConfig(
                    # The harness runs in-process, but every episode still gets
                    # a runtime; the default (Prime) would provision a cloud box.
                    runtime=vf.SubprocessConfig(),
                    max_turns=max_turns,
                    # None = the taskset module's own harness.
                    harness=None,
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=num_env_workers),
                address="tcp://127.0.0.1:0",
                max_concurrent=max_concurrent,
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=error_reward,
        ),
        generation_server=GenerationServer.Config(max_rollout_tokens=max_rollout_tokens),
    )
