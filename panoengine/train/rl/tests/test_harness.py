# Copyright (c) Panocular AI.
#
# verifiers_rollouter: the bring-your-own-harness rollouter. The episode itself
# needs a generator and is exercised on GPU (examples/rl_tool_agent).

import sys
import textwrap

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf  # noqa: E402

from panoengine.train.rl.harness import verifiers_rollouter  # noqa: E402

_TASKSET = """
import verifiers.v1 as vf
from panoengine.train.rl.harness import AgentHarness

class LookupTask(vf.Task[vf.TaskData]):
    pass

class LookupConfig(vf.TasksetConfig):
    pass

class LookupTaskset(vf.Taskset[LookupTask, LookupConfig]):
    def load(self):
        return [LookupTask(vf.TaskData(idx=0, prompt="hi"), self.config.task)]

class LookupHarness(AgentHarness):
    pass

__all__ = {exports}
"""


def _module(tmp_path, monkeypatch, name, exports):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_TASKSET.format(exports=exports)))
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__(name)
    return sys.modules[name]


def test_rollouter_runs_the_tasksets_own_harness_locally(tmp_path, monkeypatch):
    mod = _module(tmp_path, monkeypatch, "pf_lookup_ok", '["LookupTaskset", "LookupHarness"]')
    cfg = verifiers_rollouter(mod.LookupConfig(), max_rollout_tokens=2048)
    agent = cfg.verifiers_env_server.environment.agent
    # Unset = the taskset module's own harness; the runtime is local, never
    # Verifiers' default cloud sandbox.
    assert agent.harness is None
    assert isinstance(agent.runtime, vf.SubprocessConfig)
    assert cfg.train_dataset.verifiers_taskset.id == "pf_lookup_ok"
    assert cfg.validation_dataset.shuffle is False
    assert cfg.generation_server.max_rollout_tokens == 2048


def test_rollouter_refuses_a_taskset_without_a_harness(tmp_path, monkeypatch):
    # Verifiers would silently fall back to its bash harness: a shell for the model.
    mod = _module(tmp_path, monkeypatch, "pf_lookup_bare", '["LookupTaskset"]')
    with pytest.raises(ValueError, match="AgentHarness"):
        verifiers_rollouter(mod.LookupConfig(), max_rollout_tokens=2048)
