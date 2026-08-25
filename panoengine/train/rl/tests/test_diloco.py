# Copyright (c) Panocular AI.
#
# DiLoCoRLReplica / DiLoCoManagerTrainer: the pieces that don't need a real
# torchft Manager or Lighthouse to exercise.

import asyncio
from types import SimpleNamespace

import torch
from torch import nn

from panoengine.train.rl.actors import DiLoCoManagerTrainer
from panoengine.train.rl.replicas import DiLoCoRLReplica


def _ep(fn):
    """A Monarch-endpoint stand-in: exposes .call(...)."""
    return SimpleNamespace(call=fn)


def test_window_sync_reports_manager_step_info():
    replica = object.__new__(DiLoCoRLReplica)
    replica._get_rank_0_value = lambda x: x

    async def diloco_step_info():
        return {"current_step": 3, "num_participants": 2}

    replica.trainer = SimpleNamespace(diloco_step_info=_ep(diloco_step_info))

    detail = asyncio.run(replica._window_sync(t0=0.0))
    assert detail == "diloco_step=3 participants=2"


def test_close_swallows_diloco_teardown_errors(monkeypatch):
    """A failed torchft Manager shutdown must not prevent the base RLTrainer
    cleanup (actor/mesh teardown) from running."""
    from panoengine.train.rl.controller import RLTrainer

    base_close_called = []

    async def fake_base_close(self):
        base_close_called.append(True)

    monkeypatch.setattr(RLTrainer, "close", fake_base_close)

    replica = object.__new__(DiLoCoRLReplica)
    replica.config = SimpleNamespace(replica_id=0)

    async def close_diloco():
        raise RuntimeError("manager shutdown failed")

    replica.trainer = SimpleNamespace(close_diloco=_ep(close_diloco))

    asyncio.run(replica.close())
    assert base_close_called == [True]


def test_diloco_state_dict_round_trip():
    """The Manager's recovery path (a replica that falls behind after a
    failed all-reduce) restores exactly the model + inner-optimizer state
    _diloco_state_dict captured -- not a full trainer checkpoint."""
    actor = object.__new__(DiLoCoManagerTrainer)
    actor.model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.1, momentum=0.9)
    actor.optimizers = SimpleNamespace(optimizers=[optimizer])

    # Take a step so the optimizer has real momentum state to round-trip.
    actor.model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    snapshot = actor._diloco_state_dict()
    assert set(snapshot) == {"model", "inner_optim"}

    # Diverge, then restore from the snapshot.
    with torch.no_grad():
        actor.model.weight.fill_(0.0)
    actor._diloco_load_state_dict(snapshot)

    assert torch.equal(actor.model.state_dict()["weight"], snapshot["model"]["weight"])
    assert optimizer.state_dict()["state"][0]["momentum_buffer"] is not None
