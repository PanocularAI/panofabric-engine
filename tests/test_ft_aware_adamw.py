"""A centralized run must get a PLAIN optimizer, not the torchft container.

The FT trainer picks TorchFTOptimizersContainer purely from the config type, and that
container's __init__ reads ft_manager.manager -- which asserts when fault tolerance is
off. So a preset hardcoding default_ft_adamw cannot run centralized at all.
"""
import sys

import pytest

from panoengine.train.strategies import adamw
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer

FLAG = "--fault_tolerance.no_enable"


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["torchtitan.train", "--module", "models.qwen3"])


def test_centralized_run_gets_a_plain_container():
    sys.argv.append(FLAG)
    cfg = adamw(lr=8e-4)
    assert not isinstance(cfg, TorchFTOptimizersContainer.Config)
    assert isinstance(cfg, OptimizersContainer.Config)


def test_fault_tolerant_run_keeps_the_ft_container():
    sys.argv.append("--fault_tolerance.enable")
    assert isinstance(adamw(lr=8e-4), TorchFTOptimizersContainer.Config)


def test_standalone_launch_keeps_the_ft_container():
    """No FT flag at all (./run_train.sh) -> the preset's semi_sync() block stands."""
    assert isinstance(adamw(lr=8e-4), TorchFTOptimizersContainer.Config)


def test_lr_and_kwargs_reach_both_paths():
    for extra in ([FLAG], ["--fault_tolerance.enable"]):
        sys.argv[:] = ["torchtitan.train", *extra]
        group = adamw(lr=1.5e-4, eps=1e-9).param_groups[0]
        assert group.optimizer_kwargs["lr"] == 1.5e-4
        assert group.optimizer_kwargs["eps"] == 1e-9
