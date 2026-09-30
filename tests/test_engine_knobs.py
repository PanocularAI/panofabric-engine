"""The engine-owned CLI knobs (panoengine.train.config) through torchtitan's own
ConfigManager, i.e. exactly the argv controld builds."""
import pytest

from torchtitan.config import ConfigManager

from panoengine.train.config import apply_cli_knobs

BASE = ["--module", "models.llama3", "--config", "llama3_debugmodel"]


def _parse(*flags):
    return apply_cli_knobs(ConfigManager().parse_args([*BASE, *flags]))


def test_defaults_leave_the_preset_alone_and_checkpointing_off():
    cfg = _parse()
    assert cfg.checkpointer is None
    single = cfg.dataloader.dataset.dataset
    assert single.source.load_dataset_kwargs["data_files"] == "tests/assets/c4_test/data.json"


def test_every_knob_lands_in_the_upstream_field():
    cfg = _parse(
        "--dataset=c4", "--dataset_path=/data/c4", "--lr=1e-5",
        "--enable_checkpoint", "--checkpointer.folder=checkpoint/run/replica-0",
    )
    source = cfg.dataloader.dataset.dataset.source
    assert source.path == "/data/c4" and source.name == "en"  # c4's own config kept
    assert "data_files" not in source.load_dataset_kwargs
    assert [o.lr for o in cfg.optimizer.optimizers] == [1e-5]
    assert cfg.checkpointer.folder == "checkpoint/run/replica-0"


def test_a_path_alone_replaces_data_files():
    source = _parse("--dataset_path=./assets/c4_test").dataloader.dataset.dataset.source
    assert source.path == "./assets/c4_test" and source.load_dataset_kwargs == {}


def test_unknown_dataset_is_refused():
    with pytest.raises(ValueError, match="known datasets"):
        _parse("--dataset=c5")


def test_lora_refuses_targets_that_match_nothing():
    from models.lora.config_registry import apply_lora, lora_qwen3_0_6b

    cfg = lora_qwen3_0_6b()
    cfg.lora_target_modules = "wq,wo"
    with pytest.raises(ValueError, match=r"renamed upstream: \{'wq': 'wqkv'\}"):
        apply_lora(cfg)
