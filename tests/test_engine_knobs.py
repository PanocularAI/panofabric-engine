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


def _training(*flags, module="models.llama3", config="llama3_debugmodel"):
    return ConfigManager().parse_args(["--module", module, "--config", config, *flags])


def test_batch_shape_knobs_are_sequences_of_seq_len_tokens():
    # llama3_debugmodel: 8 sequences x 2048 tokens.
    t = apply_cli_knobs(_training("--local_batch_size=2", "--global_batch_size=16",
                                  "--seq_len=4096")).training
    assert t.num_tokens_per_microbatch_per_dp_rank == 2 * 4096
    assert t.num_tokens_per_train_step == 16 * 4096
    assert t.max_context_length == 4096


def test_seq_len_alone_keeps_the_presets_batch_in_sequences():
    t = apply_cli_knobs(_training("--seq_len=4096")).training
    assert t.num_tokens_per_microbatch_per_dp_rank == 8 * 4096  # was 8 x 2048


def test_a_longer_seq_len_grows_the_models_rope_cache():
    from torchtitan.protocols.module import Module

    cfg = apply_cli_knobs(_training("--seq_len=8192"))
    lengths = {c.max_context_length for _, c, *_ in cfg.model.traverse(Module.Config, recurse=True)
               if hasattr(c, "max_context_length")}
    assert lengths == {8192} and cfg.model.max_context_length == 8192


def test_resnet_counts_images_not_image_side_tokens():
    t = apply_cli_knobs(_training("--local_batch_size=32", module="models.resnet",
                                  config="resnet18_cifar10")).training
    assert t.num_tokens_per_microbatch_per_dp_rank == 32


def test_rl_lr_reaches_every_trainer_optimizer():
    pytest.importorskip("vllm")
    from panoengine.train.rl.controller import apply_lr

    cfg = ConfigManager().parse_args(["--module", "panoengine.train.rl.config_registry",
                                      "--config", "rl_heloco_qwen3_0_6b", "--lr=5e-6"])
    apply_lr(cfg)
    assert [o.lr for o in cfg.trainer.optimizer.optimizers] == [5e-6]
