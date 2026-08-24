"""The training surface.

Each recipe package holds a ``config_registry`` whose zero-arg functions return
a fully-assembled training config — trainer, optimizer, loss, LR schedule,
dataloader, activation-checkpoint policy, fault-tolerance manager and
checkpoint manager. torchtitan's ``ConfigManager`` selects one with
``--module panoengine.train.<group>.<pkg> --config <fn>``.

Groups:

* ``pretrain``  — llama3, qwen3, gpt_oss (torchtitan-native), and
  hf_transformers (any HF architecture through the transformers backend).
* ``finetune``  — lora. Parameter-efficient recipes. NOTE: these train on raw
  text (c4), so they are LoRA continued-pretraining, not instruction tuning.
  Instruction tuning needs the chat-template / prompt-loss-masking data path
  that does not exist yet (specs/engine-repo-unification-plan.md §1.5).
* ``recipes``   — resnet, the non-LLM reference recipe.

The decentralized strategy (solo / diloco / heloco / async) is NOT fixed by the
recipe: presets carry a default and the launcher overrides it with
``--fault_tolerance.semi_sync_method=...``. See ``panoengine.train.strategies``.
"""
