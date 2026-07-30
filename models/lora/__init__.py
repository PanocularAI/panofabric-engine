"""FT glue for LoRA fine-tuning of pretrained models (torchtitan LoRAConverter).

Selected from a RunSpec via model.module: models.lora. Each preset in
.config_registry pins one pretrained model: the architecture comes from
torchtitan's native model registry (which has the HF<->DCP state-dict adapter
LoRA fine-tuning needs for weight loading); the weights come from the HF repo
the panofabric control plane's LORA_PRESET_REPOS map pairs with the preset
(the spec layer validates submissions against that map and auto-fetches the
weights into hf_assets_path before launch — preset names here and map keys
there must match 1:1).

The torchtitan-heavy presets live in .config_registry, imported lazily by
torchtitan's ConfigManager.
"""
