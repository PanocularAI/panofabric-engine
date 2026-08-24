# Compatibility shim — see ./__init__.py. torchtitan's ConfigManager resolves a
# preset with getattr(module, config_name), which a star-import re-export answers.
from panoengine.train.finetune.lora.config_registry import *  # noqa: F401,F403
