# Compatibility shim. The recipe moved to `panoengine.train.finetune.lora`;
# `models.lora` stays importable because it is a runspec default and lives in
# stored run specs (panofabric spec/runspec.py). See specs/engine-repo-unification-plan.md §1.2.
from panoengine.train.finetune.lora import *  # noqa: F401,F403
