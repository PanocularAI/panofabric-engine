# Compatibility shim. The recipe moved to `panoengine.train.pretrain.hf_transformers`;
# `models.hf_transformers` stays importable because it is a runspec default and lives in
# stored run specs (panofabric spec/runspec.py). See specs/engine-repo-unification-plan.md §1.2.
from panoengine.train.pretrain.hf_transformers import *  # noqa: F401,F403
