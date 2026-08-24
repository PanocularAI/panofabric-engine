# Compatibility shim. The recipe moved to `panoengine.train.recipes.resnet`;
# `models.resnet` stays importable because it is a runspec default and lives in
# stored run specs (panofabric spec/runspec.py). See specs/engine-repo-unification-plan.md §1.2.
from panoengine.train.recipes.resnet import *  # noqa: F401,F403
