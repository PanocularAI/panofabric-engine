"""Transitional shim: `python -m panoserve.weights` -> `panoengine.serve.weights`.

The control plane launches this module by name ON THE NODE, inside the image
(panofabric spec/islands.py, spec/coordination.py). The image and the control
plane are separate artifacts with separate lifecycles, and controld's deploy
strategy kills in-flight runs, so the two cannot be required to land in the same
instant. This shim makes the engine answer to BOTH names for one transition.

Delete once no live image and no stored run emits panoserve.* — see
specs/engine-repo-unification-plan.md §1.6.
"""

import runpy

runpy.run_module("panoengine.serve.weights", run_name="__main__", alter_sys=True)
