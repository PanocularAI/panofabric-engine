"""panoengine — decentralized training and serving for large models.

Two halves, one engine:

* ``panoengine.train`` — training recipes for pretraining, LoRA fine-tuning and
  RL, each of which gets DiLoCo / HeLoCo / async decentralized training for free
  because every recipe is a ``FaultTolerantTrainer.Config``.
* ``panoengine.serve`` — the spliced-pipeline inference plane.

Recipes compose, implementations subclass: everything here composes public
classes from the torchtitan and torchft forks. Code that has to subclass or
patch upstream internals lives in those forks instead.
"""
