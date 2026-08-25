"""The training machinery the recipes compose — not the recipes themselves.

The recipes live in the top-level ``models`` package, one per model
(``models.llama3``, ``models.lora``, ``models.resnet``, ...). That is the path a
run spec stores and the one torchtitan's ConfigManager resolves for ``--module``.

What is here is what a recipe builds ON:

* ``strategies`` — ``semi_sync()``, the shared fault-tolerance block every recipe
  drops into its ``FaultTolerantTrainer.Config``. One default, because the
  strategy is chosen at launch, not at preset-construction time; read that
  module's docstring before adding a per-strategy factory.
* ``rl`` — decentralized RL post-training: the presets, the four replica
  coordination strategies, the Monarch trainer actors and their launch entry
  points. It builds on ``torchtitan.experiments.rl``, imported from the fork.

The dependency direction is one-way: a recipe imports from here, nothing here
imports a recipe. The decentralized algorithms themselves (async DiLoCo, HeLoCo,
the parameter server, relay and rollout queue) are a level up again, in
``panoengine.decentralized``, which needs neither torchtitan nor vLLM.
"""
