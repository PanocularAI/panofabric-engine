"""RL post-training: the presets, and the decentralized coordinators they name.

The RL *implementation* is NOT here. It stays in the torchtitan fork, at
``torchtitan.experiments.rl``, and is imported from there — it was briefly
vendored into this package and that copy was deleted, because duplicating a
directory the fork already carries means every upstream rebase improves one copy
and not the other. See FORK-DELTA.md.

What lives here is what builds on it:

* ``config_registry`` — the presets. Reachable as ``--module decentralized_rl``
  (a stored-spec shorthand torchtitan's ConfigManager resolves through its
  experiment registry, which the fork answers with a re-export shim), or by its
  own fully-qualified path.
* ``controller`` — ``RLTrainer`` (synchronous orchestration over torchtitan's
  ``Controller``) and ``RLControllerMixin``: the outer train loop, the
  sync_every-step window runner, and the optional LlamaRL-style
  generation/training overlap. The strategies plug into its hooks.
* ``replicas`` — the four coordination strategies:

  - ``DiLoCoRLReplica`` — N workers sync through a torchft Manager/Lighthouse
    quorum, stock synchronous DiLoCo.
  - ``HeLoCoRLReplica`` — N workers sync pseudo-gradients through a standalone
    CPU parameter server with no barrier (client ``HeLoCoRLClient``; server
    ``python -m panoengine.decentralized.parameter_server``).
  - ``AsyncInferenceReplica`` — decoupled generation: one pure-learner trainer
    broadcasts weights outward through a relay-server tier to independent 
    ``AsyncInferenceWorker`` processes (``worker``, SHARDCAST-style); workers 
    push rollouts into a standalone queue process.
  - ``HeLoCoAsyncInferenceReplica`` — both combined: N pure-learner HeLoCo
    trainers share one rollout queue and coordinate through the parameter
    server (run with ``--relay_addr`` so it publishes global theta to the relay
    for the generator pool).

* ``actors`` — the strategies' Monarch trainer actors (``DiLoCoManagerTrainer``,
  ``HeLoCoPolicyTrainer``, ``SnapshotPolicyTrainer``), each subclassing
  torchtitan's ``PolicyTrainer``.
* ``train`` / ``worker`` — launch entry points. ``python -m
  panoengine.train.rl.train`` starts a worker for any strategy (the --config
  picks it, mirroring ``torchtitan.experiments.rl.train``); ``worker`` is the
  async-inference generator role, which spawns no trainer.

Those classes subclass torchtitan's experimental ``PolicyTrainer`` and
``Controller`` across the package boundary. They override nothing — every method
they define is an addition — so an ordinary cross-package subclass is all they
need. The cost is that upstream's experimental-API drift surfaces here; it is
bounded by this repo's SHA pin on the fork.

Deliberately no re-exports: importing a coordinator pulls the RL stack including
vLLM (~10s), and nothing should pay that just for importing the package. Import
classes from their defining submodule.

The CPU-only processes — parameter server, relay server, rollout queue — are NOT
here; they live in ``panoengine.decentralized``, which needs neither torchtitan
nor vLLM. Nothing here may be imported from them.
"""
