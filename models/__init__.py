"""The training recipes: one package per model, each with a ``config_registry``.

``models.<pkg>`` is the canonical path — it is the default value of
``RunSpec.model.module`` in the panofabric control plane, it is what that plane
validates against (``^models\\.<pkg>$``, spec/runspec.py) and stores in every run
spec, and it is what torchtitan's ConfigManager resolves for ``--module``. The
recipes live here rather than under ``panoengine.train`` so that path is the real
one instead of a re-export of it.

Each ``models/<pkg>/`` holds whatever that recipe needs: a ``model_registry``
returning a ``FaultTolerantModelSpec`` (``__init__.py``), the zero-arg preset
functions (``config_registry.py``), and for non-LLM recipes the model itself
(see ``resnet/``, which carries its own model, loss, dataloader and parallelize
plan).

What stays in ``panoengine.train`` is the machinery every recipe composes, not
the recipes: ``strategies.semi_sync`` (the shared fault-tolerance block) and
``rl`` (the decentralized RL coordinators). A recipe imports from there; nothing
there imports a recipe.
"""

#: The recipes this repo ships. Also the guard the control plane uses to keep a
#: tenant code overlay from shadowing a built-in (spec/runspec.py's
#: _ENGINE_BUILTIN_PKGS mirrors it).
_supported_models = frozenset(
    ["llama3", "gpt_oss", "qwen3", "resnet", "hf_transformers", "lora"]
)
