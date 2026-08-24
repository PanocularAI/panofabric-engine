"""Compatibility shim package. The recipes live in ``panoengine.train`` now.

``models.<pkg>`` cannot simply be renamed: it is the default value of
``RunSpec.model.module`` in the panofabric control plane, it is validated there
against the pattern ``models\\.<pkg>`` (spec/runspec.py), and it is baked into
every run spec already stored in the database. So each ``models/<pkg>/`` here is
a two-line re-export of its new home, and the package stays until nothing emits
these paths any more (specs/engine-repo-unification-plan.md §1.2, §4).

New code should import from ``panoengine.train.*`` directly.
"""

_supported_models = frozenset(
    ["llama3", "gpt_oss", "qwen3", "resnet", "hf_transformers", "lora"]
)

# Where each shim forwards to — the single source of truth for the mapping, so
# tests can assert the shims and the real recipes stay in agreement.
_canonical_modules = {
    "llama3": "panoengine.train.pretrain.llama3",
    "qwen3": "panoengine.train.pretrain.qwen3",
    "gpt_oss": "panoengine.train.pretrain.gpt_oss",
    "hf_transformers": "panoengine.train.pretrain.hf_transformers",
    "lora": "panoengine.train.finetune.lora",
    "resnet": "panoengine.train.recipes.resnet",
}
