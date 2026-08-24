"""RL post-training recipes.

The presets live here (they only COMPOSE public classes). The trainer
implementations they name — the Monarch actors and the replica/controller
classes — stay in the torchtitan fork, because they subclass upstream's own
experimental PolicyTrainer and Controller. See FORK-DELTA.md.

Selected as ``--module decentralized_rl`` (a stored-spec shorthand torchtitan
resolves through its experiment registry; the fork keeps a re-export shim).
"""
