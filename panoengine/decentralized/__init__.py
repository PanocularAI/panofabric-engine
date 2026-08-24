"""Decentralized-training algorithms: async DiLoCo and HeLoCo.

These need only torch and torchft — NOT torchtitan. That is what keeps the
dependency direction one-way (see the plan's §8.2): the torchtitan fork's RL
adapter can depend on this package without a cycle.
"""
