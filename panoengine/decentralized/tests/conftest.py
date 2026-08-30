"""Test-suite default: bitwise-fp32 wire.

The legacy suites assert BITWISE equality between worker and server params
(torch.equal) — a deliberate contract that catches real algorithm bugs. The
bf16 wire (default-on in production) rounds every exchange by one bf16 ulp,
which is invisible to training but fails every exact assert. Rather than
loosen those contracts to rtol=8e-3 (which would also mask genuine drift),
tests run with the wire pinned to fp32; the bf16 path has its own suite
(bf16_wire_test.py), which opts back in explicitly.
"""
import pytest

from panoengine.decentralized import async_diloco, parameter_server


@pytest.fixture(autouse=True)
def _fp32_wire_by_default(monkeypatch):
    # Both inits: HeLoCoRLClient overrides __init__ entirely (it skips
    # super().__init__), so patching the parent alone misses it.
    for cls in (async_diloco.AsyncDiLoCo, parameter_server.HeLoCoRLClient):
        orig = cls.__init__

        def init(self, *args, __orig=orig, **kwargs):
            kwargs.setdefault("wire_bf16", False)
            return __orig(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", init)
