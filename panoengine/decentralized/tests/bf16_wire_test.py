"""bf16 wire format, NEGOTIATED (2026-08-30).

The links between the two showcase clusters measured ~1 Gbps/site with the HTTP path
already at 80-85% of link speed, so bytes convert 1:1 into boundary time: fp32 both ways
was 4.77 GB and 40-80 s per exchange. bf16 halves both directions. It must be negotiated,
not assumed -- a bf16 push to a pre-bf16 server 500s, and _step_post_hook's catch-all
would then silently drop the window.
"""
import json
import urllib.request
from unittest import TestCase

import torch
import torch.nn as nn
import torch.optim as optim

from panoengine.decentralized.async_diloco import (
    AsyncDiLoCo,
    AsyncDiLoCoServer,
    _bf16_bytes,
    _bytes_to_tensor,
    _read_exact,
    _tensor_to_bytes,
)
from panoengine.decentralized.tests.async_diloco_test import (
    _make_model,
    _total_numel,
)


def _raw_sync(addr, model, *, accept_bf16, upload_dtype=None, grad_value=1.0):
    """One raw-protocol exchange, controlling both sides of the negotiation.
    upload_dtype=None -> pull only (flag=0, no body): reads a snapshot WITHOUT
    advancing the revision, so two fetches compare the same state."""
    total = _total_numel(model)
    push = upload_dtype is not None
    header = {"flag": 1 if push else 0, "speed": 1.0, "baseline_revision": 0}
    if accept_bf16:
        header["accept_dtype"] = "bfloat16"
    body = b""
    if push:
        header["numel"] = total
        flat = torch.full((total,), grad_value)
        header["dtype"] = upload_dtype
        body = (_bf16_bytes(flat) if upload_dtype == "bfloat16"
                else _tensor_to_bytes(flat))
    req = urllib.request.Request(
        addr, data=(json.dumps(header) + "\n").encode() + body,
        headers={"Content-Type": "application/octet-stream"}, method="POST")
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        rh = json.loads(resp.readline(1 << 16))
        n = int(rh["numel"])
        if rh.get("dtype") == "bfloat16":
            params = _bytes_to_tensor(_read_exact(resp, n * 2),
                                      torch.bfloat16).to(torch.float32)
        else:
            params = _bytes_to_tensor(_read_exact(resp, n * 4), torch.float32)
    return rh, params


class TestBf16Wire(TestCase):
    def _server(self, **kw):
        model = _make_model()
        return AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1),
                                 port=0, **kw), model

    def test_bf16_download_matches_fp32_within_one_ulp(self):
        """Same revision fetched both ways: the bf16 body must be the fp32 body
        rounded, never garbage from a mis-sized read."""
        server, model = self._server()
        rh32, p32 = _raw_sync(server.address(), model, accept_bf16=False)
        rh16, p16 = _raw_sync(server.address(), model, accept_bf16=True)
        self.assertEqual(rh32["dtype"], "float32")
        self.assertEqual(rh16["dtype"], "bfloat16")
        torch.testing.assert_close(p16, p32, rtol=8e-3, atol=1e-6)

    def test_bf16_upload_applies_like_fp32(self):
        """A bf16 pseudo-gradient must produce the same outer step as fp32 (the
        fill value 1.0 is exactly representable in bf16)."""
        torch.manual_seed(7)
        s32, m32 = self._server()
        torch.manual_seed(7)
        s16, m16 = self._server()
        _raw_sync(s32.address(), m32, accept_bf16=False, upload_dtype="float32")
        _raw_sync(s16.address(), m16, accept_bf16=True, upload_dtype="bfloat16")
        for (n, a), (_, b) in zip(m32.named_parameters(), m16.named_parameters()):
            torch.testing.assert_close(a, b, msg=f"divergence in {n}")

    def test_bf16_upload_streaming_path(self):
        """grace_period=0 servers read the body via _stream_body_into_bufs -- the
        production path in every showcase run -- which needs its own bf16 branch."""
        server, model = self._server(grace_period=0.0)
        rh, _ = _raw_sync(server.address(), model, accept_bf16=True,
                          upload_dtype="bfloat16")
        self.assertTrue(rh["applied"])

    def test_old_client_still_gets_fp32(self):
        """No accept_dtype in the header (an old worker) -> fp32 body, and the
        dtype key still present so the response is self-describing."""
        server, model = self._server()
        rh, _ = _raw_sync(server.address(), model, accept_bf16=False)
        self.assertEqual(rh["dtype"], "float32")

    def test_worker_never_uploads_bf16_blind(self):
        """A fresh AsyncDiLoCo must default to fp32 uploads until a response has
        NAMED its dtype -- a blind bf16 push to an old server is a dropped window."""
        model = _make_model()
        worker = AsyncDiLoCo(
            "http://127.0.0.1:9/sync",  # never dialed
            model, optim.SGD(model.parameters(), lr=0.1), sync_every=1,
            wire_bf16=True)
        self.assertFalse(worker._server_bf16)

    def test_worker_learns_capability_from_the_response(self):
        """End-to-end through the real worker: after its initial pull from a
        bf16-capable server, uploads switch to bf16 and the model still syncs."""
        server, smodel = self._server()
        model = _make_model()
        worker = AsyncDiLoCo(server.address(), model,
                             optim.SGD(model.parameters(), lr=0.1),
                             sync_every=1, wire_bf16=True)
        with worker:
            self.assertTrue(worker._server_bf16,
                            "initial pull must set the capability")
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            worker._inner_optimizer.step()
        # server-side model must have taken the (bf16) push
        self.assertGreaterEqual(server.status()["applied_pushes"], 0)


class TestWireEnvEscape(TestCase):
    def test_env_disables_the_compressed_wire(self):
        """The fp32 control arm of the convergence study: islands reach the
        worker constructor only through their env, so PF_WIRE_BF16=0 must pin
        the wire to bitwise fp32 (bf16 AND deltas both ride this flag)."""
        import os
        from unittest import mock
        model = _make_model()
        with mock.patch.dict(os.environ, {"PF_WIRE_BF16": "0"}):
            w = AsyncDiLoCo("http://127.0.0.1:9/sync", model,
                            optim.SGD(model.parameters(), lr=0.1),
                            sync_every=1, wire_bf16=None)
        self.assertFalse(w._wire_bf16)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PF_WIRE_BF16", None)
            w2 = AsyncDiLoCo("http://127.0.0.1:9/sync", model,
                             optim.SGD(model.parameters(), lr=0.1),
                             sync_every=1, wire_bf16=None)
        self.assertTrue(w2._wire_bf16)
