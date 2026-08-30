"""Delta+int8 downloads.

After bf16 the boundary floor is ~38 s, of which ~10 s is still the download
(1.19 GB at the measured ~1 Gbps/site). A worker already names the revision it
holds, so the server ships only the int8-quantized CHANGE since then (~0.6 GB):
a delta is small-magnitude and zero-centered — the same signal class as the
pseudo-gradient upload, which already travels int8. Negotiated per exchange
(`accept_delta` offer, `dtype: delta_int8` reply), with full-download fallbacks
whenever the baseline left the server's ring, the worker holds no whole
baseline, or a refresh round is due (drift from chained quantized deltas is
bounded by _delta_refresh_every, reset by any full download).
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
)
from panoengine.decentralized.tests.async_diloco_test import (
    _make_model,
    _total_numel,
    push_pull,
)


def _worker(server, model, **kw):
    kw.setdefault("wire_bf16", True)   # the conftest pins False for legacy suites
    return AsyncDiLoCo(server.address(), model,
                       optim.SGD(model.parameters(), lr=0.1),
                       sync_every=1, **kw)


def _one_window(worker, model):
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    worker._inner_optimizer.step()


class TestDeltaDownload(TestCase):
    def _server(self, **kw):
        model = _make_model()
        return AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1),
                                 port=0, **kw), model

    def test_worker_tracks_server_through_delta_syncs(self):
        """The end-to-end contract: after several sync windows (all deltas past
        the first), the worker's adopted baseline matches the server's master
        within quantization tolerance."""
        server, smodel = self._server()
        model = _make_model()
        with _worker(server, model) as w:
            for _ in range(4):
                _one_window(w, model)
        flat_server = torch.cat(
            [p.detach().reshape(-1).float() for p in smodel.parameters()])
        torch.testing.assert_close(w._baseline_flat(), flat_server,
                                   rtol=0, atol=2e-2)
        self.assertGreater(w._deltas_since_full, 0,
                           "at least one exchange must actually have been a delta")

    def test_first_exchange_is_never_a_delta(self):
        """No adopted baseline yet -> the initial pull must not offer a delta."""
        server, _ = self._server()
        model = _make_model()
        w = _worker(server, model)
        self.assertFalse(w._have_baseline)
        with w:
            self.assertTrue(w._have_baseline, "initial pull adopts a baseline")
            self.assertEqual(w._deltas_since_full, 0)

    def test_evicted_baseline_falls_back_to_full(self):
        """A baseline older than the server's ring must yield a FULL download,
        not an error: advance the server past the ring, then sync."""
        server, _ = self._server()
        model = _make_model()
        with _worker(server, model) as w:
            for _ in range(server._served_max + 2):   # evict w's baseline
                push_pull(server.address(), model, grad_value=0.01)
            _one_window(w, model)                     # stale revision offer
            self.assertEqual(w._deltas_since_full, 0,
                             "the fallback full download resets the counter")

    def test_refresh_round_requests_full(self):
        """Every _delta_refresh_every-th exchange must go full, bounding drift."""
        server, smodel = self._server()
        model = _make_model()
        with _worker(server, model) as w:
            w._delta_refresh_every = 2
            fulls = 0
            for _ in range(5):
                before = w._deltas_since_full
                _one_window(w, model)
                fulls += w._deltas_since_full == 0 and before != 0
            # the counter may legitimately SIT at refresh_every (that is the
            # "due" state); what must hold is that refreshes actually happened
            self.assertLessEqual(w._deltas_since_full, 2)
            self.assertGreaterEqual(fulls, 1, "no refresh round ever went full")
        flat_server = torch.cat(
            [p.detach().reshape(-1).float() for p in smodel.parameters()])
        torch.testing.assert_close(w._baseline_flat(), flat_server,
                                   rtol=0, atol=2e-2)

    def test_fp32_wire_never_offers_delta(self):
        """wire_bf16=False means a bitwise-fp32 wire, deltas included."""
        server, smodel = self._server()
        model = _make_model()
        with _worker(server, model, wire_bf16=False) as w:
            for _ in range(3):
                _one_window(w, model)
            self.assertEqual(w._deltas_since_full, 0)
        flat_server = torch.cat(
            [p.detach().reshape(-1).float() for p in smodel.parameters()])
        self.assertTrue(torch.equal(w._baseline_flat(), flat_server),
                        "fp32 wire stays BITWISE exact")

    def test_delta_response_is_smaller_than_full(self):
        """The point of the exercise: a delta body is ~1/4 of an fp32 body."""
        server, _ = self._server()
        model = _make_model()
        total = _total_numel(model)
        # prime revision 0 into the ring
        header = {"flag": 0, "speed": 0.0, "baseline_revision": 0,
                  "accept_dtype": "bfloat16"}
        def fetch(hdr):
            req = urllib.request.Request(
                server.address(), data=(json.dumps(hdr) + "\n").encode(),
                headers={"Content-Type": "application/octet-stream"},
                method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                rh = json.loads(resp.readline(1 << 16))
                body = resp.read()
            return rh, body
        fetch(header)                                  # retains rev 0
        rh, body = fetch({**header, "accept_delta": 1})
        self.assertEqual(rh["dtype"], "delta_int8")
        self.assertEqual(rh["delta_from"], 0)
        self.assertLess(len(body), total * 2)          # < the bf16 body even
