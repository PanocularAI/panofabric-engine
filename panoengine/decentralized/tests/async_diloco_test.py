# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import multiprocessing
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn, optim

from panoengine.decentralized.async_diloco import (
    AsyncDiLoCo,
    AsyncDiLoCoServer,
    DelayedNesterovOptimizer,
    _bytes_to_tensor,
    _dequantize_int8,
    _GraceBatch,
    _quantize_int8,
    _read_exact,
)


def _make_model(d: int = 8) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


def _total_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _fp32_client(*args, **kwargs):
    """AsyncDiLoCo with the wire pinned to bitwise fp32. Spawned-child tests must
    use this: the conftest autouse patch that pins the wire for in-process tests
    does not propagate into torch.multiprocessing children, and these suites
    assert exact equality across the exchange."""
    kwargs.setdefault("wire_bf16", False)
    return AsyncDiLoCo(*args, **kwargs)


def push_pull(
    addr: str,
    model: nn.Module,
    full_sync: bool = True,
    speed: float = 1.0,
    grad_value: float = 1.0,
    baseline_revision: int = 0,
    quantize: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int, int, bool]:
    """One raw-protocol sync against an AsyncDiLoCoServer (or subclass): a
    single HTTP POST with a JSON header line plus raw tensor bytes.

    Returns ``(params, new_steps, revision, applied)`` where params is the
    unflattened server response keyed by parameter name.
    """
    total = _total_numel(model)
    header: Dict[str, object] = {
        "flag": 1 if full_sync else 0,
        "speed": speed,
        "baseline_revision": baseline_revision,
    }
    body = b""
    if full_sync:
        flat = torch.full((total,), grad_value)
        header["numel"] = total
        if quantize:
            numels = [p.numel() for _, p in model.named_parameters()]
            q, scales = _quantize_int8(flat, numels)
            header["dtype"] = "int8"
            body = scales.numpy().tobytes() + q.numpy().tobytes()
        else:
            header["dtype"] = "float32"
            body = flat.numpy().tobytes()

    request = urllib.request.Request(
        addr,
        data=(json.dumps(header) + "\n").encode() + body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        resp_header = json.loads(resp.readline())
        numel = int(resp_header["numel"])
        flat_params = torch.frombuffer(
            bytearray(resp.read(numel * 4)), dtype=torch.float32
        )
    assert flat_params.numel() == numel

    params: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        params[name] = flat_params[offset : offset + n].view(p.shape).clone()
        offset += n
    return (
        params,
        int(resp_header["new_steps"]),
        int(resp_header["revision"]),
        bool(resp_header["applied"]),
    )


def push_pull_flat(
    addr, flat, *, quantize, numels, fragment=None, num_fragments=None
):
    """One raw-protocol full-sync push of an ARBITRARY flat pseudo-gradient
    (push_pull only supports a constant fill value). Drains and discards the
    response body; returns the response header dict. ``fragment`` /
    ``num_fragments`` build a fragment push's header (see the server's wire
    format docstring); ``numels`` must then be the fragment's own numels."""
    header = {
        "flag": 1,
        "speed": 1.0,
        "baseline_revision": 0,
        "numel": flat.numel(),
    }
    if fragment is not None:
        header["fragment"] = fragment
        header["num_fragments"] = num_fragments
    if quantize:
        q, scales = _quantize_int8(flat, numels)
        header["dtype"] = "int8"
        body = scales.numpy().tobytes() + q.numpy().tobytes()
    else:
        header["dtype"] = "float32"
        body = flat.numpy().tobytes()
    request = urllib.request.Request(
        addr,
        data=(json.dumps(header) + "\n").encode() + body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        resp_header = json.loads(resp.readline())
        resp.read()
    return resp_header


class TestDelayedNesterovOptimizer(TestCase):
    def test_period1_matches_standard_nesterov(self) -> None:
        """nesterov_period=1: every push is a milestone, should match SGD+Nesterov exactly."""
        lr, beta = 0.1, 0.9
        g = torch.tensor([1.0, 2.0])

        p_ref = torch.tensor([1.0, 1.0])
        m_ref = torch.zeros_like(p_ref)
        m_ref = beta * m_ref + g
        p_ref = p_ref - lr * (g + beta * m_ref)

        p_dn = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        dn = DelayedNesterovOptimizer([p_dn], lr=lr, momentum=beta, nesterov_period=1)
        p_dn.grad = g.clone()
        dn.step()

        torch.testing.assert_close(p_dn.data, p_ref)

    def test_intermediate_steps_update_params(self) -> None:
        """Non-milestone pushes still update params (pure gradient steps, no momentum)."""
        lr, N = 0.1, 3
        g = torch.tensor([1.0])
        p = torch.nn.Parameter(torch.tensor([0.0]))
        dn = DelayedNesterovOptimizer([p], lr=lr, momentum=0.9, nesterov_period=N)
        p_before = p.data.clone()
        p.grad = g.clone()
        dn.step()
        self.assertFalse(torch.equal(p.data, p_before))

    def test_end_to_end_with_server(self) -> None:
        """DN optimizer integrated with AsyncDiLoCoServer updates global params at milestone."""
        d = 8
        global_model = _make_model(d)
        outer_opt = DelayedNesterovOptimizer(
            global_model.parameters(), lr=0.1, momentum=0.9, nesterov_period=2
        )
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        for _ in range(2):
            push_pull(addr, global_model, grad_value=1.0)

        for name, p in global_model.named_parameters():
            self.assertFalse(torch.equal(p.data, initial[name]))


class TestAsyncDiLoCoServer(TestCase):
    def test_server_applies_outer_step(self) -> None:
        """Server receives pseudo-grads and updates global params via outer optimizer."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        new_global, _, revision, applied = push_pull(
            server.address(), global_model, grad_value=1.0
        )

        self.assertTrue(applied)
        self.assertEqual(revision, 1)
        for name, p in global_model.named_parameters():
            torch.testing.assert_close(
                new_global[name], initial[name] - 0.1 * torch.ones_like(p.data)
            )

    def test_concurrent_workers_serialize(self) -> None:
        """Multiple concurrent workers each complete without errors."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        results, errors = [], []

        def worker() -> None:
            try:
                params, _, _, _ = push_pull(addr, global_model, grad_value=0.0)
                results.append(params)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(results), 3)

    def test_revision_increments_per_push(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        _, _, rev1, applied1 = push_pull(server.address(), model)
        _, _, rev2, applied2 = push_pull(server.address(), model)
        self.assertTrue(applied1 and applied2)
        self.assertEqual((rev1, rev2), (1, 2))

    def test_session_cap_returns_503(self) -> None:
        """R7: syncs beyond max_sessions get 503 instead of exhausting
        server threads (monitoring endpoints stay available)."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            max_sessions=0,  # every sync is over capacity
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            push_pull(server.address(), model)
        self.assertEqual(ctx.exception.code, 503)
        # /status is unaffected by the sync session cap.
        with urllib.request.urlopen(server.status_address()) as resp:
            self.assertEqual(resp.status, 200)

    def test_read_exact_is_zero_copy_and_byte_identical(self) -> None:
        """These payloads are whole-model-sized (2.2 GiB fp32 at 0.6B, ~30 GiB at
        8B), so every extra allocation lands on the parameter server's peak RSS —
        two workers OOM'd a 32 GiB cloud hub. _read_exact must fill ONE writable
        buffer that _bytes_to_tensor wraps in place, with the bytes unchanged."""
        want = torch.arange(64, dtype=torch.float32)
        raw = want.numpy().tobytes()

        buf = _read_exact(io.BytesIO(raw), len(raw))
        self.assertIsInstance(buf, bytearray)   # writable => frombuffer won't copy
        self.assertEqual(bytes(buf), raw)       # byte-identical to the wire

        got = _bytes_to_tensor(buf, torch.float32)
        self.assertTrue(torch.equal(got, want))
        # The tensor must ALIAS the buffer, not copy it: mutate the buffer and
        # the tensor sees it. This is the property that saves the copy.
        buf[0:4] = b"\x00\x00\x00\x00"
        self.assertEqual(got[0].item(), 0.0)

    def test_read_exact_handles_chunked_streams_and_short_reads(self) -> None:
        """A socket delivers a multi-GB body in many chunks; readinto must be
        driven to completion, and an early EOF must still raise."""

        class _Dribble(io.RawIOBase):
            """Returns at most 7 bytes per readinto call."""

            def __init__(self, data: bytes) -> None:
                self._data, self._pos = data, 0

            def readable(self) -> bool:
                return True

            def readinto(self, b) -> int:  # type: ignore[no-untyped-def]
                n = min(7, len(b), len(self._data) - self._pos)
                b[:n] = self._data[self._pos : self._pos + n]
                self._pos += n
                return n

        raw = torch.arange(50, dtype=torch.float32).numpy().tobytes()
        self.assertEqual(bytes(_read_exact(_Dribble(raw), len(raw))), raw)

        with self.assertRaises(IOError):
            _read_exact(io.BytesIO(raw[:20]), len(raw))

    def test_busy_503_retries_the_push_instead_of_dropping_it(self) -> None:
        """A session cap is only safe with this retry.

        503 means "all max_sessions slots busy, come back" — not a failure. If it
        escapes, AsyncDiLoCo._step_post_hook's catch-all drops the push and
        re-baselines, silently throwing away the window's pseudo-gradient. So the
        client must wait Retry-After and re-send the SAME push."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0, max_sessions=1
        )
        self.addCleanup(server.shutdown)
        worker = _make_model()
        worker.load_state_dict(model.state_dict())
        real_urlopen = urllib.request.urlopen
        calls: Dict[str, int] = {"n": 0}

        with AsyncDiLoCo(
            server.address(),
            worker,
            optim.SGD(worker.parameters(), lr=0.1),
            sync_every=1,
            busy_retries=5,
        ) as ad:
            # Refuse the first two attempts with a real 503, then let it through.
            def flaky(request, *args, **kwargs):  # type: ignore[no-untyped-def]
                calls["n"] += 1
                if calls["n"] <= 2:
                    raise urllib.error.HTTPError(
                        server.address(), 503, "busy",
                        {"Retry-After": "0"}, None,   # 0 keeps the test fast
                    )
                return real_urlopen(request, *args, **kwargs)

            with patch("urllib.request.urlopen", side_effect=flaky):
                flat, _, revision, applied = ad._session_roundtrip(
                    1.0, 1.0, torch.zeros(_total_numel(worker))
                )

        self.assertEqual(calls["n"], 3, "should have retried twice, then succeeded")
        self.assertTrue(applied, "the retried push must land, not be dropped")
        self.assertEqual(revision, 1)
        self.assertEqual(flat.numel(), _total_numel(worker))

    def test_busy_503_gives_up_after_busy_retries(self) -> None:
        """Bounded, not infinite: once the budget is spent the 503 propagates and
        the existing drop-and-resync path takes over."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0, max_sessions=0
        )
        self.addCleanup(server.shutdown)
        worker = _make_model()
        # max_sessions=0 refuses EVERY push, including __enter__'s initial pull,
        # so drive the client directly rather than through the context manager.
        ad = AsyncDiLoCo.__new__(AsyncDiLoCo)
        ad._server_address = server.address()
        ad._sync_timeout = 5.0
        ad._busy_retries = 2
        ad._total_numel = _total_numel(worker)
        ad._param_numels = [p.numel() for p in worker.parameters()]
        ad._quantize = False
        ad._wire_bf16 = False       # _session_roundtrip reads both bf16 flags
        ad._server_bf16 = False
        ad._baseline_revision = 0
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            ad._session_roundtrip(1.0, 1.0, torch.zeros(_total_numel(worker)))
        self.assertEqual(ctx.exception.code, 503)

    def test_streamed_apply_is_bitwise_identical_to_materialized(self) -> None:
        """THE safety proof for the streaming request path (heloco-ps-memory §6
        step 3): reading a push chunk-by-chunk into the persistent buffers must
        change WHERE bytes land, never their values. Same pushes through HTTP
        (streaming) and through _handle_sync with a materialized flat tensor
        (the old path, still used by grace batching) must yield bitwise-equal
        global params, momentum, and response — across both server classes
        (HeLoCo adds per-block norm/dot corrections that are NOT element-wise,
        which is exactly why streaming reuses the whole materialized dict
        instead of applying chunk-by-chunk) and both wire dtypes."""
        from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer

        def build(kind: str):
            torch.manual_seed(1234)   # identical initial weights per build
            model = _make_model()
            if kind == "heloco":
                opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
                return HeLoCoServer(model, opt, port=0), model
            opt = DelayedNesterovOptimizer(
                model.parameters(), lr=0.1, momentum=0.9, nesterov_period=2
            )
            return AsyncDiLoCoServer(model, opt, port=0), model

        for kind in ("base", "heloco"):
            for quantize in (False, True):
                with self.subTest(kind=kind, quantize=quantize):
                    streamed_srv, streamed_model = build(kind)
                    self.addCleanup(streamed_srv.shutdown)
                    reference_srv, reference_model = build(kind)
                    self.addCleanup(reference_srv.shutdown)

                    torch.manual_seed(99)
                    total = _total_numel(streamed_model)
                    pushes = [torch.randn(total) for _ in range(3)]

                    for flat in pushes:
                        if quantize:
                            # The materialized reference must see the SAME
                            # dequantized values the wire delivers.
                            numels = [
                                p.numel()
                                for p in reference_model.parameters()
                            ]
                            q, scales = _quantize_int8(flat, numels)
                            ref_flat = _dequantize_int8(q, scales, numels)
                        else:
                            ref_flat = flat.clone()
                        reference_srv._handle_sync(
                            is_full_sync=True,
                            worker_speed=1.0,
                            baseline_revision=0,
                            flat_grads=ref_flat,
                        )
                        # HTTP => the streaming read path (grace_period == 0).
                        push_pull_flat(
                            streamed_srv.address(), flat, quantize=quantize,
                            numels=[p.numel() for p in streamed_model.parameters()],
                        )

                    # Global params bitwise equal...
                    for (n, ps), (_, pr) in zip(
                        streamed_model.named_parameters(),
                        reference_model.named_parameters(),
                    ):
                        self.assertTrue(
                            torch.equal(ps.data, pr.data),
                            f"{kind}/{quantize}: params diverged at {n}",
                        )
                    # ...and so is the outer-optimizer momentum.
                    for ps, pr in zip(
                        streamed_model.parameters(),
                        reference_model.parameters(),
                    ):
                        ms = streamed_srv._outer_optimizer.state.get(ps, {}).get("m")
                        mr = reference_srv._outer_optimizer.state.get(pr, {}).get("m")
                        self.assertEqual(ms is None, mr is None)
                        if ms is not None:
                            self.assertTrue(torch.equal(ms, mr))
                    # And the servers agree they applied the same pushes.
                    self.assertEqual(
                        streamed_srv._applied_pushes,
                        reference_srv._applied_pushes,
                    )

    def test_grace_batching_still_uses_private_gradients(self) -> None:
        """grace_period > 0 must keep the materializing path: batching holds
        SEVERAL workers' gradients at once, which the single shared streaming
        buffer cannot represent — routing it through streaming would silently
        merge concurrent workers' pushes into one buffer."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            grace_period=0.2,
        )
        self.addCleanup(server.shutdown)
        results: list = []

        def worker(v: float) -> None:
            results.append(push_pull(server.address(), model, grad_value=v))

        threads = [threading.Thread(target=worker, args=(v,)) for v in (1.0, 2.0)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(len(results), 2)
        # Both pushes applied as distinct gradients: revision advanced by 2.
        self.assertEqual(server._revision, 2)
        # And streaming buffers were never allocated on this server.
        self.assertIsNone(server._stream_bufs)

    def test_stale_baseline_rejected(self) -> None:
        """A push whose baseline revision is ahead of the server (checkpoint-restore
        scenario) must be rejected without touching the global params."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}

        params, _, revision, applied = push_pull(
            server.address(), model, baseline_revision=5
        )

        self.assertFalse(applied)
        self.assertEqual(revision, 0)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.data, initial[name])
            torch.testing.assert_close(params[name], initial[name])


class TestAsyncDiLoCo(TestCase):
    def test_sync_resets_model_to_global(self) -> None:
        """After sync, worker model matches the server's updated global params."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        sync_every = 3

        with AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=sync_every):
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(sync_every):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        for name, p in worker_model.named_parameters():
            torch.testing.assert_close(p.data.cpu(), global_model.state_dict()[name])

    def test_initial_pull_syncs_worker_to_server(self) -> None:
        """__enter__ pulls server params so worker starts from server's current weights."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0)

        worker_model = _make_model(d)  # different random weights
        self.assertTrue(any(
            not torch.equal(p.data, global_model.state_dict()[n])
            for n, p in worker_model.named_parameters()
        ), "test setup: worker and server should differ initially")

        with AsyncDiLoCo(server.address(), worker_model, optim.SGD(worker_model.parameters(), lr=0.0), sync_every=100):
            for name, p in worker_model.named_parameters():
                torch.testing.assert_close(p.data.cpu(), global_model.state_dict()[name])

    def test_worker_ids_unique_per_instance(self) -> None:
        """Worker heartbeat ids must be unique per instance (not a module counter)."""
        d = 8
        w1 = AsyncDiLoCo("http://unused", _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.0), sync_every=10)
        w2 = AsyncDiLoCo("http://unused", _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.0), sync_every=10)
        self.assertNotEqual(w1._worker_id, w2._worker_id)

    def test_quantized_upload_download_stays_fp32(self) -> None:
        """R5: quantization applies to the upload only; the pulled params must
        equal the server's authoritative fp32 params exactly."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0,
            should_quantize=True,
        )
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        with AsyncDiLoCo(
            server.address(), worker_model, inner_opt, sync_every=2,
            should_quantize=True,
        ):
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(2):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        for name, p in worker_model.named_parameters():
            self.assertTrue(
                torch.equal(p.data.cpu(), global_model.state_dict()[name]),
                f"download was degraded for {name!r}",
            )

    def test_single_worker_matches_sync_diloco(self) -> None:
        """Convergence parity: with one worker, AsyncDiLoCo must reproduce
        hand-rolled synchronous DiLoCo exactly (inner state persists across
        windows, outer SGD on pseudo-gradients)."""
        torch.manual_seed(0)
        d, H, windows = 4, 3, 3
        outer_lr, inner_lr, momentum = 0.5, 0.05, 0.9
        data = [(torch.randn(8, d), torch.randn(8, d)) for _ in range(H * windows)]

        ref_global = nn.Linear(d, d)
        init = {k: v.clone() for k, v in ref_global.state_dict().items()}

        # Reference: synchronous DiLoCo, inner optimizer state persists.
        ref_local = nn.Linear(d, d)
        ref_local.load_state_dict(ref_global.state_dict())
        ref_inner = optim.SGD(ref_local.parameters(), lr=inner_lr, momentum=momentum)
        it = iter(data)
        for _ in range(windows):
            for _ in range(H):
                x, y = next(it)
                ref_inner.zero_grad()
                ((ref_local(x) - y) ** 2).mean().backward()
                ref_inner.step()
            with torch.no_grad():
                for gp, lp in zip(ref_global.parameters(), ref_local.parameters()):
                    gp.data.sub_(outer_lr * (gp.data - lp.data))
                for gp, lp in zip(ref_global.parameters(), ref_local.parameters()):
                    lp.data.copy_(gp.data)

        # AsyncDiLoCo, single worker, same init and data.
        srv_model = nn.Linear(d, d)
        wrk_model = nn.Linear(d, d)
        srv_model.load_state_dict(init)
        wrk_model.load_state_dict(init)
        server = AsyncDiLoCoServer(
            srv_model, optim.SGD(srv_model.parameters(), lr=outer_lr), port=0
        )
        inner = optim.SGD(wrk_model.parameters(), lr=inner_lr, momentum=momentum)
        it = iter(data)
        with AsyncDiLoCo(server.address(), wrk_model, inner, sync_every=H):
            for _ in range(windows * H):
                x, y = next(it)
                inner.zero_grad()
                ((wrk_model(x) - y) ** 2).mean().backward()
                inner.step()

        for gp, sp in zip(ref_global.parameters(), srv_model.parameters()):
            torch.testing.assert_close(sp.data, gp.data, atol=1e-6, rtol=1e-5)


class TestInt8Quantization(TestCase):
    """R5: upload-only blockwise symmetric int8 quantization."""

    def test_roundtrip_error_bound(self) -> None:
        """Per-element error is bounded by max|block|/254 within each block."""
        torch.manual_seed(0)
        numels = [64, 1, 300, 17]
        chunks = [torch.randn(n) * scale for n, scale in zip(numels, [1.0, 100.0, 1e-4, 3.0])]
        flat = torch.cat(chunks)

        q, scales = _quantize_int8(flat, numels)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(len(scales), len(numels))
        out = _dequantize_int8(q, scales, numels)

        offset = 0
        for n, chunk in zip(numels, chunks):
            bound = chunk.abs().max().item() / 254.0 + 1e-7
            err = (out[offset : offset + n] - chunk).abs().max().item()
            self.assertLessEqual(err, bound)
            offset += n

    def test_zero_block_roundtrips_exactly(self) -> None:
        flat = torch.zeros(10)
        q, scales = _quantize_int8(flat, [4, 6])
        torch.testing.assert_close(_dequantize_int8(q, scales, [4, 6]), flat)

    def test_constant_block_roundtrips_exactly(self) -> None:
        """A constant block maps to ±127 exactly — no quantization error."""
        flat = torch.cat([torch.full((8,), 3.5), torch.full((5,), -0.25)])
        q, scales = _quantize_int8(flat, [8, 5])
        torch.testing.assert_close(_dequantize_int8(q, scales, [8, 5]), flat)

    def test_server_applies_quantized_push(self) -> None:
        """An int8 push with constant blocks is applied exactly like fp32."""
        model = _make_model()
        ref_model = _make_model()
        ref_model.load_state_dict(model.state_dict())
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            should_quantize=True,
        )
        ref_server = AsyncDiLoCoServer(
            ref_model, optim.SGD(ref_model.parameters(), lr=0.1), port=0
        )

        params_q, _, _, applied = push_pull(
            server.address(), model, grad_value=1.0, quantize=True
        )
        params_ref, _, _, _ = push_pull(
            ref_server.address(), ref_model, grad_value=1.0
        )

        self.assertTrue(applied)
        for name in params_q:
            torch.testing.assert_close(params_q[name], params_ref[name])


class TestInnerOptimizerState(TestCase):
    """B5: DiLoCo persists inner optimizer state across windows by default;
    clearing is an opt-in deviation."""

    def _run_windows(self, reset_inner_state: bool) -> optim.Optimizer:
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0)
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01, momentum=0.9)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        with AsyncDiLoCo(
            server.address(), worker_model, inner_opt, sync_every=2,
            reset_inner_state=reset_inner_state,
        ):
            for _ in range(2):  # exactly one full window ending in a sync
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()
        return inner_opt

    def test_state_persists_across_windows_by_default(self) -> None:
        inner_opt = self._run_windows(reset_inner_state=False)
        self.assertGreater(len(inner_opt.state), 0)

    def test_reset_inner_state_opt_in(self) -> None:
        inner_opt = self._run_windows(reset_inner_state=True)
        self.assertEqual(len(inner_opt.state), 0)


class TestDyLU(TestCase):
    def _push(self, addr: str, model: nn.Module, speed: float) -> int:
        _, new_steps, _, _ = push_pull(addr, model, speed=speed, grad_value=0.0)
        return new_steps

    def test_first_worker_gets_full_H(self) -> None:
        """First worker is the fastest by definition — receives H steps back."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        self.assertEqual(self._push(server.address(), model, 50.0), 100)

    def test_slow_worker_gets_fewer_steps(self) -> None:
        """Worker at half the reference speed receives H/2 steps."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        addr = server.address()
        self._push(addr, model, 100.0)   # fast worker — registers v=100 in pool
        self.assertEqual(self._push(addr, model, 50.0), 50)

    def test_outlier_does_not_shrink_everyone(self) -> None:
        """R10: with a large pool, one outlier speed must not become the
        reference — the percentile excludes it."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        addr = server.address()
        for _ in range(15):
            self._push(addr, model, 100.0)
        self._push(addr, model, 1000.0)  # one mis-measured outlier window
        # Reference is p90 of the pool (=100), not the 1000 outlier, so a
        # normal worker keeps its full window instead of dropping to 10.
        self.assertEqual(self._push(addr, model, 100.0), 100)

    def test_worker_applies_dylu_steps(self) -> None:
        """Worker's sync_every updates to the DyLU recommendation after a sync."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0, dylu_H=100)

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        sync_every = 3

        trainer = AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=sync_every)
        with trainer:
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(sync_every):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        self.assertGreater(trainer._sync_every, 0)


class TestGracePeriod(TestCase):
    def _make_server(self, grace_period: float) -> Tuple[AsyncDiLoCoServer, nn.Module]:
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            grace_period=grace_period,
        )
        return server, model

    def _grads(self, model: nn.Module, value: float) -> Dict[str, torch.Tensor]:
        return {n: torch.full_like(p, value) for n, p in model.named_parameters()}

    def test_two_workers_share_one_batch(self) -> None:
        """Two pushes inside the grace window are applied as one batch: both
        workers receive the same post-batch snapshot and revision."""
        server, model = self._make_server(grace_period=0.5)
        addr = server.address()
        results, errors = [], []

        def worker() -> None:
            try:
                results.append(push_pull(addr, model, grad_value=1.0))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(results), 2)
        (p1, _, rev1, ap1), (p2, _, rev2, ap2) = results
        self.assertTrue(ap1 and ap2)
        self.assertEqual(rev1, rev2)
        self.assertEqual(rev1, 2)  # one outer step per worker in the batch
        for name in p1:
            torch.testing.assert_close(p1[name], p2[name])

    def test_batch_detached_at_claim_time(self) -> None:
        """B2: once a processor is elected the batch is closed — a late
        arrival opens a fresh batch instead of racing the processor."""
        server, model = self._make_server(grace_period=0.05)

        batch, is_processor = server._grace_accumulate_and_wait(
            self._grads(model, 1.0), 1.0
        )
        self.assertTrue(is_processor)
        # Batch was detached at claim time: late arrivals can't join it.
        self.assertIsNone(server._grace_batch)

        late_batch, late_is_processor = server._grace_accumulate_and_wait(
            self._grads(model, 2.0), 1.0
        )
        self.assertTrue(late_is_processor)
        self.assertIsNot(late_batch, batch)
        self.assertEqual(len(batch.grads_list), 1)
        self.assertEqual(len(late_batch.grads_list), 1)

    def test_processor_failure_publishes_error_to_waiters(self) -> None:
        """B2: if the elected processor dies, waiters get an error promptly
        instead of spinning until the transport timeout."""
        server, model = self._make_server(grace_period=30.0)  # deadline far out
        outcome: Dict[str, object] = {}

        def waiter() -> None:
            batch, is_processor = server._grace_accumulate_and_wait(
                self._grads(model, 1.0), 1.0
            )
            outcome["is_processor"] = is_processor
            outcome["error"] = batch.error

        t = threading.Thread(target=waiter)
        t.start()

        # Wait for the waiter to open the batch, then claim it on its behalf
        # (simulating another session's processor) and publish a failure.
        deadline = time.monotonic() + 5
        while server._grace_batch is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(server._grace_batch)
        with server._grace_cond:
            batch = server._grace_batch
            batch.claimed = True
            server._grace_batch = None
            server._grace_cond.notify_all()

        server._grace_batch_publish(batch, error="RuntimeError: boom")

        t.join(timeout=5)  # well under the 30s deadline
        self.assertFalse(t.is_alive(), "waiter hung after processor failure")
        self.assertEqual(outcome["is_processor"], False)
        self.assertEqual(outcome["error"], "RuntimeError: boom")

    def test_failed_batch_returns_http_error(self) -> None:
        """A sync whose grace batch failed must get a fast HTTP 500 so the
        worker drops the push and resyncs — no hanging until a transport
        timeout."""
        server, model = self._make_server(grace_period=0.05)
        addr = server.address()

        with patch.object(
            server, "_apply_one", side_effect=RuntimeError("optimizer exploded")
        ):
            start = time.monotonic()
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                push_pull(addr, model, grad_value=1.0)
            self.assertEqual(ctx.exception.code, 500)
            self.assertLess(time.monotonic() - start, 30)


class TestWorkerResilience(TestCase):
    """B3/R11: a PS outage must not kill the fleet."""

    def test_worker_survives_server_outage_and_resyncs(self) -> None:
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0
        )
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        def step() -> None:
            inner_opt.zero_grad()
            nn.CrossEntropyLoss()(worker_model(x), y).backward()
            inner_opt.step()

        trainer = AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=2)
        with trainer:
            good_addr = trainer._server_address
            # Simulate a PS outage: nothing listens here.
            trainer._server_address = "http://127.0.0.1:9/sync"

            for _ in range(2):
                step()  # boundary sync fails — must NOT raise
            self.assertTrue(trainer._pending_resync)

            for _ in range(2):
                step()  # training continues while the server is down
            self.assertTrue(trainer._pending_resync)

            # Server comes back.
            trainer._server_address = good_addr
            trainer._resync_at = 0.0
            for _ in range(2):
                step()  # boundary triggers the pull-only resync
            self.assertFalse(trainer._pending_resync)

            # Worker re-baselined to the server's authoritative params.
            for name, p in worker_model.named_parameters():
                torch.testing.assert_close(
                    p.data.cpu(), global_model.state_dict()[name]
                )


class TestCheckpoint(TestCase):
    def test_checkpoint_written_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "server.ckpt")
            model = _make_model()
            server = AsyncDiLoCoServer(
                model, optim.SGD(model.parameters(), lr=0.1), port=0,
                checkpoint_path=path, checkpoint_every=1,
            )
            push_pull(server.address(), model, grad_value=1.0)
            self.assertTrue(os.path.exists(path))
            expected = {n: p.detach().clone() for n, p in model.named_parameters()}

            # A fresh server restores model + revision from the checkpoint.
            model2 = _make_model()
            server2 = AsyncDiLoCoServer(
                model2, optim.SGD(model2.parameters(), lr=0.1), port=0,
                checkpoint_path=path,
            )
            self.assertEqual(server2._revision, 1)
            for name, p in model2.named_parameters():
                torch.testing.assert_close(p.data, expected[name])

            # And serves the restored params (with the restored revision).
            params, _, revision, _ = push_pull(
                server2.address(), model2, full_sync=False
            )
            self.assertEqual(revision, 1)
            for name in params:
                torch.testing.assert_close(params[name], expected[name])

    def test_save_checkpoint_atomic_no_tmp_left(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "server.ckpt")
            model = _make_model()
            server = AsyncDiLoCoServer(
                model, optim.SGD(model.parameters(), lr=0.1), port=0
            )
            server.save_checkpoint(path)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))


class TestAdvertiseHost(TestCase):
    """B4: all worker-facing addresses must honor advertise_host so
    multi-host deployments aren't at the mercy of socket.gethostname()."""

    def _make_server(self, **kwargs) -> AsyncDiLoCoServer:
        model = _make_model()
        return AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.0), port=0, **kwargs
        )

    def test_addresses_use_advertise_host(self) -> None:
        server = self._make_server(advertise_host="ps.example.com")
        self.assertIn("//ps.example.com:", server.address())
        self.assertIn("//ps.example.com:", server.heartbeat_address())
        self.assertIn("//ps.example.com:", server.status_address())

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"TORCHFT_PS_ADVERTISE_HOST": "10.1.2.3"}):
            server = self._make_server()
        self.assertIn("//10.1.2.3:", server.address())
        self.assertIn("//10.1.2.3:", server.heartbeat_address())

    def test_explicit_arg_beats_env(self) -> None:
        with patch.dict(os.environ, {"TORCHFT_PS_ADVERTISE_HOST": "10.1.2.3"}):
            server = self._make_server(advertise_host="ps.example.com")
        self.assertIn("//ps.example.com:", server.address())

    def test_advertised_sessions_work_end_to_end(self) -> None:
        """A worker syncing via an advertised (non-gethostname) address must
        complete a full push/pull."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            advertise_host="localhost",
        )
        self.assertIn("//localhost:", server.address())
        _, _, revision, applied = push_pull(server.address(), model)
        self.assertTrue(applied)
        self.assertEqual(revision, 1)


class TestStatusEndpoint(TestCase):
    def test_status_reports_progress(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        push_pull(server.address(), model)

        with urllib.request.urlopen(server.status_address()) as resp:
            status = json.loads(resp.read())

        self.assertEqual(status["revision"], 1)
        self.assertEqual(status["applied_pushes"], 1)
        self.assertEqual(status["worker_count"], 0)
        self.assertIsNotNone(status["last_outer_step_time"])


class TestShutdown(TestCase):
    def test_shutdown_stops_server(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0)
        addr = server.address()
        server.shutdown()
        self.assertTrue(server._shutdown_event.is_set())
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(addr, timeout=5)


class TestHeartbeat(TestCase):
    def _make_server(self, **kwargs) -> AsyncDiLoCoServer:
        model = _make_model()
        return AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, **kwargs)

    def _make_worker(self, server: AsyncDiLoCoServer, **kwargs) -> AsyncDiLoCo:
        model = _make_model()
        return AsyncDiLoCo(
            server.address(), model, optim.SGD(model.parameters(), lr=0.0),
            sync_every=1000, heartbeat_address=server.heartbeat_address(), **kwargs
        )

    def test_worker_registers_on_enter(self) -> None:
        server = self._make_server(heartbeat_timeout=5.0)
        with self._make_worker(server, heartbeat_interval=0.05):
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 1)

    def test_worker_deregisters_after_timeout(self) -> None:
        server = self._make_server(heartbeat_timeout=0.3)
        with self._make_worker(server, heartbeat_interval=0.05):
            time.sleep(0.2)
        time.sleep(0.8)
        self.assertEqual(server.worker_count(), 0)

    def test_two_workers_both_visible(self) -> None:
        server = self._make_server(heartbeat_timeout=5.0)
        w1 = self._make_worker(server, heartbeat_interval=0.05)
        w2 = self._make_worker(server, heartbeat_interval=0.05)
        with w1, w2:
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 2)

    def test_single_port_for_all_endpoints(self) -> None:
        """R6: one port to open/advertise — /sync, /heartbeat and /status all
        live on the same server."""
        server = self._make_server()
        sync_port = server.address().rsplit(":", 1)[1].split("/")[0]
        hb_port = server.heartbeat_address().rsplit(":", 1)[1].split("/")[0]
        status_port = server.status_address().rsplit(":", 1)[1].split("/")[0]
        self.assertEqual(sync_port, hb_port)
        self.assertEqual(hb_port, status_port)


class TestWorkerRejoin(TestCase):
    def test_rejoined_worker_gets_updated_params(self) -> None:
        """Worker rejoining after a sync pulls the server's latest global params."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        w1 = _make_model(d)
        w1.load_state_dict(global_model.state_dict())
        o1 = optim.SGD(w1.parameters(), lr=0.01)
        with AsyncDiLoCo(server.address(), w1, o1, sync_every=2):
            for _ in range(2):
                o1.zero_grad()
                nn.CrossEntropyLoss()(w1(x), y).backward()
                o1.step()

        server_params = {n: p.data.clone() for n, p in global_model.named_parameters()}

        w2, o2 = _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.01)
        with AsyncDiLoCo(server.address(), w2, o2, sync_every=100):
            for name, p in w2.named_parameters():
                torch.testing.assert_close(p.data.cpu(), server_params[name])

    def test_training_continues_after_rejoin(self) -> None:
        """Global params keep updating through a worker leave-and-rejoin cycle."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))
        initial = {n: p.data.clone() for n, p in global_model.named_parameters()}

        for _ in range(2):
            wm = _make_model(d)
            wo = optim.SGD(wm.parameters(), lr=0.01)
            with AsyncDiLoCo(server.address(), wm, wo, sync_every=2):
                for _ in range(2):
                    wo.zero_grad()
                    nn.CrossEntropyLoss()(wm(x), y).backward()
                    wo.step()

        self.assertTrue(any(
            not torch.equal(p.data, initial[n]) for n, p in global_model.named_parameters()
        ))


def _mp_worker_main(addr: str, hb_addr: str, d: int, queue) -> None:
    """Real multi-process worker: separate interpreter, real sockets."""
    try:
        import torch
        from torch import nn, optim

        from panoengine.decentralized.async_diloco import AsyncDiLoCo

        model = nn.Sequential(nn.Linear(d, d))
        inner = optim.SGD(model.parameters(), lr=0.01)
        with AsyncDiLoCo(
            addr, model, inner, sync_every=2,
            heartbeat_address=hb_addr, heartbeat_interval=0.05,
        ) as trainer:
            x, y = torch.randn(4, d), torch.randn(4, d)
            for _ in range(4):
                inner.zero_grad()
                ((model(x) - y) ** 2).mean().backward()
                inner.step()
            # Stay alive briefly so the parent can observe both heartbeats.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                time.sleep(0.05)
        queue.put(("ok", trainer._worker_id))
    except Exception as e:
        queue.put(("error", repr(e)))


class TestMultiProcess(TestCase):
    """Real multi-process test over actual sockets: two separate worker
    processes must register distinct heartbeat ids (B1) and sync against a
    server addressed by hostname (B4)."""

    def test_two_processes_sync_and_register(self) -> None:
        d = 4
        global_model = nn.Sequential(nn.Linear(d, d))
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1),
            port=0, advertise_host="localhost",
        )

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_mp_worker_main,
                args=(server.address(), server.heartbeat_address(), d, queue),
            )
            for _ in range(2)
        ]
        for p in procs:
            p.start()
        try:
            # Both processes must appear as distinct active workers.
            deadline = time.monotonic() + 60
            while server.worker_count() < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertEqual(server.worker_count(), 2)

            results = [queue.get(timeout=60) for _ in range(2)]
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()

        statuses = [r[0] for r in results]
        self.assertEqual(statuses, ["ok", "ok"], f"worker failures: {results}")
        worker_ids = {r[1] for r in results}
        self.assertEqual(len(worker_ids), 2, "worker ids collided across processes")
        # Both workers pushed at least once.
        self.assertGreaterEqual(server._revision, 2)


# --------------------------------------------------------------------------- #
# Replica mode (replica_pg): one PS session per multi-rank, DTensor-sharded
# replica. Two gloo CPU processes shard a seeded model Shard(0) — with an
# UNEVEN leading dim, so shard sizes differ per rank — and must behave as ONE
# parameter-server worker (rank-0 session; broadcast-adopted params).
# --------------------------------------------------------------------------- #
import torch.distributed as pt_dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from panoengine.decentralized.async_diloco import _local_shard_slices

_REPLICA_D_IN, _REPLICA_D_OUT = 4, 5  # 5 rows over 2 ranks: 3/2, uneven


def _replica_base_model(seed: int = 1234) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(_REPLICA_D_IN, _REPLICA_D_OUT)


def _shard_params_(model: nn.Module) -> None:
    """Replace every parameter with its Shard(0) DTensor over the WORLD mesh
    (every rank passes the identical seeded full tensor)."""
    mesh = init_device_mesh("cpu", (pt_dist.get_world_size(),))
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            setattr(
                mod,
                name,
                nn.Parameter(distribute_tensor(p.detach(), mesh, [Shard(0)])),
            )


def _drive_windows(
    model: nn.Module, opt: optim.Optimizer, windows: int, sync_every: int
) -> None:
    """Deterministic inner steps: grad == ones everywhere, so replica and
    single-process reference runs push identical pseudo-gradients."""
    for _ in range(windows * sync_every):
        opt.zero_grad()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()


def _init_replica_dist(rank: int, dist_port: int) -> None:
    pt_dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{dist_port}",
        rank=rank,
        world_size=2,
    )


def _replica_sync_child(
    rank: int, dist_port: int, server_addr: str, hb_addr: str
) -> None:
    """Child for test_two_rank_replica_syncs_shards: 2 windows over uneven
    Shard(0) params, then per-rank slice equality against the server."""
    _init_replica_dist(rank, dist_port)
    try:
        model = _replica_base_model()
        _shard_params_(model)
        opt = optim.SGD(model.parameters(), lr=0.1)
        client = _fp32_client(
            server_addr,
            model,
            opt,
            sync_every=3,
            heartbeat_address=hb_addr,
            replica_pg=pt_dist.group.WORLD,
        )
        # Only the lead may carry a worker identity (a follower heartbeat
        # would register the replica twice).
        assert (client._heartbeat_url is not None) == (rank == 0), rank
        with client:
            _drive_windows(model, opt, windows=2, sync_every=3)
        assert client._baseline_revision == (2 if rank == 0 else 0), (
            rank,
            client._baseline_revision,
        )
        # Every rank's shard must equal its slice of the server's params.
        server_params, _, revision, _ = push_pull(
            server_addr, model, full_sync=False
        )
        assert revision == 2, revision
        for name, p in model.named_parameters():
            expect = server_params[name][_local_shard_slices(p)]
            assert torch.equal(p.to_local(), expect), (rank, name)
    finally:
        pt_dist.destroy_process_group()


def _replica_dylu_child(
    rank: int, dist_port: int, server_addr: str
) -> None:
    """Child for test_dylu_reaches_every_rank: the server's new_steps must
    land on BOTH ranks (via the outcome broadcast), or window lengths drift
    and the next collective deadlocks."""
    _init_replica_dist(rank, dist_port)
    try:
        model = _replica_base_model()
        _shard_params_(model)
        opt = optim.SGD(model.parameters(), lr=0.1)
        client = _fp32_client(
            server_addr, model, opt, sync_every=5,
            replica_pg=pt_dist.group.WORLD,
        )
        with client:
            _drive_windows(model, opt, windows=1, sync_every=5)
        assert client._sync_every == 10, (rank, client._sync_every)
    finally:
        pt_dist.destroy_process_group()


def _replica_failure_child(
    rank: int, dist_port: int, server_addr: str, flag_path: str
) -> None:
    """Child for test_server_death_keeps_ranks_lockstep: after the server
    dies, every boundary must still complete on BOTH ranks (FAIL/SKIP words
    keep them in lockstep) with training continuing locally."""
    _init_replica_dist(rank, dist_port)
    try:
        model = _replica_base_model()
        _shard_params_(model)
        opt = optim.SGD(model.parameters(), lr=0.1)
        client = _fp32_client(
            server_addr, model, opt, sync_every=2,
            replica_pg=pt_dist.group.WORLD,
        )
        with client:
            _drive_windows(model, opt, windows=1, sync_every=2)  # healthy
            pt_dist.barrier()
            # parent shuts the server down, then touches the flag file
            deadline = time.monotonic() + 30
            while not os.path.exists(flag_path):
                assert time.monotonic() < deadline, "parent never flagged"
                time.sleep(0.05)
            before = {
                n: p.to_local().clone() for n, p in model.named_parameters()
            }
            # failed sync boundary + a resync-attempt boundary: both must
            # return on both ranks rather than hang
            _drive_windows(model, opt, windows=2, sync_every=2)
            if rank == 0:
                assert client._pending_resync, "lead should be in resync"
            assert client._baseline_revision == (1 if rank == 0 else 0)
            # no adopt happened: params are the locally-trained values
            for n, p in model.named_parameters():
                lr_drift = 0.1 * 2 * 2  # lr * steps of two windows
                expect = before[n] - lr_drift
                assert torch.allclose(p.to_local(), expect), (rank, n)
    finally:
        pt_dist.destroy_process_group()


def _replica_blend_child(
    rank: int, dist_port: int, server_addr: str
) -> None:
    """Child for test_blend_alpha_replica: fragment_update_alpha lerps each
    rank's OWN pre-adopt shard."""
    _init_replica_dist(rank, dist_port)
    try:
        model = _replica_base_model()
        _shard_params_(model)
        init = {n: p.to_local().clone() for n, p in model.named_parameters()}
        opt = optim.SGD(model.parameters(), lr=0.1)
        alpha = 0.5
        client = _fp32_client(
            server_addr, model, opt, sync_every=1,
            fragment_update_alpha=alpha,
            replica_pg=pt_dist.group.WORLD,
        )
        with client:
            _drive_windows(model, opt, windows=1, sync_every=1)
        server_params, _, _, _ = push_pull(server_addr, model, full_sync=False)
        for n, p in model.named_parameters():
            local_pre = init[n] - 0.1  # one ones-grad SGD step
            expect = torch.lerp(
                server_params[n][_local_shard_slices(p)], local_pre, alpha
            )
            assert torch.allclose(p.to_local(), expect), (rank, n)
    finally:
        pt_dist.destroy_process_group()


class TestReplicaMode(TestCase):
    def _free_port(self) -> int:
        import socket as _socket

        with _socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _spawn(self, fn, *args) -> None:
        torch.multiprocessing.spawn(
            fn, args=(self._free_port(), *args), nprocs=2, join=True
        )

    def _server(self, lr: float = 0.5, **kw) -> AsyncDiLoCoServer:
        model = _replica_base_model()
        return AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=lr), port=0, **kw
        )

    def test_two_rank_replica_syncs_shards(self) -> None:
        server = self._server(heartbeat_timeout=5.0)
        try:
            self._spawn(
                _replica_sync_child, server.address(),
                server.heartbeat_address(),
            )
            self.assertEqual(server.status()["revision"], 2)
            # Equivalence: a single-process worker over the SAME seeded model
            # and grad schedule must leave an identical server.
            ref_server = self._server()
            try:
                ref_model = _replica_base_model()
                ref_opt = optim.SGD(ref_model.parameters(), lr=0.1)
                with AsyncDiLoCo(
                    ref_server.address(), ref_model, ref_opt, sync_every=3
                ):
                    _drive_windows(ref_model, ref_opt, windows=2, sync_every=3)
                got, _, _, _ = push_pull(
                    server.address(), _replica_base_model(), full_sync=False
                )
                want, _, _, _ = push_pull(
                    ref_server.address(), _replica_base_model(),
                    full_sync=False,
                )
                for name in got:
                    torch.testing.assert_close(got[name], want[name])
            finally:
                ref_server.shutdown()
        finally:
            server.shutdown()

    def test_dylu_reaches_every_rank(self) -> None:
        server = self._server(dylu_H=10)
        try:
            self._spawn(_replica_dylu_child, server.address())
        finally:
            server.shutdown()

    def test_server_death_keeps_ranks_lockstep(self) -> None:
        server = self._server()
        flag = tempfile.mktemp(prefix="asyncdiloco-down-")
        try:
            procs_err: list = []

            def _run() -> None:
                try:
                    self._spawn(
                        _replica_failure_child, server.address(), flag
                    )
                except Exception as e:  # surfaced after join
                    procs_err.append(e)

            t = threading.Thread(target=_run)
            t.start()
            deadline = time.monotonic() + 30
            while server.status()["revision"] < 1:
                self.assertLess(time.monotonic(), deadline, "no first sync")
                time.sleep(0.05)
            server.shutdown()
            with open(flag, "w"):
                pass
            t.join(timeout=90)
            self.assertFalse(t.is_alive(), "replica hung after server death")
            self.assertEqual(procs_err, [])
        finally:
            if os.path.exists(flag):
                os.unlink(flag)

    def test_blend_alpha_replica(self) -> None:
        server = self._server()
        try:
            self._spawn(_replica_blend_child, server.address())
        finally:
            server.shutdown()

    def test_replica_fragment_rotation(self) -> None:
        server = self._server(num_fragments=2)
        try:
            self._spawn(_replica_fragment_child, server.address())
            self.assertEqual(server.status()["applied_pushes"], 4)
        finally:
            server.shutdown()


def _replica_fragment_child(
    rank: int, dist_port: int, server_addr: str
) -> None:
    """Child for test_replica_fragment_rotation: fragment boundaries must
    stay lockstep across ranks — the pseudo-gradient gathers and adopt
    broadcasts are collectives on the MAIN thread; only the lead's HTTP
    roundtrip overlaps in the background."""
    _init_replica_dist(rank, dist_port)
    try:
        model = _replica_base_model()
        _shard_params_(model)
        opt = optim.SGD(model.parameters(), lr=0.1)
        client = _fp32_client(
            server_addr, model, opt, sync_every=4, num_fragments=2,
            replica_pg=pt_dist.group.WORLD,
        )
        with client:
            # sync_every=4, P=2 → a fragment boundary every 2 steps; 8 steps
            # = 4 boundaries: launch f0 / adopt f0 + launch f1 / adopt f1 +
            # launch f0 / adopt f0 + launch f1 (drained at exit, not adopted).
            _drive_windows(model, opt, windows=4, sync_every=2)
        assert client._frag_idx == 0, client._frag_idx  # 4 rotations wrap
        # The lead adopted pushes 1..3 (revisions 1..3); push 4 was drained.
        assert client._baseline_revision == (3 if rank == 0 else 0), (
            rank,
            client._baseline_revision,
        )
    finally:
        pt_dist.destroy_process_group()


class TestFragmentSync(TestCase):
    """Fragment-wise sync (Decoupled DiLoCo, arXiv 2604.21428): the
    partition, wire validation, bitwise equivalence of fragment pushes to
    whole-model pushes, and the client's staggered overlap pipeline."""

    def test_fragment_bounds_partition(self) -> None:
        from panoengine.decentralized.async_diloco import _fragment_bounds

        self.assertEqual(_fragment_bounds([10, 10, 10, 10], 1), [(0, 4)])
        self.assertEqual(_fragment_bounds([10, 10, 10, 10], 2), [(0, 2), (2, 4)])
        self.assertEqual(
            _fragment_bounds([10, 10, 10, 10], 4),
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        # A huge tail param: every remaining fragment still gets >= 1 param.
        self.assertEqual(_fragment_bounds([1, 1, 1, 100], 2), [(0, 3), (3, 4)])
        # Coverage, contiguity and non-emptiness for arbitrary shapes.
        numels = [3, 7, 1, 9, 2, 8, 5]
        for p in range(1, len(numels) + 1):
            bounds = _fragment_bounds(numels, p)
            self.assertEqual(len(bounds), p)
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], len(numels))
            for (a0, b0), (a1, _) in zip(bounds, bounds[1:]):
                self.assertEqual(b0, a1)
            for a0, b0 in bounds:
                self.assertGreater(b0, a0)
        with self.assertRaises(ValueError):
            _fragment_bounds([1, 2], 3)
        with self.assertRaises(ValueError):
            _fragment_bounds([1, 2], 0)

    def test_ctor_validation(self) -> None:
        model = _make_model()
        opt = optim.SGD(model.parameters(), lr=0.5)
        with self.assertRaises(ValueError):
            AsyncDiLoCoServer(
                model, opt, port=0, num_fragments=2, grace_period=0.5
            )
        with self.assertRaises(ValueError):
            AsyncDiLoCo(
                "http://localhost:1/sync", model, opt,
                sync_every=5, num_fragments=2,  # not divisible
            )
        with self.assertRaises(ValueError):
            AsyncDiLoCo(
                "http://localhost:1/sync", model, opt,
                sync_every=4, num_fragments=0,
            )

    def test_fragmented_pushes_bitwise_equal_whole_push(self) -> None:
        """The server-level convergence-neutrality proof: P fragment pushes
        of the same deltas commit BITWISE the same θ and momentum as one
        whole-model push — block correction, look-ahead and momentum are all
        per-parameter, and int8 quantization blocks are per-parameter too."""
        from panoengine.decentralized.async_diloco import _fragment_bounds
        from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer

        def heloco_opt(m: nn.Module) -> optim.Optimizer:
            return HeLoCoOptimizer(m.parameters(), lr=0.7, momentum=0.9)

        def sgd_opt(m: nn.Module) -> optim.Optimizer:
            return optim.SGD(m.parameters(), lr=0.5, momentum=0.9)

        cases = [
            ("heloco-fp32", HeLoCoServer, heloco_opt, False),
            ("heloco-int8", HeLoCoServer, heloco_opt, True),
            ("base-sgd-fp32", AsyncDiLoCoServer, sgd_opt, False),
        ]
        P = 3
        for label, cls, opt_f, quantize in cases:
            with self.subTest(label):
                torch.manual_seed(11)
                model_a = _make_model()
                torch.manual_seed(11)
                model_b = _make_model()
                opt_a, opt_b = opt_f(model_a), opt_f(model_b)
                server_a = cls(model_a, opt_a, port=0, should_quantize=quantize)
                server_b = cls(
                    model_b, opt_b, port=0, should_quantize=quantize,
                    num_fragments=P,
                )
                try:
                    numels = [p.numel() for p in model_a.parameters()]
                    bounds = _fragment_bounds(numels, P)
                    # Two rounds: round 2 corrects against seeded momentum.
                    for rnd in range(2):
                        flat = torch.randn(
                            sum(numels),
                            generator=torch.Generator().manual_seed(100 + rnd),
                        )
                        push_pull_flat(
                            server_a.address(), flat,
                            quantize=quantize, numels=numels,
                        )
                        off = 0
                        for f, (i, j) in enumerate(bounds):
                            n = sum(numels[i:j])
                            push_pull_flat(
                                server_b.address(), flat[off : off + n],
                                quantize=quantize, numels=numels[i:j],
                                fragment=f, num_fragments=P,
                            )
                            off += n
                    for (name, p_a), p_b in zip(
                        model_a.named_parameters(), model_b.parameters()
                    ):
                        self.assertTrue(torch.equal(p_a, p_b), name)
                    for p_a, p_b in zip(
                        model_a.parameters(), model_b.parameters()
                    ):
                        s_a = opt_a.state.get(p_a, {})
                        s_b = opt_b.state.get(p_b, {})
                        for key in ("m", "momentum_buffer"):
                            if key in s_a or key in s_b:
                                self.assertTrue(
                                    torch.equal(s_a[key], s_b[key]), key
                                )
                    self.assertEqual(server_a.status()["revision"], 2)
                    self.assertEqual(server_b.status()["revision"], 2 * P)
                    # Pull-only responses (whole model on both) bitwise equal
                    # — covers the fragmented server's snapshot concatenation
                    # and HeLoCo's fragment-scoped look-ahead.
                    pa, _, _, _ = push_pull(
                        server_a.address(), model_a, full_sync=False
                    )
                    pb, _, _, _ = push_pull(
                        server_b.address(), model_b, full_sync=False
                    )
                    for name in pa:
                        self.assertTrue(torch.equal(pa[name], pb[name]), name)
                finally:
                    server_a.shutdown()
                    server_b.shutdown()

    def test_materialized_fragment_apply_matches_whole(self) -> None:
        """_handle_sync with a materialized fragment flat buffer (the
        non-streaming entry, exercising _unflatten's fragment slicing)."""
        from panoengine.decentralized.async_diloco import _fragment_bounds
        from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer

        torch.manual_seed(17)
        model_a = _make_model()
        torch.manual_seed(17)
        model_b = _make_model()
        opt_a = HeLoCoOptimizer(model_a.parameters(), lr=0.7, momentum=0.9)
        opt_b = HeLoCoOptimizer(model_b.parameters(), lr=0.7, momentum=0.9)
        server_a = HeLoCoServer(model_a, opt_a, port=0)
        server_b = HeLoCoServer(model_b, opt_b, port=0, num_fragments=2)
        try:
            numels = [p.numel() for p in model_a.parameters()]
            flat = torch.randn(
                sum(numels), generator=torch.Generator().manual_seed(23)
            )
            server_a._handle_sync(
                is_full_sync=True, worker_speed=1.0, baseline_revision=0,
                flat_grads=flat,
            )
            off = 0
            for f, (i, j) in enumerate(_fragment_bounds(numels, 2)):
                n = sum(numels[i:j])
                server_b._handle_sync(
                    is_full_sync=True, worker_speed=1.0, baseline_revision=0,
                    flat_grads=flat[off : off + n], fragment=f,
                )
                off += n
            for (name, p_a), p_b in zip(
                model_a.named_parameters(), model_b.parameters()
            ):
                self.assertTrue(torch.equal(p_a, p_b), name)
        finally:
            server_a.shutdown()
            server_b.shutdown()

    def test_wire_validation(self) -> None:
        def mk_server(num_fragments: int) -> AsyncDiLoCoServer:
            model = _make_model()
            return AsyncDiLoCoServer(
                model, optim.SGD(model.parameters(), lr=0.5), port=0,
                num_fragments=num_fragments,
            )

        from panoengine.decentralized.async_diloco import _fragment_bounds

        model = _make_model()
        numels = [p.numel() for p in model.parameters()]
        i, j = _fragment_bounds(numels, 2)[0]
        frag0_numels = numels[i:j]
        n0 = sum(frag0_numels)

        server2 = mk_server(num_fragments=2)
        try:
            # Whole-model push to a fragmented server.
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                push_pull(server2.address(), model)
            self.assertEqual(ctx.exception.code, 500)
            # num_fragments mismatch.
            with self.assertRaises(urllib.error.HTTPError):
                push_pull_flat(
                    server2.address(), torch.zeros(n0), quantize=False,
                    numels=frag0_numels, fragment=0, num_fragments=3,
                )
            # Fragment index out of range.
            with self.assertRaises(urllib.error.HTTPError):
                push_pull_flat(
                    server2.address(), torch.zeros(n0), quantize=False,
                    numels=frag0_numels, fragment=5, num_fragments=2,
                )
            # Wrong fragment numel.
            with self.assertRaises(urllib.error.HTTPError):
                push_pull_flat(
                    server2.address(), torch.zeros(n0 + 1), quantize=False,
                    numels=frag0_numels, fragment=0, num_fragments=2,
                )
            # A valid fragment push still works after the failures.
            resp = push_pull_flat(
                server2.address(), torch.zeros(n0), quantize=False,
                numels=frag0_numels, fragment=0, num_fragments=2,
            )
            self.assertTrue(resp["applied"])
            self.assertEqual(resp["numel"], n0)  # fragment-sized response
        finally:
            server2.shutdown()

        server1 = mk_server(num_fragments=1)
        try:
            # Fragment push to a whole-model server.
            with self.assertRaises(urllib.error.HTTPError):
                push_pull_flat(
                    server1.address(), torch.zeros(n0), quantize=False,
                    numels=frag0_numels, fragment=0, num_fragments=2,
                )
        finally:
            server1.shutdown()

    def test_client_schedule_and_overlap(self) -> None:
        """sync_every=4, P=2 → a boundary every 2 steps rotating f0,f1,f0,f1;
        each exchange launches at its boundary and is adopted at the next
        (the slow roundtrip forces the join to actually block — backpressure
        — without losing a single push)."""
        torch.manual_seed(3)
        s_model = nn.Linear(4, 5)  # 2 params → f0=weight, f1=bias
        server = AsyncDiLoCoServer(
            s_model, optim.SGD(s_model.parameters(), lr=0.5), port=0,
            num_fragments=2,
        )
        try:
            torch.manual_seed(3)
            model = nn.Linear(4, 5)
            opt = optim.SGD(model.parameters(), lr=0.1)
            client = _fp32_client(
                server.address(), model, opt, sync_every=4, num_fragments=2
            )
            order: list = []
            real = client._session_roundtrip

            def slow_roundtrip(flag, speed, flat_grads, fragment=None):
                if flat_grads is not None:
                    order.append(fragment)
                    time.sleep(0.05)
                return real(flag, speed, flat_grads, fragment=fragment)

            client._session_roundtrip = slow_roundtrip
            with client:
                _drive_windows(model, opt, windows=4, sync_every=2)  # 8 steps
            self.assertEqual(order, [0, 1, 0, 1])
            self.assertEqual(server.status()["applied_pushes"], 4)
            # Adopted pushes 1..3 (revisions 1..3); push 4 drained at exit.
            self.assertEqual(client._baseline_revision, 3)
            self.assertFalse(client._pending_resync)
            # f0 (weight) was last committed by push 3 and adopted at the
            # final boundary; push 4 only touched f1 — so the client's
            # weight must equal the server's CURRENT weight exactly.
            server_params, _, _, _ = push_pull(
                server.address(), model, full_sync=False
            )
            self.assertTrue(
                torch.equal(model.weight.data, server_params["weight"])
            )
        finally:
            server.shutdown()

    def test_join_failure_triggers_whole_model_resync(self) -> None:
        torch.manual_seed(5)
        s_model = nn.Linear(4, 5)
        server = AsyncDiLoCoServer(
            s_model, optim.SGD(s_model.parameters(), lr=0.5), port=0,
            num_fragments=2,
        )
        try:
            torch.manual_seed(9)  # different init: resync must adopt server's
            model = nn.Linear(4, 5)
            opt = optim.SGD(model.parameters(), lr=0.1)
            client = _fp32_client(
                server.address(), model, opt, sync_every=4, num_fragments=2
            )
            real = client._session_roundtrip

            def flaky(flag, speed, flat_grads, fragment=None):
                if flat_grads is not None:  # every push fails; pulls succeed
                    raise RuntimeError("boom")
                return real(flag, speed, flat_grads, fragment=fragment)

            client._session_roundtrip = flaky
            with client:
                _drive_windows(model, opt, windows=1, sync_every=2)  # launch f0
                _drive_windows(model, opt, windows=1, sync_every=2)  # join→fail
                self.assertTrue(client._pending_resync)
                self.assertIsNone(client._inflight)
                _drive_windows(model, opt, windows=1, sync_every=2)  # resync
            self.assertFalse(client._pending_resync)
            self.assertEqual(server.status()["applied_pushes"], 0)
            server_params = dict(s_model.named_parameters())
            for name, p in model.named_parameters():
                self.assertTrue(
                    torch.equal(p.data, server_params[name].data), name
                )
        finally:
            server.shutdown()

    def test_rejected_push_resyncs_whole_model(self) -> None:
        """A stale-baseline rejection means EVERY fragment's baseline is
        suspect (server checkpoint restore) — the client must skip the
        fragment-sized response and re-baseline the whole model."""
        torch.manual_seed(7)
        s_model = nn.Linear(4, 5)
        server = AsyncDiLoCoServer(
            s_model, optim.SGD(s_model.parameters(), lr=0.5), port=0,
            num_fragments=2,
        )
        try:
            torch.manual_seed(7)
            model = nn.Linear(4, 5)
            opt = optim.SGD(model.parameters(), lr=0.1)
            client = _fp32_client(
                server.address(), model, opt, sync_every=4, num_fragments=2
            )
            with client:
                client._baseline_revision = 5  # ahead of server revision 0
                _drive_windows(model, opt, windows=1, sync_every=2)  # launch
                _drive_windows(model, opt, windows=1, sync_every=2)  # join→rej
                self.assertTrue(client._pending_resync)
                _drive_windows(model, opt, windows=1, sync_every=2)  # resync
            self.assertFalse(client._pending_resync)
            self.assertEqual(client._baseline_revision, 0)
            self.assertEqual(server.status()["applied_pushes"], 0)
        finally:
            server.shutdown()


class TestCohortGate(TestCase):
    """`min_replicas`: the parameter-server equivalent of torchft's lighthouse
    `min_replicas`.

    heloco is asynchronous by design, so a lone worker trains and pushes happily.
    That let a two-island cross-cluster run go green having averaged
    nothing: the fast island finished before the slow one had provisioned, and the
    server never saw more than one worker at a time.
    """

    def _server(self, d: int = 8) -> AsyncDiLoCoServer:
        global_model = _make_model(d)
        return AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.1),
                                 port=0)

    def _worker(self, server: AsyncDiLoCoServer, d: int = 8, **kw) -> AsyncDiLoCo:
        m = _make_model(d)
        return AsyncDiLoCo(server.address(), m, optim.SGD(m.parameters(), lr=0.01),
                           sync_every=100, heartbeat_address=server.heartbeat_address(),
                           **kw)

    def test_default_does_not_wait(self) -> None:
        """min_replicas unset -> the historical behavior, a solo worker just runs."""
        server = self._server()
        started = time.monotonic()
        with self._worker(server):
            pass
        self.assertLess(time.monotonic() - started, 20.0)

    def test_single_worker_times_out_instead_of_training_alone(self) -> None:
        """THE regression: with a cohort of 2 required and only one present, entering
        must RAISE. Training alone is what silently defeated the run."""
        server = self._server()
        worker = self._worker(server, min_replicas=2, min_replicas_timeout=3.0)
        with self.assertRaises(TimeoutError) as ctx:
            with worker:
                pass
        msg = str(ctx.exception)
        self.assertIn("min_replicas", msg)          # names the knob to change
        self.assertIn("parameter server", msg)     # and where it was waiting

    def test_proceeds_once_the_cohort_arrives(self) -> None:
        """A second worker registering releases the first: the gate opens on
        worker_count, and each worker counts ITSELF via its own heartbeat."""
        server = self._server()

        # Stand in for the second island: register a heartbeat id directly, which is
        # exactly what a remote worker's heartbeat thread does.
        def join_late() -> None:
            time.sleep(1.0)
            urllib.request.urlopen(
                f"{server.heartbeat_address()}?worker_id=island-1", timeout=5
            ).read()

        t = threading.Thread(target=join_late, daemon=True)
        t.start()
        started = time.monotonic()
        with self._worker(server, min_replicas=2, min_replicas_timeout=30.0):
            waited = time.monotonic() - started
        t.join(timeout=5)
        self.assertGreater(waited, 0.5)            # it really did wait
        self.assertLess(waited, 25.0)              # and it really did proceed
        # Both islands are accounted for -- but this worker has now LEFT its context,
        # so it announced /done and moved from active to finished. Counting only
        # worker_count here would assert that a completed worker is still training.
        status = server.status()
        self.assertGreaterEqual(
            status["worker_count"] + status["finished_count"], 2)


    def test_cohort_is_rechecked_mid_run(self) -> None:
        """Lighthouse parity: the floor holds for the WHOLE run, not just startup.

        Enter with the cohort satisfied, let the peer stop heartbeating, and the next
        sync boundary must block and then fail -- not push alone. Without this a run
        that starts as two islands and loses one silently becomes a solo run, which
        is the same green-but-meaningless outcome as never overlapping at all.
        """
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0,
            heartbeat_timeout=1.0,        # let the peer expire quickly
        )
        # Peer joins once and then goes silent, like an island that finished early.
        urllib.request.urlopen(
            f"{server.heartbeat_address()}?worker_id=island-1", timeout=5
        ).read()

        m = _make_model(d)
        inner = optim.SGD(m.parameters(), lr=0.01)
        sync_every = 2
        worker = AsyncDiLoCo(
            server.address(), m, inner, sync_every=sync_every,
            heartbeat_address=server.heartbeat_address(),
            min_replicas=2, min_replicas_timeout=3.0,
        )
        with self.assertRaises(TimeoutError):
            with worker:
                self.assertGreaterEqual(server.worker_count(), 2)   # startup was fine
                time.sleep(1.5)                                     # peer expires
                x, y = torch.randn(4, d), torch.randint(0, d, (4,))
                for _ in range(sync_every):                         # -> boundary
                    inner.zero_grad()
                    nn.CrossEntropyLoss()(m(x), y).backward()
                    inner.step()
