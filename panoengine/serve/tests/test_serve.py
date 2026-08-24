# Copyright (c) Panocular AI.
#
# Serve-plane unit proofs, in the pipeline's build order:
#   sharder       — proportional ranges, edge-tensor placement/stripping,
#                   manifest integrity (CPU-only, tiny synthetic checkpoint)
#   transport     — bidirectional tensors over one dialed connection, dtype
#                   fidelity (incl. bfloat16), latency injection
#   weights       — verified streamed fetch: ranged chunking, integrity
#                   refusal, verified resume
#   WAN sim       — LinkProfile pacing/jitter (no GPUs)
#   gateway       — least-loaded routing, bearer auth, streaming passthrough
#   weights CLI   — plain-island verified whole-model-dir fetch

import asyncio
import hashlib
import http.server
import json
import socket
import threading
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from panoengine.serve import stage_transport as st  # noqa: E402
from panoengine.serve.gateway import Gateway  # noqa: E402
from panoengine.serve.sharder import plan_ranges, shard_checkpoint  # noqa: E402
from panoengine.serve.weights import (  # noqa: E402
    IntegrityError, fetch_file, fetch_manifest, fetch_model_dir,
)


# ---------------------------------------------------------------- sharder -- #
def _tiny_checkpoint(dir_, num_layers=8, hidden=16, vocab=32):
    """A fake Qwen-shaped checkpoint: embeddings, N identical layers, head."""
    t = {}
    t["model.embed_tokens.weight"] = torch.randn(vocab, hidden)
    for i in range(num_layers):
        t[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        t[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(hidden, hidden)
    t["model.norm.weight"] = torch.randn(hidden)
    t["lm_head.weight"] = torch.randn(vocab, hidden)
    save_file(t, str(dir_ / "model.safetensors"), metadata={"format": "pt"})
    (dir_ / "config.json").write_text(json.dumps(
        {"model_type": "qwen3", "num_hidden_layers": num_layers,
         "hidden_size": hidden, "vocab_size": vocab,
         # complete dims so the stage config stays loadable as a real HF config
         "intermediate_size": hidden, "num_attention_heads": 2,
         "num_key_value_heads": 2, "head_dim": hidden // 2,
         "tie_word_embeddings": False}))
    (dir_ / "tokenizer.json").write_text("{}")
    return t


def test_plan_ranges_proportional_and_edge_discounted():
    # equal memory, no edge weight -> even split
    assert plan_ranges(8, [10, 10], 0, 0) == [(0, 4), (4, 8)]
    # 3:1 memory -> 3:1 layers
    assert plan_ranges(8, [30, 10], 0, 0) == [(0, 6), (6, 8)]
    # heavy embeddings discount stage 0's share
    gb = 1 << 30
    assert plan_ranges(8, [10, 10], embed_bytes=5 * gb, head_bytes=0)[0][1] < 4
    # every stage keeps >= 1 layer even when its budget is ~all edge bytes
    r = plan_ranges(4, [1, 1], embed_bytes=(1 << 30), head_bytes=0)
    assert all(hi - lo >= 1 for lo, hi in r)
    with pytest.raises(ValueError, match="at least 2"):
        plan_ranges(8, [10], 0, 0)


def test_shard_checkpoint_strips_and_manifests(tmp_path):
    src = tmp_path / "model"
    src.mkdir()
    _tiny_checkpoint(src, num_layers=8)
    out = tmp_path / "staged"

    manifest = shard_checkpoint(src, out, [10, 10])
    s0, s1 = manifest["stages"]
    assert (s0["layer_start"], s0["layer_end"]) == (0, 4)
    assert (s1["layer_start"], s1["layer_end"]) == (4, 8)
    assert s0["has_embeddings"] and not s0["has_head"]
    assert s1["has_head"] and not s1["has_embeddings"]

    t0 = load_file(str(out / "stage0" / "model.safetensors"))
    t1 = load_file(str(out / "stage1" / "model.safetensors"))
    # ownership: stage 0 = embeddings + layers 0-3, NO head; stage 1 mirror
    assert "model.embed_tokens.weight" in t0 and "lm_head.weight" not in t0
    assert "lm_head.weight" in t1 and "model.embed_tokens.weight" not in t1
    assert "model.layers.3.mlp.up_proj.weight" in t0
    assert "model.layers.3.mlp.up_proj.weight" not in t1
    assert "model.layers.4.self_attn.q_proj.weight" in t1

    # stage configs claim only their own depth + carry the range
    cfg1 = json.loads((out / "stage1" / "config.json").read_text())
    assert cfg1["num_hidden_layers"] == 4
    assert cfg1["panofabric_stage"] == {"stage": 1, "layer_start": 4,
                                        "layer_end": 8,
                                        "num_source_layers": 8,
                                        "engine_mode": False}
    # tokenizer sidecars replicated
    assert (out / "stage0" / "tokenizer.json").exists()

    # manifest sha256s verify against the bytes on disk
    for stg in manifest["stages"]:
        for name, digest in stg["files_sha256"].items():
            data = (out / f"stage{stg['stage']}" / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest

    # the split actually shrinks the per-stage footprint
    full = (src / "model.safetensors").stat().st_size
    for s in (0, 1):
        assert (out / f"stage{s}" / "model.safetensors").stat().st_size < full


def test_shard_checkpoint_engine_mode(tmp_path):
    """engine_mode: 0-based layer keys per stage, embed/norm/head replicated
    to every stage (each stage is a full standalone model for vLLM)."""
    src = tmp_path / "model"
    src.mkdir()
    _tiny_checkpoint(src, num_layers=8)
    out = tmp_path / "staged"
    shard_checkpoint(src, out, [10, 10], engine_mode=True)
    t0 = load_file(str(out / "stage0" / "model.safetensors"))
    t1 = load_file(str(out / "stage1" / "model.safetensors"))
    for t in (t0, t1):   # every stage is self-sufficient
        assert "model.embed_tokens.weight" in t
        assert "lm_head.weight" in t and "model.norm.weight" in t
    # stage 1 holds source layers 4-7 as LOCAL layers 0-3
    assert "model.layers.0.self_attn.q_proj.weight" in t1
    assert "model.layers.3.mlp.up_proj.weight" in t1
    assert not any(k.startswith("model.layers.4.") for k in t1)
    # stage tensors differ (different source layers) despite same keys
    assert not torch.equal(t0["model.layers.0.self_attn.q_proj.weight"],
                           t1["model.layers.0.self_attn.q_proj.weight"])


# -------------------------------------------------------------- transport -- #
def test_forward_hidden_states_backward_tokens():
    async def scenario():
        server = await st.listen("127.0.0.1", 0)
        up = await st.dial("127.0.0.1", server.port)     # upstream dials down
        down = await server.accept()
        try:
            # forward: hidden + residual COALESCED into one frame (one
            # latency payment — the splice's per-step hot path)
            hidden = torch.randn(4, 7, 64).to(torch.bfloat16)
            residual = torch.randn(4, 7, 64).to(torch.bfloat16)
            await up.send("hr", hidden, residual)
            tag, got = await down.recv()
            assert tag == "hr" and len(got) == 2
            assert got[0].dtype == torch.bfloat16 and got[0].shape == (4, 7, 64)
            assert torch.equal(got[0], hidden)
            assert torch.equal(got[1], residual)

            # backward on the SAME connection: sampled token ids
            tokens = torch.randint(0, 32000, (4,), dtype=torch.int64)
            await down.send("tok", tokens)
            tag, (got,) = await up.recv()
            assert tag == "tok" and torch.equal(got, tokens)

            # tensor-free frame: the connection-classifying hello
            await up.send("hello:ring")
            tag, got = await down.recv()
            assert tag == "hello:ring" and got == []

            # dtype fidelity across the common set
            for dt in (torch.float32, torch.float16, torch.int32, torch.bool):
                t = (torch.rand(3, 5) > 0.5) if dt == torch.bool \
                    else torch.ones(3, 5, dtype=dt)
                await up.send("x", t)
                _, (r,) = await down.recv()
                assert r.dtype == dt and torch.equal(r, t)
        finally:
            await up.close()
            await down.close()
            await server.close()

    asyncio.run(scenario())


def test_latency_injection_delays_sends():
    async def scenario():
        server = await st.listen("127.0.0.1", 0)
        up = await st.dial("127.0.0.1", server.port,
                           profile=st.LinkProfile(latency_ms=120))
        down = await server.accept()
        try:
            t0 = time.monotonic()
            await up.send("d", torch.zeros(2))
            await down.recv()
            assert time.monotonic() - t0 >= 0.11   # one-way injected delay
        finally:
            await up.close()
            await down.close()
            await server.close()

    asyncio.run(scenario())


def test_back_to_back_frames_pipeline_one_latency():
    """Propagation delays DELIVERY, not the sender: K frames sent together
    arrive ~one latency later, not K latencies (the old serialized-sleep
    model — the source of the measured ~9.8 latency payments per token on a
    6-link topology). Only the per-byte pacing term serializes."""
    async def scenario():
        server = await st.listen("127.0.0.1", 0)
        up = await st.dial("127.0.0.1", server.port,
                           profile=st.LinkProfile(latency_ms=120))
        down = await server.accept()
        try:
            t0 = time.monotonic()
            for i in range(4):
                await up.send(f"f{i}", torch.full((8,), float(i)))
            for i in range(4):        # in-order delivery preserved
                tag, (got,) = await down.recv()
                assert tag == f"f{i}" and got[0] == float(i)
            took = time.monotonic() - t0
            assert took >= 0.11, f"latency not applied ({took:.3f}s)"
            assert took < 0.30, \
                f"frames serialized instead of pipelining ({took:.3f}s)"
        finally:
            await up.close()
            await down.close()
            await server.close()

    asyncio.run(scenario())


def test_concurrent_sends_never_interleave_frames():
    async def scenario():
        server = await st.listen("127.0.0.1", 0)
        up = await st.dial("127.0.0.1", server.port)
        down = await server.accept()
        try:
            tensors = {f"t{i}": torch.full((256, 256), float(i))
                       for i in range(8)}
            await asyncio.gather(*[up.send(tag, t)
                                   for tag, t in tensors.items()])
            for _ in range(8):
                tag, (got,) = await down.recv()
                assert torch.equal(got, tensors[tag])
        finally:
            await up.close()
            await down.close()
            await server.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------- weights -- #
@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    ranged = {"n": 0}

    class _H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def send_head(self):
            if self.headers.get("Range"):
                ranged["n"] += 1
                lo, hi = self.headers["Range"].removeprefix("bytes=").split("-")
                data = (root / self.path.lstrip("/")).read_bytes()
                body = data[int(lo): int(hi) + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                import io
                return io.BytesIO(body)
            return super().send_head()

        def do_HEAD(self):
            f = (root / self.path.lstrip("/"))
            self.send_response(200)
            self.send_header("Content-Length", str(f.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def log_message(self, *a):
            pass

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{port}", ranged
    srv.shutdown()


def test_chunked_fetch_verifies_and_resumes(server, tmp_path):
    root, base, ranged = server
    blob = bytes(range(256)) * 100_000            # ~25MB -> several chunks
    (root / "w.bin").write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()

    dest = tmp_path / "out" / "w.bin"
    assert fetch_file(f"{base}/w.bin", dest, digest) == dest
    assert dest.read_bytes() == blob
    assert ranged["n"] >= 3                       # actually chunked

    # verified resume: no re-download of a good file
    before = ranged["n"]
    fetch_file(f"{base}/w.bin", dest, digest)
    assert ranged["n"] == before

    # corrupt local file -> refetched, not trusted
    dest.write_bytes(b"garbage")
    fetch_file(f"{base}/w.bin", dest, digest)
    assert dest.read_bytes() == blob


def test_integrity_mismatch_refuses(server, tmp_path):
    root, base, _ = server
    (root / "bad.bin").write_bytes(b"actual bytes")
    wrong = hashlib.sha256(b"expected bytes").hexdigest()
    dest = tmp_path / "bad.bin"
    with pytest.raises(IntegrityError, match="sha256 mismatch"):
        fetch_file(f"{base}/bad.bin", dest, wrong)
    assert not dest.exists()                      # nothing poisoned


def test_manifest_fetch(server, tmp_path):
    root, base, _ = server
    files = {}
    for name in ("model.safetensors", "config.json"):
        data = name.encode() * 1000
        (root / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    out = fetch_manifest(base, files, tmp_path / "stage0")
    assert sorted(p.name for p in out) == ["config.json", "model.safetensors"]
    for p in out:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == files[p.name]


# ------------------------------------------------------------- WAN sim -- #
def test_link_profile_paces_and_jitters_deterministically():
    """The WAN link model: bandwidth pacing scales with byte count, jitter is
    seeded (reproducible runs), zero-profile is a no-op."""
    import asyncio
    import time as _time

    from panoengine.serve.stage_transport import LinkProfile, dial, listen

    # deterministic jitter: same seed -> same delays
    a = [d for d, _ in zip(LinkProfile(10, 5, seed=7).delays(), range(4))]
    b = [d for d, _ in zip(LinkProfile(10, 5, seed=7).delays(), range(4))]
    c = [d for d, _ in zip(LinkProfile(10, 5, seed=8).delays(), range(4))]
    assert a == b and a != c
    assert all(0.010 <= pre < 0.015 for pre, _ in a)
    # bandwidth term: 1 Mbps -> 8e-6 s/byte
    _, per_byte = next(LinkProfile(bandwidth_mbps=1).delays())
    assert abs(per_byte - 8e-6) < 1e-9

    async def scenario():
        # 100KB over a 8 Mbps zero-latency link ~= 0.1s; unlimited ~= instant
        srv = await listen("127.0.0.1", 0,
                           profile=LinkProfile(bandwidth_mbps=8))
        up = await dial("127.0.0.1", srv.port,
                        profile=LinkProfile(bandwidth_mbps=8))
        down = await srv.accept()
        payload = torch.zeros(100_000 // 4, dtype=torch.float32)  # ~100KB
        t0 = _time.monotonic()
        await up.send("x", payload)
        await down.recv()
        paced = _time.monotonic() - t0
        assert paced >= 0.09, f"bandwidth pacing missing ({paced:.3f}s)"
        await up.close()
        await down.close()
        await srv.close()

    asyncio.run(scenario())


def test_hello_protocol_classifies_up_and_ring_links():
    """The splice boot handshake, GPU-free: a 3-stage topology's LAST stage
    listens once; its upstream neighbor introduces itself as "up" (learning
    from the ack that its next hop IS the last stage) and stage 0's ring
    dial is classified off the same listener. Tokens then return to stage 0
    in ONE hop, and tag-checked recv catches lockstep violations loudly."""
    from panoengine.serve.engine_stage import SyncLink

    last_up = SyncLink()                      # the last stage's listener side
    ring_box: dict = {}

    def on_ring(link, loop):
        ring_box["link"] = SyncLink.adopt(link, loop)

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    listener = threading.Thread(
        target=lambda: last_up.listen_classify(
            port, "last", want_ring=True, on_ring=on_ring),
        daemon=True)
    listener.start()

    mid_down = SyncLink()                     # stage 1 dials its next hop
    assert mid_down.dial_hello("127.0.0.1", port, "up") == "last"
    listener.join(timeout=30)
    assert not listener.is_alive()

    ring = SyncLink()                         # stage 0 closes the ring
    assert ring.dial_hello("127.0.0.1", port, "ring") == "last"
    deadline = time.monotonic() + 10
    while "link" not in ring_box and time.monotonic() < deadline:
        time.sleep(0.01)
    last_ring = ring_box["link"]

    # last stage sends the sampled token straight to stage 0 (one payment)
    toks = torch.tensor([[7], [11]])
    last_ring.send("tok", toks)
    assert torch.equal(ring.recv("tok")[0], toks)
    # and the ordinary up link still works alongside it
    mid_down.send("hr", torch.ones(3), torch.zeros(3))
    h, r = last_up.recv("hr")
    assert torch.equal(h, torch.ones(3)) and torch.equal(r, torch.zeros(3))
    # tag mismatch = stages out of lockstep -> loud protocol error
    mid_down.send("tok", toks)
    with pytest.raises(RuntimeError, match="out of lockstep"):
        last_up.recv("hr")

    for lk in (mid_down, ring, last_ring, last_up):
        lk.close()


def test_recv_any_waits_without_link_timeout():
    """Regression: the ring reader idles indefinitely between requests, so
    recv_any must wait with timeout=None — a bounded wait killed the reader
    thread after _LINK_TIMEOUT_S idle and the orphaned recv coroutine then
    swallowed the first frame of the next request."""
    from panoengine.serve.engine_stage import SyncLink

    class FakeLink:
        async def recv(self):
            return "tok:0", []

        async def close(self):
            pass

    s = SyncLink()
    s._link = FakeLink()
    timeouts = []
    real_run = s._run

    def spy(coro, timeout="<default>"):
        timeouts.append(timeout)
        return real_run(coro) if timeout == "<default>" else real_run(
            coro, timeout)

    s._run = spy
    tag, tensors = s.recv_any()
    assert (tag, tensors) == ("tok:0", [])
    assert timeouts == [None]
    s.close()


# ---------------------------------------------------------------- gateway -- #


def _backend(name: str, *, healthy: bool = True):
    """A fake vLLM server: /health + an echoing chat endpoint."""

    async def health(request):
        return web.Response(status=200 if healthy else 503)

    async def chat(request):
        body = await request.json()
        return web.json_response({"backend": name, "echo": body,
                                  "auth": request.headers.getall(
                                      "Authorization", [])})

    app = web.Application()
    app.add_routes([web.get("/health", health),
                    web.post("/v1/chat/completions", chat)])
    return app


def test_routing_health_and_auth():
    async def scenario():
        b1, b2 = TestServer(_backend("b1")), TestServer(_backend("b2", healthy=False))
        await b1.start_server()
        await b2.start_server()
        u1 = f"http://127.0.0.1:{b1.port}"
        u2 = f"http://127.0.0.1:{b2.port}"
        gw = Gateway([u1, u2], api_keys=["k-test"])
        gw_server = TestServer(gw.app())
        await gw_server.start_server()
        try:
            await gw.probe_once()
            assert gw.healthy == {u1: True, u2: False}

            import aiohttp
            async with aiohttp.ClientSession() as s:
                gw_url = f"http://127.0.0.1:{gw_server.port}/v1/chat/completions"
                # no key -> 401
                async with s.post(gw_url, json={"m": 1}) as r:
                    assert r.status == 401
                # with key -> routed to the HEALTHY backend only
                hdrs = {"Authorization": "Bearer k-test"}
                for _ in range(3):
                    async with s.post(gw_url, json={"m": 1}, headers=hdrs) as r:
                        assert r.status == 200
                        assert (await r.json())["backend"] == "b1"
                # gateway health endpoint (no auth: it's operational, not /v1)
                async with s.get(
                    f"http://127.0.0.1:{gw_server.port}/gateway/health"
                ) as r:
                    payload = await r.json()
                    assert payload["targets"][u1]["healthy"] is True
                    assert payload["requests"] == 3 and payload["rejected"] == 1
                # with no upstream key the client's own bearer is forwarded
                async with s.post(gw_url, json={"m": 1}, headers=hdrs) as r:
                    assert (await r.json())["auth"] == ["Bearer k-test"]
        finally:
            await gw_server.close()
            await b1.close()
            await b2.close()

    asyncio.run(scenario())


def test_upstream_key_replaces_the_client_bearer():
    """The backends are launched with VLLM_API_KEY=<upstream key>, so a public
    replica port cannot bypass this gateway's api-key check. The client's own
    bearer must NOT reach them (nor ride along in a second header)."""
    async def scenario():
        b1 = TestServer(_backend("b1"))
        await b1.start_server()
        u1 = f"http://127.0.0.1:{b1.port}"
        gw = Gateway([u1], api_keys=["k-client"], upstream_key="k-internal")
        gw.healthy[u1] = True
        gw_server = TestServer(gw.app())
        await gw_server.start_server()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                url = f"http://127.0.0.1:{gw_server.port}/v1/chat/completions"
                # lowercase inbound header: the swap must not leave both
                async with s.post(url, json={"m": 1},
                                  headers={"authorization": "Bearer k-client"}) as r:
                    assert r.status == 200
                    assert (await r.json())["auth"] == ["Bearer k-internal"]
                # the client's key is still what the GATEWAY checks
                async with s.post(url, json={"m": 1},
                                  headers={"Authorization": "Bearer k-internal"}) as r:
                    assert r.status == 401
        finally:
            await gw_server.close()
            await b1.close()

    asyncio.run(scenario())


def test_least_inflight_pick_and_cold_start_fallback():
    gw = Gateway(["http://a", "http://b"])
    # cold start: nothing healthy -> fall back to all targets, least inflight
    gw.inflight["http://a"] = 3
    assert gw.pick() == "http://b"
    # only-healthy wins even with more inflight
    gw.healthy["http://a"] = True
    assert gw.pick() == "http://a"


def test_upstream_failure_returns_502_and_marks_unhealthy():
    async def scenario():
        dead = "http://127.0.0.1:9"   # discard port: connection refused
        gw = Gateway([dead])
        gw.healthy[dead] = True
        gw_server = TestServer(gw.app())
        await gw_server.start_server()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"http://127.0.0.1:{gw_server.port}/v1/chat/completions",
                    json={},
                ) as r:
                    assert r.status == 502
            assert gw.healthy[dead] is False
            assert gw.inflight[dead] == 0   # decremented on the error path
        finally:
            await gw_server.close()

    asyncio.run(scenario())


# ------------------------------------------------------- weights CLI -- #


def test_fetch_model_dir_verified(tmp_path):
    """The plain-island wrapper pulls a whole verified model dir from a
    manifest tree (manifest.json: filename -> sha256)."""
    root = tmp_path / "srv"
    root.mkdir()
    files = {}
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        data = name.encode() * 500
        (root / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    (root / "manifest.json").write_text(json.dumps(files))

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    handler = type("H", (http.server.SimpleHTTPRequestHandler,),
                   {"log_message": lambda *a: None})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.RequestHandlerClass = lambda *a, **kw: handler(
        *a, directory=str(root), **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = fetch_model_dir(f"http://127.0.0.1:{port}", tmp_path / "model")
        assert sorted(p.name for p in dest.iterdir()) == sorted(files)
        for name, digest in files.items():
            got = hashlib.sha256((dest / name).read_bytes()).hexdigest()
            assert got == digest
    finally:
        srv.shutdown()


def test_boot_waits_are_governed_by_link_timeout(monkeypatch):
    """The chain's BOOT waits — the listener's accept window and the dial
    retry window — must follow _LINK_TIMEOUT_S, not a hardcoded 600s.
    Regression: the tail stage's accept was pinned at 600s while cloud
    stages start minutes apart (image-pull variance alone was measured at
    13 min between same-region nodes), so the tail died before its upstream
    ever provisioned and took the whole run with it."""
    from panoengine.serve import engine_stage as es

    monkeypatch.setattr(es, "_LINK_TIMEOUT_S", 0.5)

    # Listener with no dialer: the accept window must expire in ~0.5s
    # (pre-fix: 600s — this test would hang ten minutes).
    ln = es.SyncLink()
    t0 = time.monotonic()
    with pytest.raises(Exception):     # TimeoutError via run_coroutine_threadsafe
        ln.listen_classify(0, "mid", False, lambda *_: None)
    assert time.monotonic() - t0 < 5

    # Dial against a dead port: the retry window must give up in ~0.5s.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    dl = es.SyncLink()
    t0 = time.monotonic()
    with pytest.raises(Exception):     # ConnectionError once retry_s expires
        dl.dial_hello("127.0.0.1", dead_port, "up")
    assert time.monotonic() - t0 < 5
