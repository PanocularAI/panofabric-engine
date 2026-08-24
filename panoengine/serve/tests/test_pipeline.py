

import pytest


def test_ctl_frame_roundtrip():
    """Control frames (admits/step/reload) survive the tensor transport
    encoding — the follower sees exactly what stage 0 broadcast."""
    # The frames ride uint8 tensors (torch.distributed.broadcast is the wire),
    # so the encoding itself needs torch — absent from control-plane CI, same
    # skip convention as test_serve.py / the engine-glue tests in test_runspec.
    pytest.importorskip("torch")
    from panoengine.serve.engine_stage import _ctl_decode, _ctl_encode

    frame = {"admits": [{"rid": "r0-v3", "token_ids": [1, 2, 3],
                         "max_tokens": 1}],
             "step": True, "reload": "http://relay/staged"}
    assert _ctl_decode(_ctl_encode(frame)) == frame


def test_ctl_wire_tp_fanout():
    """The TP-peer ctl fan-out (length prefix + payload over an in-place
    tensor broadcast): a peer rank reconstructs exactly the frame rank 0
    sent — including the idle heartbeat shape (step False, no admits)."""
    pytest.importorskip("torch")   # frames are length-prefixed uint8 tensors
    from panoengine.serve.engine_stage import _ctl_wire

    for frame in (
        {"admits": [{"rid": "r7", "prompt": "hi", "max_tokens": 8}],
         "step": True, "reload": None},
        {"admits": [], "step": False, "reload": "http://relay/staged"},
    ):
        wire = []                     # what rank 0 put on the group, in order
        assert _ctl_wire(frame, True, wire.append) == frame

        replay = iter(wire)

        def fill(t):                  # dist.broadcast semantics: in-place
            t.copy_(next(replay))

        assert _ctl_wire(None, False, fill) == frame


def test_reload_stage_weights_uses_vllm_load_weights(tmp_path):
    """Hot-swap goes through vLLM's own load_weights (HF-name mapping),
    feeding it exactly the shard's tensors."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from panoengine.serve.engine_stage import reload_stage_weights

    tensors = {"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
               "model.embed_tokens.weight": torch.ones(8, 4)}
    save_file(tensors, str(tmp_path / "model.safetensors"),
              metadata={"format": "pt"})

    seen = []

    class _FakeModel:
        def load_weights(self, it):
            seen.extend(name for name, _ in it)
            return set(seen)

    class _FakeMR:
        model = _FakeModel()

    n = reload_stage_weights(_FakeMR(), tmp_path)
    assert n == 2 and sorted(seen) == sorted(tensors)


def test_gateway_retarget_during_inflight_request():
    """A retarget that drops a target while a request to it is in flight must
    not corrupt the counters. The scheduler re-points the fleet after every
    recovery — exactly when requests to the dead island are in flight — and
    the naive `finally: inflight[target] -= 1` raised KeyError there (or, if
    the target came back, drove the count negative and biased routing)."""
    from panoengine.serve.gateway import Gateway

    gw = Gateway(["http://a:1", "http://b:2"], health_interval_s=3600)
    target = "http://a:1"
    gw.inflight[target] += 1                       # request in flight

    gw.retarget(["http://b:2", "http://c:3"])      # 'a' dropped mid-request
    # the in-flight request's finally-block equivalent
    if target in gw.inflight:
        gw.inflight[target] = max(0, gw.inflight[target] - 1)
    assert target not in gw.inflight               # not resurrected
    assert gw.inflight == {"http://b:2": 0, "http://c:3": 0}

    # dropped-then-readded must not go negative (which would pin routing)
    gw2 = Gateway(["http://a:1"], health_interval_s=3600)
    gw2.inflight["http://a:1"] += 1
    gw2.retarget(["http://b:2"])
    gw2.retarget(["http://a:1"])                   # back, counter reset to 0
    gw2.inflight["http://a:1"] = max(0, gw2.inflight["http://a:1"] - 1)
    assert gw2.inflight["http://a:1"] == 0
    assert min(gw2.inflight.values()) >= 0


def test_claim_reload_is_atomic():
    """_claim_reload hands the pending url to the lockstep thread exactly
    once: the old read-then-clear dropped a reload posted in between, after
    the API had already answered 'scheduled'."""
    import threading

    from panoengine.serve.engine_stage import SpliceServer

    srv = SpliceServer.__new__(SpliceServer)
    srv._pending_reload = "http://relay/m"
    srv._reload_lock = threading.Lock()
    assert srv._claim_reload() == "http://relay/m"
    assert srv._claim_reload() is None          # claimed once
