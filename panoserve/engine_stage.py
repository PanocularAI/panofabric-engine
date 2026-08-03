# Copyright (c) Panocular AI.
#
# vLLM-engine stage for the cross-site pipeline, built on the
# prime-vllm technique (github.com/PrimeIntellect-ai/prime-vllm):
#
# Each stage runs an INDEPENDENT, standalone vLLM engine (TP=1, PP=1 from
# vLLM's point of view) over an engine-mode sharder checkpoint (its layer
# range renumbered 0-based, embed/norm/head replicated). Stages are stitched
# with torch forward hooks — vLLM's own PP machinery (and its SupportsPP
# restriction) is never involved:
#
#   non-first stage: PRE-hook on layers[0]   <= inject upstream (hidden,
#                                               residual); positions local
#   non-last  stage: POST-hook on layers[-1] => send (hidden, residual) down,
#                                               ONE frame (one latency payment)
#   last      stage: sampler POST-hook       => send sampled tokens DIRECTLY to
#                                               stage 0 on the ring-return link
#   non-last  stage: sampler POST-hook       <= overwrite the LOCAL sampler's
#                                               tokens with the ring's real ones
#
# Token distribution: stage 0 receives the token straight off the last stage
# (one hop — the ring closes with stage 0 dialing the last stage, the same
# dial-out trick as every other link) and forwards it DOWNSTREAM for the
# middle stages' bookkeeping. The forward pipelines ahead of the next step's
# hidden-state frame on the same ordered link, so middles get their copy off
# the critical path — per-token cost is the post's N x latency floor, not the
# 2N+1 payments the old two-frames-per-hop + hop-by-hop upstream relay paid.
#
# Lockstep needs no shared scheduler: every stage calls the SAME
# llm.generate(prompts, sampling_params) with identical inputs, so every
# engine's scheduler advances the same sequences in the same order; each
# stage's KV cache covers only its own layers. Real KV cache + vLLM kernels,
# full-batch synchronous schedule (the blog's recommendation for
# memory-bandwidth-bound decode).
#
# vLLM version note: prime-vllm pins the V0 engine; here the V1 engine is
# driven in-process via VLLM_ENABLE_V1_MULTIPROCESSING=0 (verified on this
# tree: InprocClient -> EngineCore -> UniProcExecutor -> GPUModelRunner),
# with enforce_eager so the hooks fire every step.
#
#   python -m panoserve.engine_stage --stage-dir staged/stage1 \
#       --listen-port 9201 [--next host:9202] --prompts p.txt --out o.txt

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path

from . import stage_transport as tp

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# Upper bound on how long a hook may block on the transport. Covers a
# hot-swap: peers wait in their recvs while every stage fetches its shard.
_LINK_TIMEOUT_S = float(os.environ.get("PANOFABRIC_LINK_TIMEOUT_S", "3600"))

# "step: N" progress lines: every step for the first N (the supervisor's
# start_steps readiness gate needs a few), then every Nth.
_STEP_LOG_EVERY = 25


class SyncLink:
    """Blocking facade over a StageLink for use inside torch hooks (which
    run synchronously inside vLLM's forward): the asyncio loop runs in a
    daemon thread and hooks block on run_coroutine_threadsafe futures.

    Links dialed or accepted by the same stage may share one loop (pass
    `loop`): the last stage's up + ring links live on its listener's loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        if loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
        self._loop = loop
        self._link: tp.StageLink | None = None
        self._server: tp.StageServer | None = None

    # Generous: a stage blocks here across a hot-swap's shard fetch (a
    # multi-GB WAN download inside the step barrier), not just a token hop.
    def _run(self, coro, timeout=_LINK_TIMEOUT_S):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def dial_hello(self, host: str, port: int, role: str,
                   profile: "tp.LinkProfile | None" = None) -> str:
        """Dial out, introduce this connection's role ("up": the normal
        i -> i+1 link; "ring": stage 0's return link to the last stage), and
        return the listener's ack role ("last" | "mid") — how stage 0 learns
        whether its downstream neighbor IS the last stage."""
        async def _dh():
            link = await tp.dial(host, port, profile=profile, retry_s=600)
            await link.send(f"hello:{role}")
            tag, _ = await link.recv()
            return link, tag.split(":", 1)[1]
        self._link, ack = self._run(_dh())
        return ack

    def listen_classify(self, port: int, ack: str, want_ring: bool,
                        on_ring, profile: "tp.LinkProfile | None" = None,
                        ) -> None:
        """Accept dial-ins and classify each by its hello role. Returns once
        the "up" link (the upstream neighbor — every non-first stage gets
        exactly one) is connected; on the LAST stage a background acceptor
        keeps running for stage 0's ring dial, delivered via on_ring(link,
        loop). Boot-order-safe: ring-before-up and up-before-ring both work,
        and stage 0 blocks on the ring hello-ack, so the ring is classified
        here before any step can run."""
        async def _accept_one(server):
            link = await server.accept(timeout=600)
            tag, _ = await link.recv()
            role = tag.split(":", 1)[1]
            await link.send(f"hello-ack:{ack}")
            return role, link

        ring_seen = False

        async def _la():
            nonlocal ring_seen
            self._server = await tp.listen("0.0.0.0", port, profile=profile)
            while True:
                role, link = await _accept_one(self._server)
                if role == "up":
                    return link
                ring_seen = True
                on_ring(link, self._loop)

        async def _ring_bg():
            try:
                role, link = await _accept_one(self._server)
            except asyncio.TimeoutError:
                return       # no ring dial ever came (defensive; see below)
            if role == "ring":
                on_ring(link, self._loop)

        self._link = self._run(_la())
        if want_ring and not ring_seen:
            asyncio.run_coroutine_threadsafe(_ring_bg(), self._loop)

    @classmethod
    def adopt(cls, link: "tp.StageLink",
              loop: asyncio.AbstractEventLoop) -> "SyncLink":
        s = cls(loop)
        s._link = link
        return s

    def send(self, tag: str, *tensors) -> None:
        self._run(self._link.send(tag, *tensors))

    def recv(self, expect: str) -> list:
        """Receive one frame, asserting its tag: the per-link frame order is
        load-bearing (hr / tok / ctl strictly alternate by construction), so
        a mismatch is a protocol bug that must fail loudly here — consuming
        a mis-ordered frame as tensor data would silently corrupt lockstep."""
        tag, tensors = self._run(self._link.recv())
        if tag != expect:
            raise RuntimeError(
                f"stage link protocol error: expected {expect!r} frame, got "
                f"{tag!r} — stages are out of lockstep")
        return tensors

    def recv_any(self) -> tuple[str, list]:
        """Receive one frame, returning its tag. Wave mode's ring reader
        consumes token frames from ALL waves off one link, so the tag is
        data ("tok:2"), not an assertion."""
        return self._run(self._link.recv())

    def close(self) -> None:
        if self._link is not None:
            self._run(self._link.close())
        if self._server is not None:
            self._run(self._server.close())


def _model_runner(llm):
    """The verified V1 in-process access chain to the GPUModelRunner."""
    ec = llm.llm_engine.engine_core
    inner = getattr(ec, "engine_core", ec)
    dw = inner.model_executor.driver_worker
    return getattr(dw, "worker", dw).model_runner


class EngineStage:
    """One splice stage. tp>1 runs the stage TENSOR-PARALLEL: launch under
    `torchrun --nproc-per-node=N` and every rank builds its own EngineStage
    (vLLM external_launcher backend => in-process model runner per rank, so
    the hooks fire on every rank). Rank 0 owns the transport; received
    tensors are broadcast to the other TP ranks (sends need no gather —
    activations between decoder layers are replicated post-all-reduce)."""

    def __init__(self, stage_dir: Path, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 8192, tp: int = 1,
                 sync_scheduler: bool = False):
        from vllm import LLM

        self.tp = tp
        self.rank = int(os.environ.get("RANK", "0"))
        info = json.loads(
            (Path(stage_dir) / "config.json").read_text())["panofabric_stage"]
        if not info.get("engine_mode"):
            raise ValueError(
                f"{stage_dir} is a stripped (HF-runtime) shard; the engine "
                "stage needs sharder --engine-mode output")
        self.info = info
        self.stage_dir = Path(stage_dir)
        self.is_first = info["layer_start"] == 0
        self.is_last = info["layer_end"] == info["num_source_layers"]
        self.steps = 0
        self.up: SyncLink | None = None      # toward stage 0
        self.down: SyncLink | None = None    # toward the last stage
        # ring-return link (N >= 3 only): stage 0 <-> last stage, carrying
        # sampled tokens straight back in ONE hop. For N == 2 the down/up
        # pair IS the ring, so this stays None and the token paths fall back
        # to it. next_is_last drives the middles' token forwarding: the stage
        # before the last must NOT forward (the last samples the token itself
        # and never expects a tok frame on its up link).
        self.ring: SyncLink | None = None
        self.next_is_last = False
        # Wave-interleaved mode (SpliceServer --waves): the driver loop sets
        # this to the wave being stepped so the hooks tag/expect frames per
        # wave ("hr:2"); None = legacy lockstep (untagged frames, sampler
        # blocks for the ring token inside the step).
        self.current_wave: int | None = None
        kwargs = {}
        if tp > 1:
            kwargs = dict(tensor_parallel_size=tp,
                          distributed_executor_backend="external_launcher",
                          disable_custom_all_reduce=True)
        if sync_scheduler:
            # Wave mode masks the scheduler's queues around each subset
            # step; the (default) AsyncScheduler keeps one-step-behind
            # bookkeeping ACROSS step() calls, which masking corrupts
            # (KeyError in _update_after_schedule) and which defers admits
            # into later — wrong-wave — steps. The synchronous scheduler
            # completes schedule->execute->update inside each step() call.
            kwargs["async_scheduling"] = False
        self.llm = LLM(model=str(stage_dir), enforce_eager=True,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, dtype="bfloat16",
                       disable_log_stats=True, **kwargs)
        self._hooks = []

    def _tp_broadcast(self, template, received):
        """Share a transport-received tensor with the other TP ranks over
        vLLM's OWN TP-group communicator. Critical: NOT the default process
        group — issuing a collective on a second communicator that shares the
        forward pass's CUDA stream races vLLM's per-layer all-reduces and
        corrupts activations non-deterministically. `template` (a local
        tensor of the SAME shape/dtype/device the received data will take —
        the schedules are identical across the pipeline, so shapes match)
        gives every rank a buffer; rank 0 fills it, then all ranks broadcast
        on the TP group, which orders correctly against vLLM's collectives."""
        if self.tp == 1:
            return received
        from vllm.distributed.parallel_state import get_tp_group
        buf = template.clone()
        if self.rank == 0:
            buf.copy_(received.reshape(template.shape).to(template.device))
        get_tp_group().broadcast(buf, src=0)
        return buf

    def ctl_broadcast(self, frame: dict | None) -> dict:
        """Fan a serving ctl frame out to this stage's other TP ranks (rank 0
        passes the frame; peers pass None and receive it). Rides the TP
        group's CPU (gloo) communicator: no CUDA kernel spins while a peer
        is parked waiting between requests, and it cannot interleave with
        the forward pass's NCCL collectives."""
        if self.tp == 1:
            return frame
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        return _ctl_wire(frame, self.rank == 0,
                         lambda t: dist.broadcast(t, src=g.ranks[0],
                                                  group=g.cpu_group))

    def connect(self, *, listen_port: int | None, next_addr: str | None,
                ring_addr: str | None = None,
                profile: "tp.LinkProfile | None" = None) -> None:
        # downstream side listens; upstream dials (one dial direction) —
        # accept AFTER dialing our own downstream so a chain boots tail-first
        if self.rank != 0:
            return          # transport is rank 0's job; ranks sync via bcast
        if next_addr:
            host, port = next_addr.rsplit(":", 1)
            self.down = SyncLink()
            ack = self.down.dial_hello(host, int(port), "up", profile)
            self.next_is_last = ack == "last"
        if self.is_first and self.down is not None and not self.next_is_last:
            # N >= 3: close the ring by dialing the LAST stage directly (its
            # existing listener; the hello names this connection's role).
            # Same dial-out direction as every other link, so NAT posture is
            # unchanged. Blocking on the hello-ack here guarantees the last
            # stage has classified the ring before any step runs.
            if not ring_addr:
                raise RuntimeError(
                    "a splice with 3+ stages needs --ring <last stage "
                    "host:listen_port> on stage 0: sampled tokens return "
                    "directly instead of relaying hop-by-hop upstream")
            rhost, rport = ring_addr.rsplit(":", 1)
            self.ring = SyncLink()
            self.ring.dial_hello(rhost, int(rport), "ring", profile)

        def _on_ring(link, loop):
            self.ring = SyncLink.adopt(link, loop)

        if not self.is_first:
            self.up = SyncLink()
            self.up.listen_classify(listen_port,
                                    "last" if self.is_last else "mid",
                                    want_ring=self.is_last, on_ring=_on_ring,
                                    profile=profile)

    def install_hooks(self) -> None:
        import torch
        mr = _model_runner(self.llm)
        layers = mr.model.model.layers
        device = next(mr.model.parameters()).device

        if not self.is_first:
            def inject(module, args, kwargs):
                # decoder layer forward(positions, hidden_states, residual):
                # keep LOCAL positions, replace hidden/residual with the
                # upstream stage's output for this exact scheduler step. The
                # local hidden_states (computed from the replicated embeds)
                # is the shape/dtype/device template for the TP broadcast.
                local_h = (args[1] if len(args) >= 2
                           else kwargs["hidden_states"])
                hr_tag = ("hr" if self.current_wave is None
                          else f"hr:{self.current_wave}")
                if self.tp == 1:
                    h_recv, r_recv = self.up.recv(hr_tag)
                    h = h_recv.to(device)
                    r = r_recv.to(device) if r_recv.numel() else None
                else:
                    got = self.up.recv(hr_tag) if self.rank == 0 else (None, None)
                    h = self._tp_broadcast(local_h, got[0])
                    # non-first stages always carry a real residual
                    r = self._tp_broadcast(local_h, got[1])
                sig = list(args)
                if len(sig) >= 2:
                    sig[1] = h
                    if len(sig) >= 3:
                        sig[2] = r
                    return tuple(sig), kwargs
                kwargs["hidden_states"] = h
                if "residual" in kwargs:
                    kwargs["residual"] = r
                return args, kwargs

            self._hooks.append(layers[0].register_forward_pre_hook(
                inject, with_kwargs=True))

        if not self.is_last:
            def ship(module, args, output):
                if self.rank != 0:      # activations replicated: rank 0 sends
                    return output
                h, r = output if isinstance(output, tuple) else (output, None)
                # Materialize to CPU HERE, in the forward thread — its current
                # stream is the one that produced h (incl. TP all-reduces).
                # The transport thread runs on a different stream; handing it
                # a CUDA tensor is a cross-stream race (the original source
                # of the TP-splice non-determinism: .cpu() there read
                # not-yet-materialized all-reduce output).
                h = h.detach().cpu()
                r = r.detach().cpu() if r is not None else None
                # ONE frame for both tensors: a frame pays the link latency
                # once, and hidden+residual always travel together
                self.down.send(
                    "hr" if self.current_wave is None
                    else f"hr:{self.current_wave}",
                    h, r if r is not None else torch.empty(0))
                return output

            self._hooks.append(layers[-1].register_forward_hook(ship))

        # This nightly's V1 sampler is a plain class (not nn.Module — no hook
        # API), and the model runner looks it up per call: intercept by
        # replacing the ATTRIBUTE with a delegating proxy.
        stage = self

        class _SamplerTap:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __call__(self, *a, **k):
                out = self._inner(*a, **k)
                wave = stage.current_wave
                if stage.is_last:
                    if stage.rank == 0:
                        toks = out.sampled_token_ids.detach().cpu()
                        tag = "tok" if wave is None else f"tok:{wave}"
                        # direct to stage 0 (ring); N == 2: up IS stage 0
                        (stage.ring or stage.up).send(tag, toks)
                elif wave is not None:
                    # Wave mode: token bookkeeping is deferred/dropped. The
                    # local sample is a placeholder — non-first stages discard
                    # their embeddings (inject overwrites hidden at layer 0),
                    # and stage 0 patches the true ring token into the
                    # runner's token cache before this wave's NEXT step, so
                    # nothing downstream of this step ever consumes it. Stops
                    # ride ctl aborts from stage 0 (every request runs
                    # ignore_eos; only stage 0 sees true tokens).
                    if stage.rank == 0 and stage.is_first:
                        # sampler-row -> request mapping for this step: the
                        # driver keys the arriving ring token frame to these.
                        # This runner's sampler is called (logits,
                        # input_batch) — row order IS the batch's req order.
                        ib = a[1] if len(a) >= 2 else k["input_batch"]
                        n = out.sampled_token_ids.shape[0]
                        stage.last_rows = list(ib.req_ids[:n])
                else:
                    real = out.sampled_token_ids
                    if stage.rank == 0:
                        if stage.is_first:
                            # one hop off the ring (N == 2: off the pair link)
                            src = stage.ring or stage.down
                            got_cpu = src.recv("tok")[0]
                        else:
                            # middles: the token arrives from UPSTREAM, queued
                            # right behind the hr frame this step consumed —
                            # already in flight, ordered, zero extra wait
                            got_cpu = stage.up.recv("tok")[0]
                        if not stage.next_is_last:
                            # forward for the next middle's bookkeeping; the
                            # frame pipelines ahead of our next hr on the same
                            # link, off the critical path. The stage BEFORE
                            # the last never forwards: the last stage samples
                            # the token itself.
                            stage.down.send("tok", got_cpu)
                        got = got_cpu.reshape(real.shape).to(real.device)
                    else:
                        got = None
                    got = stage._tp_broadcast(real, got)
                    real.copy_(got)
                if stage.is_first:
                    # control-plane progress convention (the supervisor parses
                    # "step: N" for readiness/progress). Emitted for the first
                    # few steps, then throttled: one line per decode step
                    # buried the island's own logs in a long session.
                    stage.steps += 1
                    if (stage.steps <= _STEP_LOG_EVERY
                            or stage.steps % _STEP_LOG_EVERY == 0):
                        print(f"step: {stage.steps}", flush=True)
                return out

        self._orig_sampler = mr.sampler
        self._mr = mr
        mr.sampler = _SamplerTap(mr.sampler)

    def generate(self, prompts: list[str], max_tokens: int,
                 seed: int = 0) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed)
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        if getattr(self, "_orig_sampler", None) is not None:
            self._mr.sampler = self._orig_sampler
        for link in (self.up, self.down, self.ring):
            if link is not None:
                link.close()


def ensure_shard(staging: Path, stage: int, source_model: str,
                 stage_memories: list[float]) -> Path:
    """Idempotent local sharding for colocated splice islands: whichever
    stage process arrives first shards the source checkpoint into `staging`
    (file-locked); the rest wait for the manifest. Remote islands would
    fetch their pre-sharded stage dir via serve/weights.py instead."""
    import fcntl
    import time as _time

    from .sharder import shard_checkpoint

    staging.mkdir(parents=True, exist_ok=True)
    manifest = staging / "pipeline.json"
    lock = staging / ".shard.lock"
    # Shards are cached under a name-keyed dir, so a resubmission with a
    # DIFFERENT model/stage count/shape would silently serve the previous
    # run's weights. Fingerprint what the shards were built from and reshard
    # when it changes.
    fingerprint = json.dumps({"model": source_model,
                              "memories": stage_memories}, sort_keys=True)
    fp_path = staging / ".shard.fingerprint"
    with open(lock, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stale = (manifest.exists()
                     and (not fp_path.exists()
                          or fp_path.read_text() != fingerprint))
            if stale:
                import shutil as _sh
                print(f"resharding {staging}: shards were built from a "
                      "different model/shape", flush=True)
                for child in staging.iterdir():
                    if child.name not in (".shard.lock",):
                        _sh.rmtree(child, ignore_errors=True) if child.is_dir() \
                            else child.unlink(missing_ok=True)
            if not manifest.exists():
                from huggingface_hub import snapshot_download
                src = snapshot_download(source_model)
                shard_checkpoint(Path(src), staging, stage_memories,
                                 engine_mode=True)
                fp_path.write_text(fingerprint)
        except BlockingIOError:
            pass
    deadline = _time.time() + 1800
    while not manifest.exists():
        if _time.time() > deadline:
            raise TimeoutError(f"sharding never completed in {staging}")
        _time.sleep(2)
    return staging / f"stage{stage}"


def reload_stage_weights(model_runner, stage_dir: Path) -> int:
    """Hot-swap this stage's weights in place from a (freshly fetched,
    sha256-verified) engine-mode shard — the RL splice variant's core
    mechanic: each stage swaps ONLY its own layer range, between lockstep
    steps, so the pipeline serves new weights without a restart. Uses
    vLLM's own load_weights (HF-name mapping incl. fused params)."""
    from safetensors.torch import load_file

    tensors = load_file(str(Path(stage_dir) / "model.safetensors"))
    loaded = model_runner.model.load_weights(iter(tensors.items()))
    return len(loaded) if loaded is not None else len(tensors)


def _ctl_encode(obj) -> "object":
    import torch
    return torch.frombuffer(bytearray(json.dumps(obj).encode()),
                            dtype=torch.uint8).clone()


def _ctl_decode(tensor) -> dict:
    return json.loads(bytes(tensor.tolist()).decode())


def _ctl_wire(frame: dict | None, is_src: bool, bcast) -> dict:
    """Length-prefixed ctl-frame broadcast over a tensor-broadcast callable
    (torch.distributed.broadcast bound to a group: in-place fill on
    receivers). Pure wire logic — unit-testable without a process group."""
    import torch
    if is_src:
        payload = _ctl_encode(frame)
        bcast(torch.tensor([payload.numel()], dtype=torch.int64))
        bcast(payload)
        return frame
    n = torch.zeros(1, dtype=torch.int64)
    bcast(n)
    payload = torch.zeros(int(n.item()), dtype=torch.uint8)
    bcast(payload)
    return _ctl_decode(payload)


def _params_from(admit: dict, *, ignore_eos: bool = False):
    from vllm import SamplingParams
    return SamplingParams(max_tokens=admit["max_tokens"], temperature=0.0,
                          seed=0, ignore_eos=ignore_eos)


def _scheduler(llm):
    """The verified V1 in-process access chain to the Scheduler (the wave
    driver masks its running queue around each subset step)."""
    ec = llm.llm_engine.engine_core
    inner = getattr(ec, "engine_core", ec)
    return inner.scheduler


# idle serving cadence: stage 0 emits a no-op ctl frame this often so
# follower stages' transport recvs and TP peers' gloo broadcasts never sit
# unfed long enough to trip a communicator timeout
_HEARTBEAT_S = 10.0


class SpliceServer:
    """Streaming/serving splice (serving lockstep).

    Every stage runs the SAME manual engine loop; per iteration stage 0
    decides which new requests join and broadcasts a control frame down the
    chain: {"admits": [(id, prompt, max_tokens)...], "step": bool}. Each
    stage relays the frame, add_request()s the SAME ids/prompts/params in
    the SAME order, then step()s — so every V1 scheduler makes identical
    decisions with REAL continuous batching (requests join mid-flight).
    The existing forward hooks move hidden states / tokens during step().
    Stage 0 additionally runs an OpenAI-compatible HTTP server; greedy-only
    (temperature forced to 0) so the splice stays deterministic."""

    def __init__(self, stage: EngineStage, http_port: int | None, *,
                 admin_token: str = "", max_model_len: int = 8192,
                 model_name: str = "cross-site-pipeline", waves: int = 0):
        import queue as _q
        self.stage = stage
        self.eng = stage.llm.llm_engine
        self.http_port = http_port
        # waves >= 1: interleaved mode — requests are partitioned into
        # `waves` groups; each group steps independently so a stage computes
        # wave j while wave i's frames are on the wire, instead of the whole
        # pipeline idling through every ring traversal. 0 = legacy lockstep.
        self.waves = waves
        self.wave_of: dict[str, int] = {}       # rid -> wave (every stage)
        self._aborts_by_wave: dict[int, list[str]] = {}
        self.inbox: "_q.Queue" = _q.Queue()      # (rid, prompt, max_tokens)
        self.results: dict[str, "_q.Queue"] = {}
        self.seq = 0
        self.admin_token = admin_token
        self.max_model_len = max_model_len
        # What /v1/models advertises and responses echo. Clients (chat UIs)
        # show this in their model picker, so it must be the REAL repo id —
        # the pipeline is an implementation detail of how it is served.
        self.model_name = model_name
        # rids whose client vanished: dropped from every stage's engine on
        # the next frame (an abandoned request would otherwise decode to
        # max_tokens in lockstep, pinning the whole pipeline)
        self._aborts: list[str] = []
        self._pending_reload: str | None = None
        # _pending_reload crosses threads: the aiohttp thread schedules a
        # reload, the lockstep thread claims it. Without the lock a reload
        # posted between frame construction and the clear was dropped after
        # the API had already answered "scheduled".
        self._reload_lock = threading.Lock()

    def _drain_aborts(self) -> list[str]:
        with self._reload_lock:
            pending, self._aborts = self._aborts, []
        return pending

    def abort(self, rid: str) -> None:
        with self._reload_lock:
            self._aborts.append(rid)
        self.results.pop(rid, None)

    def _claim_reload(self) -> str | None:
        with self._reload_lock:
            pending, self._pending_reload = self._pending_reload, None
        return pending

    def _do_reload(self, manifest_url: str) -> None:
        import shutil as _sh
        import tempfile as _tf

        from .weights import fetch_stage
        # Fetch into a temp dir and ALWAYS remove it: a stage shard is
        # multi-GB and an RL run reloads on every weight push, so leaking
        # one per reload fills the disk within a handful of swaps.
        dest = Path(_tf.mkdtemp(prefix="stage-reload-"))
        try:
            fetch_stage(manifest_url, self.stage.info["stage"], dest)
            n = reload_stage_weights(_model_runner(self.stage.llm), dest)
        finally:
            _sh.rmtree(dest, ignore_errors=True)
        print(f"hot-swapped stage weights ({n} tensors) from "
              f"{manifest_url}", flush=True)

    # ------------------------- stage 0: HTTP ------------------------- #

    def _start_http(self) -> None:
        import queue as _q

        from aiohttp import web
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.stage.stage_dir)
        server = self

        async def chat(request):
            body = await request.json()
            # Validate HERE: an admitted request is broadcast to every stage
            # before any engine sees it, so a request vLLM rejects would take
            # down the whole pipeline instead of returning a 4xx.
            try:
                messages = body["messages"]
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                asked = body.get("max_tokens")
                asked = int(asked) if asked is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                return web.json_response(
                    {"error": f"invalid request: {exc}"}, status=400)
            n_prompt = len(tok(prompt).input_ids)
            # No max_tokens => fill the remaining context window, matching
            # vLLM's own OpenAI server. A small constant default silently
            # truncated every client that omits the field (chat UIs do),
            # cutting thinking models off inside their <think> block.
            room = server.max_model_len - n_prompt
            if room < 1:
                return web.json_response(
                    {"error": f"prompt ({n_prompt} tokens) fills the "
                              f"context window ({server.max_model_len})"},
                    status=400)
            max_tokens = room if asked is None else asked
            if not 1 <= max_tokens <= server.max_model_len:
                return web.json_response(
                    {"error": f"max_tokens must be 1..{server.max_model_len}"},
                    status=400)
            if n_prompt + max_tokens > server.max_model_len:
                return web.json_response(
                    {"error": f"prompt ({n_prompt} tokens) + max_tokens "
                              f"({max_tokens}) exceeds the engine's "
                              f"max_model_len ({server.max_model_len})"},
                    status=400)
            rid = f"r{server.seq}"
            server.seq += 1
            out_q: "_q.Queue" = _q.Queue()
            server.results[rid] = out_q
            server.inbox.put({"rid": rid, "prompt": prompt,
                              "max_tokens": max_tokens})
            loop = asyncio.get_event_loop()
            if body.get("stream"):
                resp = web.StreamResponse(headers={
                    "Content-Type": "text/event-stream"})
                await resp.prepare(request)
                try:
                    while True:
                        delta, fin = await loop.run_in_executor(None, out_q.get)
                        chunk = {"object": "chat.completion.chunk", "choices": [{
                            "index": 0, "delta": {"content": delta},
                            "finish_reason": "stop" if fin else None}]}
                        await resp.write(
                            f"data: {json.dumps(chunk)}\n\n".encode())
                        if fin:
                            await resp.write(b"data: [DONE]\n\n")
                            return resp
                except (ConnectionResetError, asyncio.CancelledError):
                    server.abort(rid)     # stop decoding on every stage
                    raise
            text = ""
            while True:
                delta, fin = await loop.run_in_executor(None, out_q.get)
                text += delta
                if fin:
                    return web.json_response({
                        "object": "chat.completion",
                        "model": self.model_name,
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": text}}]})

        async def reload_weights(request):
            # Replacing the served model is privileged: gate it on the
            # launch's admin token (the serving port binds 0.0.0.0 and the
            # fleet gateway proxies every path through to stage 0).
            auth = request.headers.get("Authorization", "")
            if (not server.admin_token
                    or auth != f"Bearer {server.admin_token}"):
                return web.json_response({"error": "admin token required"},
                                         status=401)
            body = await request.json()
            url = body.get("manifest_url")
            if not isinstance(url, str) or not url:
                return web.json_response({"error": "manifest_url required"},
                                         status=400)
            with server._reload_lock:
                server._pending_reload = url
            return web.json_response({"status": "reload scheduled at next "
                                                "step barrier"})

        async def models(request):
            return web.json_response({"object": "list", "data": [
                {"id": server.model_name, "object": "model",
                 "owned_by": "panofabric",
                 # how it is served, for anything that wants the detail
                 "panofabric_serving": "spliced pipeline"}]})

        app = web.Application()
        app.add_routes([web.post("/v1/chat/completions", chat),
                        web.post("/admin/reload", reload_weights),
                        web.get("/v1/models", models),
                        web.get("/health", lambda r: web.Response(text="ok"))])
        loop = asyncio.new_event_loop()

        def _serve():
            asyncio.set_event_loop(loop)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "0.0.0.0", self.http_port)
            loop.run_until_complete(site.start())
            loop.run_forever()

        threading.Thread(target=_serve, daemon=True).start()

    def _admit(self, admits: list[dict], *, ignore_eos: bool = False) -> None:
        for a in admits:
            self.eng.add_request(a["rid"], a.get("prompt") or {
                "prompt_token_ids": a["token_ids"]},
                _params_from(a, ignore_eos=ignore_eos))

    def _abort(self, rids: list[str]) -> None:
        """Drop requests from THIS stage's engine (called on every stage from
        the same ctl frame, so the schedulers stay identical)."""
        for rid in rids:
            try:
                self.eng.abort_request(rid)
            except Exception:      # noqa: BLE001 - already finished/unknown
                pass

    def _emit(self, frame: dict) -> None:
        """Send a ctl frame down the transport ring AND fan it out to this
        stage's TP peer ranks (each a no-op when absent)."""
        if self.stage.down is not None:
            self.stage.down.send("ctl", _ctl_encode(frame))
        self.stage.ctl_broadcast(frame)

    def _broadcast_step(self, admits: list[dict]) -> list:
        frame = {"admits": admits, "step": True,
                 "aborts": self._drain_aborts(),
                 "reload": self._claim_reload()}
        self._emit(frame)
        if frame["reload"]:
            self._do_reload(frame["reload"])
        self._abort(frame["aborts"])
        self._admit(admits)
        outs = self.eng.step()
        return outs

    def _fail_requests(self, admits: list[dict], exc: Exception) -> None:
        """Hand an admit-time error back to the waiting HTTP handlers instead
        of leaving them blocked on their result queues forever."""
        for a in admits:
            q = self.results.pop(a["rid"], None)
            if q is not None:
                q.put((f"\n[error: {exc}]", True))

    def _heartbeat(self) -> None:
        """No-op ctl frame emitted while idle: keeps follower/peer recvs
        inside their (gloo / socket) timeouts, and lets a hot-swap scheduled
        during an idle stretch propagate without waiting for a request."""
        frame = {"admits": [], "step": False,
                 "aborts": self._drain_aborts(),
                 "reload": self._claim_reload()}
        self._emit(frame)
        # apply the broadcast aborts to THIS stage's engine too — followers
        # abort on the frame, and an asymmetric abort desyncs the schedulers
        self._abort(frame["aborts"])
        for rid in frame["aborts"]:
            self.wave_of.pop(rid, None)
        if frame["reload"]:
            self._do_reload(frame["reload"])

    # ------------------------- driver loops -------------------------- #

    def run_first(self) -> None:
        import queue as _q
        prev: dict[str, str] = {}
        while True:
            admits = []
            # block while idle so the chain sleeps between requests, waking
            # every _HEARTBEAT_S to emit a no-op frame down the ring
            if not self.eng.has_unfinished_requests():
                try:
                    admits.append(self.inbox.get(timeout=_HEARTBEAT_S))
                except _q.Empty:
                    self._heartbeat()
                    continue
            while not self.inbox.empty():
                try:
                    admits.append(self.inbox.get_nowait())
                except _q.Empty:
                    break
            for a in admits:
                prev[a["rid"]] = ""
            try:
                outs = self._broadcast_step(admits)
            except Exception as exc:      # noqa: BLE001 - never kill the ring
                # The frame is already broadcast, so every stage is in the
                # same state; report to the clients and keep serving rather
                # than taking the whole pipeline down.
                print(f"step failed: {exc!r}", flush=True)
                self._fail_requests(admits, exc)
                continue
            for out in outs:
                text = out.outputs[0].text
                delta = text[len(prev.get(out.request_id, "")):]
                prev[out.request_id] = text
                if delta or out.finished:
                    q = self.results.get(out.request_id)
                    if q is not None:
                        q.put((delta, out.finished))
                if out.finished:
                    prev.pop(out.request_id, None)
                    self.results.pop(out.request_id, None)

    def run_follower(self) -> None:
        while True:
            frame = _ctl_decode(self.stage.up.recv("ctl")[0])
            if self.stage.down is not None:
                self.stage.down.send("ctl", _ctl_encode(frame))
            self.stage.ctl_broadcast(frame)     # local TP peers, if any
            if frame.get("reload"):
                self._do_reload(frame["reload"])
            self._abort(frame.get("aborts") or [])
            self._admit(frame["admits"])
            if frame["step"]:
                self.eng.step()

    # --------------------- wave-interleaved drivers -------------------- #
    #
    # Same lockstep contract per wave (identical admits/steps on every
    # stage), but `waves` independent request groups rotate through each
    # stage's engine: while wave i's hidden states / token ride the WAN,
    # the stage steps wave j. Per-sequence latency stays ring-bound;
    # aggregate scales toward xwaves. Correctness rests on two facts:
    # non-first stages DISCARD local embeddings (inject overwrites hidden
    # at layer 0), so their placeholder token bookkeeping feeds nothing;
    # and stage 0 patches each true ring token into its runner's CPU token
    # cache before that wave's next step embeds it.

    def _step_wave(self, wave: int | None) -> list:
        """One engine step restricted to `wave`'s requests: other waves'
        running requests are stashed off the scheduler's queue around the
        step (their KV blocks stay allocated; V1 schedules from
        self.running each step, so masking it is sufficient)."""
        if wave is None:
            return self.eng.step()
        sched = self._sched
        mine, others = [], []
        for r in sched.running:
            # the engine mints an INTERNAL id "{external}-{8 hex}" per
            # request (v1 input_processor); our maps key by the external id
            ext = (getattr(r, "external_req_id", None)
                   or r.request_id.rsplit("-", 1)[0])
            (mine if self.wave_of.get(ext) == wave else others).append(r)
        # waiting must be masked too: the scheduler may defer an admit past
        # its own wave's step (budget, prefill throttling), and the NEXT
        # wave's schedule() would otherwise pull it into the wrong wave
        wq = sched.waiting
        kept, stashed = [], []
        while wq:
            r = wq.pop_request()
            ext = (getattr(r, "external_req_id", None)
                   or r.request_id.rsplit("-", 1)[0])
            (kept if self.wave_of.get(ext) == wave else stashed).append(r)
        for r in kept:
            wq.add_request(r)
        sched.running = mine
        self.stage.current_wave = wave
        try:
            return self.eng.step()
        finally:
            sched.running.extend(others)
            for r in stashed:
                wq.add_request(r)
            self.stage.current_wave = None

    def _patch_next_input(self, rid: str, tok: int) -> None:
        """Overwrite the placeholder the runner booked for `rid`'s last
        sampled token with the TRUE ring token, before the wave's next step
        gathers it into input_ids (V1 builds inputs from the CPU token
        cache each step, so patching here is sufficient and race-free —
        the wave cannot step again until this ran)."""
        rs = self.stage._mr.req_states
        row = rs.req_id_to_index.get(rid)
        if row is None:      # finished/aborted between steps
            return
        # This runner builds decode input ids from last_sampled_tokens (see
        # vllm/v1/worker/gpu/input_batch.py's prepare kernel), so overwriting
        # the placeholder here is patching exactly what the wave's next step
        # embeds. all_token_ids keeps the placeholder — irrelevant for greedy
        # serving (it feeds penalties/prompt-logprobs, which the splice
        # forces off), noted in case those ever get enabled.
        rs.last_sampled_tokens[row, 0] = tok

    def run_first_waves(self) -> None:
        import queue as _q

        from transformers import AutoTokenizer

        G = self.waves
        stage = self.stage
        self._sched = _scheduler(stage.llm)
        tok = AutoTokenizer.from_pretrained(stage.stage_dir)
        eos_id = tok.eos_token_id
        ring = stage.ring or stage.down

        row_q = {w: _q.Queue() for w in range(G)}   # per step: rid row order
        tok_q = {w: _q.Queue() for w in range(G)}   # arrived token tensors
        outstanding = [False] * G     # wave has a step in flight on the ring
        live: list[set] = [set() for _ in range(G)]
        true_ids: dict[str, list[int]] = {}
        prev_text: dict[str, str] = {}
        max_toks: dict[str, int] = {}

        def reader() -> None:
            while True:
                tag, tensors = ring.recv_any()
                tok_q[int(tag.split(":", 1)[1])].put(tensors[0])

        threading.Thread(target=reader, daemon=True).start()

        def absorb(w: int) -> None:
            """Wave w's ring token arrived: patch runners' view, stream true
            text, decide stops. Runs on the driver thread only."""
            toks = tok_q[w].get()
            if row_q[w].empty():
                # the matching step failed after the last stage had already
                # sampled (its frame still arrives): drop it, free the wave
                outstanding[w] = False
                return
            rids = row_q[w].get_nowait()
            for irid, t in zip(rids, toks.flatten().tolist()):
                # row rids are the runner's INTERNAL ids ("{external}-{8hex}",
                # v1 input_processor); our bookkeeping keys by external
                rid = irid.rsplit("-", 1)[0]
                if rid not in live[w]:
                    continue                      # aborted mid-flight
                true_ids[rid].append(t)
                self._patch_next_input(irid, t)
                text = tok.decode(true_ids[rid], skip_special_tokens=True)
                delta = text[len(prev_text[rid]):]
                prev_text[rid] = text
                eos = t == eos_id
                # length-finish is decided HERE, not from engine outputs:
                # this runner reports request completion asynchronously (a
                # step late), and there is no ring token after the last step
                # to flush a late finish against
                length_done = len(true_ids[rid]) >= max_toks[rid]
                finished = eos or length_done
                q = self.results.get(rid)
                if q is not None and (delta or finished):
                    q.put((delta, finished))
                if eos and not length_done:
                    # every stage runs ignore_eos: ONLY this loop sees true
                    # tokens, so the stop is ours to call — as a synchronized
                    # ctl abort on the request's own wave (the engines would
                    # otherwise decode to max_tokens)
                    self._aborts_by_wave.setdefault(w, []).append(rid)
                if finished:
                    live[w].discard(rid)
                    for m in (true_ids, prev_text, max_toks):
                        m.pop(rid, None)
                    self.results.pop(rid, None)
                    if length_done:
                        # engines self-finished this request in the same
                        # step everywhere — safe to forget its wave now
                        self.wave_of.pop(rid, None)
                    # EOS-only stop: the request keeps RUNNING on every
                    # engine (ignore_eos) until its abort frame lands, so
                    # the wave mapping must survive until that emission —
                    # dropping it now would misclassify the request out of
                    # its wave's steps and diverge stage 0 from followers
                    # (the abort-emission path pops it)
            outstanding[w] = False

        next_wave = 0
        while True:
            moved = False
            for w in range(G):
                while not tok_q[w].empty():
                    absorb(w)
                    moved = True
            # client-side aborts (HTTP disconnects) land in self._aborts
            # without a wave; route them
            for rid in self._drain_aborts():
                wv = self.wave_of.get(rid)
                if wv is not None:
                    self._aborts_by_wave.setdefault(wv, []).append(rid)
            # step the next eligible wave
            stepped = False
            for k in range(G):
                w = (next_wave + k) % G
                if outstanding[w]:
                    continue
                admits = []
                if not self.inbox.empty() and len(live[w]) == min(
                        len(live[v]) for v in range(G)):
                    # new requests join the emptiest eligible wave
                    while not self.inbox.empty():
                        try:
                            admits.append(self.inbox.get_nowait())
                        except _q.Empty:
                            break
                aborts = self._aborts_by_wave.pop(w, [])
                if not admits and not live[w] and not aborts:
                    continue
                frame = {"admits": admits, "step": True, "aborts": aborts,
                         "reload": self._claim_reload(), "wave": w}
                self._emit(frame)
                if frame["reload"]:
                    self._do_reload(frame["reload"])
                self._abort(aborts)
                for rid in aborts:
                    self.wave_of.pop(rid, None)   # now out of every engine
                for a in admits:
                    rid = a["rid"]
                    self.wave_of[rid] = w
                    live[w].add(rid)
                    true_ids[rid] = []
                    prev_text[rid] = ""
                    max_toks[rid] = a["max_tokens"]
                # ignore_eos everywhere: stage 0's own engine books
                # placeholder samples too, and a placeholder that happens to
                # be the EOS id must not end the request
                self._admit(admits, ignore_eos=True)
                stage.last_rows = None
                try:
                    outs = self._step_wave(w)
                except Exception as exc:  # noqa: BLE001 - never kill the ring
                    import traceback
                    traceback.print_exc()
                    print(f"step failed: {exc!r}", flush=True)
                    self._fail_requests(admits, exc)
                    for a in admits:
                        live[w].discard(a["rid"])
                    continue
                if stage.last_rows:
                    # sampler rows for this step, captured by the tap in
                    # batch order — the mapping key for the ring token frame.
                    # An abort-only step samples nothing: no frame will come,
                    # so the wave must NOT be marked in flight (deadlock).
                    row_q[w].put(list(stage.last_rows))
                    outstanding[w] = True
                del outs      # finish/streaming ride the ring tokens, not
                #               engine outputs (which lag a step on this
                #               runner and carry placeholder text anyway)
                next_wave = (w + 1) % G
                stepped = True
                break
            if not (moved or stepped):
                # idle or every wave in flight: wait briefly for a token
                # frame or a new request; heartbeat keeps peers' recvs fed
                try:
                    a = self.inbox.get(timeout=0.002 if any(
                        outstanding) else _HEARTBEAT_S)
                    self.inbox.put(a)     # re-queue; admitted next pass
                except _q.Empty:
                    if not any(outstanding):
                        self._heartbeat()

    def run_follower_waves(self) -> None:
        self._sched = _scheduler(self.stage.llm)
        try:
            while True:
                frame = _ctl_decode(self.stage.up.recv("ctl")[0])
                if self.stage.down is not None:
                    self.stage.down.send("ctl", _ctl_encode(frame))
                self.stage.ctl_broadcast(frame)     # local TP peers, if any
                if frame.get("reload"):
                    self._do_reload(frame["reload"])
                w = frame.get("wave")
                for a in frame["admits"]:
                    self.wave_of[a["rid"]] = w
                aborts = frame.get("aborts") or []
                self._abort(aborts)
                for rid in aborts:
                    self.wave_of.pop(rid, None)
                self._admit(frame["admits"], ignore_eos=True)
                if frame["step"]:
                    self._step_wave(w)
        except BaseException:
            # a dead follower silently stalls the whole chain: make the
            # cause unmissable in the island log before dying
            print("FOLLOWER DRIVER DIED — pipeline will stall", flush=True)
            raise

    def run_tp_peer(self) -> None:
        """Non-zero TP ranks of ANY stage. No transport, no HTTP — each ctl
        frame arrives from the stage's rank 0 over the TP cpu group; the
        rank mirrors the admit/step decisions so its engine participates in
        the stage's collectives (and the splice hooks) in lockstep."""
        if self.waves:
            self._sched = _scheduler(self.stage.llm)
        while True:
            frame = self.stage.ctl_broadcast(None)
            if frame.get("reload"):
                self._do_reload(frame["reload"])
            w = frame.get("wave")
            for a in frame["admits"]:
                self.wave_of[a["rid"]] = w
            self._abort(frame.get("aborts") or [])
            self._admit(frame["admits"], ignore_eos=self.waves > 0)
            if frame["step"]:
                self._step_wave(w) if self.waves else self.eng.step()


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM pipeline engine stage")
    p.add_argument("--stage-dir", required=True)
    p.add_argument("--listen-port", type=int, default=None)
    p.add_argument("--next", default=None, help="host:port of next stage")
    p.add_argument("--ring", default=None,
                   help="stage 0, 3+ stages: host:listen_port of the LAST "
                        "stage — closes the token-return ring in one hop")
    p.add_argument("--prompts", default=None,
                   help="file, one prompt per line — IDENTICAL on all stages")
    p.add_argument("--prompt", action="append", default=[],
                   help="inline prompt (repeatable); alternative to --prompts")
    p.add_argument("--ensure-from", default=None,
                   help="HF model id: shard it into --stage-dir's parent if "
                        "the stage checkpoint is missing (colocated splices)")
    p.add_argument("--manifest-url", default=None,
                   help="REMOTE-SITE weights: fetch this stage's shard "
                        "(sha256-verified) from a published manifest tree "
                        "instead of any local/shared checkpoint")
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="WAN sim: uniform jitter added to --latency-ms")
    p.add_argument("--bandwidth-mbps", type=float, default=0.0,
                   help="WAN sim: pace link sends to this bandwidth")
    p.add_argument("--stage", type=int, default=None,
                   help="stage index (with --ensure-from)")
    p.add_argument("--stage-memories", default=None,
                   help="comma-separated per-stage GiB (with --ensure-from)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel ranks for THIS stage (launch under "
                        "torchrun --nproc-per-node=N)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="context window per stage (prompt + output). 8192 "
                        "suits thinking models, whose <think> blocks alone "
                        "can exceed a 2k window")
    p.add_argument("--latency-ms", type=float, default=0.0)
    p.add_argument("--out", default=None, help="stage 0: write outputs")
    p.add_argument("--served-model-name", default="",
                   help="model id to advertise on /v1/models (defaults to "
                        "--ensure-from's repo id); clients show this")
    p.add_argument("--admin-token", default="",
                   help="bearer required by POST /admin/reload (weight "
                        "hot-swap); empty disables the route")
    p.add_argument("--serve-port", type=int, default=None,
                   help="SERVING splice: stage 0 exposes an OpenAI-compatible "
                        "server on this port (no --prompt needed); other "
                        "stages follow the broadcast control frames")
    p.add_argument("--waves", type=int, default=0,
                   help="serving splice: interleave requests across this "
                        "many independent wave groups so stages compute one "
                        "wave while another's frames ride the WAN (aggregate "
                        "throughput ~x waves; per-sequence speed unchanged). "
                        "0 = legacy lockstep. Must MATCH on every stage.")
    args = p.parse_args()

    stage_dir = Path(args.stage_dir)
    if args.manifest_url:
        from .weights import fetch_stage
        if args.stage is None:
            p.error("--manifest-url requires --stage")
        stage_dir = fetch_stage(args.manifest_url, args.stage, stage_dir)
    elif args.ensure_from:
        stage_dir = ensure_shard(
            stage_dir.parent, args.stage, args.ensure_from,
            [float(x) for x in (args.stage_memories or "").split(",") if x])

    stage = EngineStage(stage_dir,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=args.max_model_len, tp=args.tp,
                        sync_scheduler=bool(args.waves))
    stage.connect(listen_port=args.listen_port, next_addr=args.next,
                  ring_addr=args.ring,
                  profile=tp.LinkProfile(latency_ms=args.latency_ms,
                                         jitter_ms=args.jitter_ms,
                                         bandwidth_mbps=args.bandwidth_mbps,
                                         seed=args.stage or 0))
    stage.install_hooks()
    if args.serve_port is not None or (not args.prompt and not args.prompts
                                       and not stage.is_first):
        # serving splice: every stage runs the manual lockstep loop forever
        server = SpliceServer(stage, args.serve_port,
                              admin_token=(
                                  os.environ.get(
                                      "PANOFABRIC_RELOAD_ADMIN_TOKEN",
                                      args.admin_token)),
                              max_model_len=args.max_model_len,
                              model_name=(args.served_model_name
                                          or args.ensure_from
                                          or "cross-site-pipeline"),
                              waves=args.waves)
        if stage.rank != 0:
            # TP peers of any stage: no HTTP bind, no transport — mirror
            # rank 0's ctl frames (the crash in the wild: every rank ran
            # the rank-0 loops, racing the port bind / hitting a None link)
            server.run_tp_peer()
        elif stage.is_first:
            server._start_http()
            (server.run_first_waves if args.waves else server.run_first)()
        else:
            (server.run_follower_waves if args.waves
             else server.run_follower)()
        return
    prompts = list(args.prompt)
    if args.prompts:
        prompts += [ln for ln in Path(args.prompts).read_text().splitlines()
                    if ln.strip()]
    if not prompts:
        raise SystemExit("no prompts: pass --prompt or --prompts")
    texts = stage.generate(prompts, args.max_new_tokens)
    if stage.is_first:
        body = "\n---\n".join(texts)
        if args.out:
            Path(args.out).write_text(body)
        print(f"STAGE0_OUTPUT\n{body}", flush=True)
    stage.close()


if __name__ == "__main__":
    main()
