# Copyright (c) Panocular AI.
#
# Stage transport for the cross-site pipeline.
#
# Point-to-point async TCP between pipeline stages: hidden states flow
# forward, sampled token ids flow backward. ONE dial direction — stage i
# dials stage i+1, and stage 0 dials the LAST stage (the ring-return link),
# so the NAT'd side always dials out and both directions of every link ride
# that single connection; only the downstream side needs a reachable port.
#
# Wire format, one FRAME (one latency payment, any number of tensors):
#   [4B big-endian header length][header JSON][tensor bytes]...[tensor bytes]
# header: {"tag": str, "tensors": [{"dtype": "float16", "shape": [..],
#                                   "nbytes": int}, ...]}
# A frame may carry zero tensors (hello frames are header-only). Tensors are
# serialized as raw contiguous bytes (numpy buffer) — no pickle, nothing
# executable on the wire. TLS and WireGuard are ops layers above.
#
# Latency model (`LinkProfile`): one-way latency + seeded jitter delay each
# frame's DELIVERY, but frames PIPELINE — a frame sent 1ms after another
# arrives ~1ms after it, not a full latency later. That is how propagation
# delay behaves on a real WAN: only transmission serializes, modeled by the
# per-byte pacing term. The previous model slept the full latency inside
# each send() in sequence, so back-to-back frames each paid the whole
# latency — the measured ~9.8 latency payments per step where the topology
# has 6 links, and the reason shipping hidden+residual as two messages cost
# double.

from __future__ import annotations

import asyncio
import json
import random
import struct
from dataclasses import dataclass

_LEN = struct.Struct(">I")


@dataclass(frozen=True)
class LinkProfile:
    """Userspace WAN simulation for one link direction: fixed one-way
    latency + seeded jitter per frame, and byte-count pacing to model
    bandwidth. Deterministic per (seed, frame sequence) so simulated runs
    are reproducible. Zero values disable each effect; LinkProfile() is a
    perfect link."""

    latency_ms: float = 0.0
    jitter_ms: float = 0.0          # uniform [0, jitter_ms) added per frame
    bandwidth_mbps: float = 0.0     # 0 = unlimited
    seed: int = 0

    def delays(self):
        """Stateful per-frame delay generator: (propagation_s, per_byte_s)."""
        rng = random.Random(self.seed)
        per_byte = (8 / (self.bandwidth_mbps * 1e6)
                    if self.bandwidth_mbps else 0.0)
        while True:
            jitter = rng.uniform(0, self.jitter_ms) if self.jitter_ms else 0.0
            yield (self.latency_ms + jitter) / 1000.0, per_byte


_MAX_HEADER = 1 << 16
_MAX_TENSOR = 1 << 31   # 2 GiB per tensor: far above any hidden-state slab


def _to_wire(tensor) -> tuple[dict, bytes]:
    import torch
    t = tensor.detach().contiguous().cpu()
    # bfloat16 has no numpy dtype: ship its raw 2-byte words as int16
    view = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    payload = view.numpy().tobytes()
    return ({"dtype": str(t.dtype).removeprefix("torch."),
             "shape": list(t.shape), "nbytes": len(payload)}, payload)


def _from_wire(header: dict, payload: bytes):
    import numpy as np
    import torch
    dtype = getattr(torch, header["dtype"])
    if dtype == torch.bfloat16:
        t = torch.from_numpy(np.frombuffer(payload, dtype=np.int16).copy())
        return t.view(torch.bfloat16).reshape(header["shape"])
    np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    arr = np.frombuffer(payload, dtype=np_dtype).copy()
    return torch.from_numpy(arr).reshape(header["shape"])


class StageLink:
    """One bidirectional stage-to-stage connection. Symmetric API on both
    ends: `send(tag, *tensors)` / `tag, tensors = await recv()`.

    Sends enqueue to a single pump task that writes each frame at its
    propagation deadline (enqueue time + latency), FIFO — the delivery
    schedule of a real long-haul link. send() returns once the frame is
    queued, so a hook forwarding a token never stalls the compute behind
    it, and a token frame pipelines with the hidden-state frame that
    follows it on the same link."""

    def __init__(self, reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter, *,
                 profile: LinkProfile | None = None):
        self._reader = reader
        self._writer = writer
        self.profile = profile or LinkProfile()
        self._delays = self.profile.delays()
        self._recv_lock = asyncio.Lock()
        self._sendq: asyncio.Queue = asyncio.Queue()
        self._pump: asyncio.Task | None = None

    async def _pump_frames(self) -> None:
        loop = asyncio.get_event_loop()
        broken = False
        while True:
            due, buffers = await self._sendq.get()
            try:
                if not broken:
                    delay = due - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # buffers written as-is, no concatenation: payloads can
                    # be MBs and joining would memcpy the whole frame again
                    self._writer.writelines(buffers)
                    await self._writer.drain()
            except (ConnectionError, OSError):
                # peer gone: keep draining the queue (dropping frames) so
                # close()'s join() completes; the recv side reports the loss
                broken = True
            finally:
                self._sendq.task_done()

    async def send(self, tag: str, *tensors) -> None:
        metas, payloads = [], []
        for t in tensors:
            meta, payload = _to_wire(t)
            metas.append(meta)
            payloads.append(payload)
        hbytes = json.dumps({"tag": tag, "tensors": metas}).encode()
        propagation_s, per_byte_s = next(self._delays)
        nbytes = _LEN.size + len(hbytes) + sum(len(p) for p in payloads)
        # propagation delays this frame's delivery; transmission time scales
        # with the frame's bytes. The pump preserves wire order, so delayed
        # frames still arrive in send order.
        due = (asyncio.get_event_loop().time()
               + propagation_s + per_byte_s * nbytes)
        if self._pump is None:
            self._pump = asyncio.get_event_loop().create_task(
                self._pump_frames())
        self._sendq.put_nowait(
            (due, [_LEN.pack(len(hbytes)), hbytes, *payloads]))

    async def recv(self) -> tuple[str, list]:
        async with self._recv_lock:
            hlen = _LEN.unpack(await self._reader.readexactly(_LEN.size))[0]
            if hlen > _MAX_HEADER:
                raise ValueError(f"header too large: {hlen}")
            header = json.loads(await self._reader.readexactly(hlen))
            tensors = []
            for meta in header["tensors"]:
                if not 0 <= meta["nbytes"] <= _MAX_TENSOR:
                    raise ValueError(f"tensor too large: {meta['nbytes']}")
                payload = await self._reader.readexactly(meta["nbytes"])
                tensors.append(_from_wire(meta, payload))
        return header["tag"], tensors

    async def close(self) -> None:
        # flush queued frames first so a final token is never dropped
        if self._pump is not None:
            await self._sendq.join()
            self._pump.cancel()
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def listen(host: str, port: int, *,
                 profile: LinkProfile | None = None) -> "StageServer":
    """Downstream side: accept dial-ins (the upstream neighbor; on the last
    stage, also stage 0's ring-return dial)."""
    server = StageServer(profile=profile)
    server._server = await asyncio.start_server(server._on_connect, host, port)
    return server


class StageServer:
    def __init__(self, *, profile: LinkProfile | None = None):
        self._profile = profile
        self._server = None
        self._accepted: asyncio.Queue[StageLink] = asyncio.Queue()

    async def _on_connect(self, reader, writer):
        await self._accepted.put(
            StageLink(reader, writer, profile=self._profile))

    async def accept(self, timeout: float = 30.0) -> StageLink:
        return await asyncio.wait_for(self._accepted.get(), timeout)

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()


async def dial(host: str, port: int, *,
               profile: LinkProfile | None = None,
               retry_s: float = 30.0) -> StageLink:
    """Upstream side: dial the downstream stage, retrying while it boots."""
    deadline = asyncio.get_event_loop().time() + retry_s
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            return StageLink(reader, writer, profile=profile)
        except (ConnectionError, OSError):
            if asyncio.get_event_loop().time() > deadline:
                raise
            await asyncio.sleep(0.3)
