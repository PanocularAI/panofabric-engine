# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
AsyncDiLoCo — Real Asynchronous DiLoCo
=======================================
A central parameter server holds the authoritative global model weights and
applies the outer optimizer. Workers independently push pseudo-gradients
after every window of inner steps and pull updated global parameters back.

Reference: https://arxiv.org/pdf/2401.09135
"""

import dataclasses
from collections import OrderedDict

import json
import logging
import math
import os
import socket
import threading
import time
import urllib.error   # HTTPError: the 503 busy-retry path in _session_roundtrip
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler
from types import TracebackType
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import parse_qs, urlparse, urlunparse

import torch
import torch.distributed as dist
import torch.profiler
from torch import nn, optim
from torch.distributed.tensor import DTensor

from torchft.http import _IPv6HTTPServer
from torchft.parameter_server import _resolve_advertise_host

logger: logging.Logger = logging.getLogger(__name__)

_MAX_HEADER_BYTES: int = 1 << 16


def _full_value(p: torch.Tensor) -> torch.Tensor:
    """The parameter's value as ONE whole tensor.

    Plain tensors and whole/replicated DTensors return the local tensor
    (storage-shared, no communication). A sharded DTensor is materialized via
    ``full_tensor()`` — a collective over the parameter's mesh, so in replica
    mode every rank must reach this call for every parameter in the same
    order (they do: ``named_parameters()`` order is fixed per model).
    """
    if isinstance(p, DTensor):
        local = p.to_local()
        if tuple(local.shape) == tuple(p.shape):
            return local
        return p.full_tensor()
    return p


def _local_shard_slices(p: "DTensor") -> Tuple[slice, ...]:
    """Index of this rank's shard inside the parameter's GLOBAL tensor.

    Uses the DTensor placement metadata (never numel/world arithmetic: FSDP2
    pads nothing here, and uneven dims give ranks different shard sizes).
    """
    # Private torch API (present in the pinned 2.13 nightlies); flagged in the
    # update-submodules ledger. The public fallback would be reconstructing
    # offsets from p.placements by hand.
    from torch.distributed.tensor._utils import (
        compute_local_shape_and_global_offset,
    )

    shape, offset = compute_local_shape_and_global_offset(
        p.shape, p.device_mesh, p.placements
    )
    return tuple(slice(o, o + s) for o, s in zip(offset, shape))


def _read_exact(stream: BinaryIO, nbytes: int) -> bytearray:
    """Read exactly ``nbytes`` from a stream or raise on early EOF.

    Returns a WRITABLE bytearray of exactly ``nbytes``, filled in place, so
    :func:`_bytes_to_tensor` can wrap it with no copy at all.

    This matters because these payloads are whole-model-sized: a pseudo-gradient
    is 2.2 GiB fp32 for a 0.6B model, ~30 GiB for an 8B one. The previous
    grow-then-freeze form (``bytearray()`` + ``extend`` + ``bytes(buf)``) paid
    THREE full-size allocations per in-flight push on the parameter server — the
    growth buffer, the immutable copy, and then another bytearray inside
    _bytes_to_tensor — which is a large part of why a cloud PS OOM'd with two
    workers (panofabric docs/heloco-ps-memory.md).
    """
    buf = bytearray(nbytes)
    _read_exact_into(stream, memoryview(buf))
    return buf


def _read_exact_into(stream: BinaryIO, view: memoryview) -> None:
    """Fill ``view`` completely from ``stream`` or raise on early EOF.

    The zero-copy core of :func:`_read_exact` — and, given a memoryview over a
    tensor's own storage, the streaming request path's way of landing wire bytes
    DIRECTLY in their destination buffer with no intermediate allocation at all.
    """
    nbytes = view.nbytes
    off = 0
    # readinto avoids materializing each chunk; every stream we use here
    # (http.server's rfile, http.client's HTTPResponse) is a BufferedIOBase.
    readinto = getattr(stream, "readinto", None)
    while off < nbytes:
        if readinto is not None:
            n = readinto(view[off:])
        else:  # pragma: no cover - test doubles / exotic streams
            chunk = stream.read(nbytes - off)
            n = len(chunk)
            if n:
                view[off : off + n] = chunk
        if not n:
            raise IOError(
                f"connection closed after {off}/{nbytes} payload bytes"
            )
        off += n


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().numpy().tobytes()


def _bytes_to_tensor(
    data: Union[bytes, bytearray], dtype: torch.dtype
) -> torch.Tensor:
    """Wrap a buffer as a tensor, WITHOUT copying when the buffer allows it.

    ``torch.frombuffer`` needs a writable buffer and keeps a reference to it, so
    a bytearray (what :func:`_read_exact` now returns) is wrapped in place — the
    payload is never duplicated. Immutable ``bytes`` still has to be copied;
    only callers holding literal payloads (tests) pass those.
    """
    if not isinstance(data, bytearray):
        data = bytearray(data)
    return torch.frombuffer(data, dtype=dtype)


def _bf16_bytes(t: torch.Tensor) -> bytes:
    """fp32 tensor -> bf16 wire bytes (numel * 2). Bitcast through uint16 because
    numpy has no bfloat16; the bytes are the bf16 pattern either way."""
    return (
        t.detach().to(torch.bfloat16).contiguous().view(torch.uint16)
        .numpy().tobytes()
    )


def _quantize_int8(
    flat: torch.Tensor, numels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Blockwise symmetric int8 quantization of a flat fp32 buffer, one block
    per parameter tensor: ``scale_b = max|x_b| / 127``, ``q = round(x/scale)``.

    Per-element error is bounded by ``scale_b/2 = max|x_b|/254`` — relative
    to each block's own magnitude, following the DiLoCo int8 practice of
    quantizing only the worker→server pseudo-gradients.

    Returns ``(int8_values, fp32_scales)`` with one scale per block.
    """
    q = torch.empty(flat.numel(), dtype=torch.int8)
    scales = torch.empty(len(numels), dtype=torch.float32)
    offset = 0
    for i, n in enumerate(numels):
        chunk = flat[offset : offset + n]
        scale = chunk.abs().max().item() / 127.0
        if scale == 0.0:
            scale = 1.0  # all-zero block: any scale round-trips to zeros
        q[offset : offset + n] = (
            torch.round(chunk / scale).clamp_(-127, 127).to(torch.int8)
        )
        scales[i] = scale
        offset += n
    return q, scales


def _dequantize_int8(
    q: torch.Tensor, scales: torch.Tensor, numels: List[int]
) -> torch.Tensor:
    """Inverse of :func:`_quantize_int8`."""
    flat = torch.empty(q.numel(), dtype=torch.float32)
    offset = 0
    for i, n in enumerate(numels):
        flat[offset : offset + n] = q[offset : offset + n].float() * scales[i]
        offset += n
    return flat


def _fragment_bounds(
    numels: List[int], num_fragments: int
) -> List[Tuple[int, int]]:
    """Partition parameters into ``num_fragments`` contiguous, numel-balanced
    fragments (Decoupled DiLoCo, arXiv 2604.21428 — the paper bin-packs
    tensors; we keep fragments CONTIGUOUS in ``named_parameters()`` order so
    every fragment is a contiguous slice of the existing flat wire layout,
    and transformer parameters are homogeneous enough that balance is
    near-identical).

    Returns half-open ``(start_param_idx, end_param_idx)`` ranges covering all
    parameters. Deterministic from ``(numels, num_fragments)`` alone, so the
    client and server derive identical tables independently — the wire only
    carries the fragment INDEX, never the layout.
    """
    n_params = len(numels)
    if num_fragments < 1:
        raise ValueError(f"num_fragments must be >= 1, got {num_fragments}")
    if num_fragments > n_params:
        raise ValueError(
            f"num_fragments ({num_fragments}) exceeds the parameter tensor "
            f"count ({n_params})"
        )
    total = sum(numels)
    bounds: List[Tuple[int, int]] = []
    start = 0
    acc = 0
    for i, n in enumerate(numels):
        acc += n
        remaining_frags = num_fragments - len(bounds) - 1  # after this one
        if remaining_frags == 0:
            continue  # the last fragment takes everything that remains
        remaining_params = n_params - (i + 1)
        # Cut once the running numel crosses this fragment's ideal boundary,
        # or when every remaining fragment needs one of the remaining params.
        if (
            acc * num_fragments >= (len(bounds) + 1) * total
            or remaining_params == remaining_frags
        ):
            bounds.append((start, i + 1))
            start = i + 1
    bounds.append((start, n_params))
    return bounds


class DelayedNesterovOptimizer(optim.Optimizer):
    """
    Outer optimizer implementing Delayed Nesterov Momentum (DN) for async DiLoCo.

    Standard Nesterov applied to every worker push amplifies gradient staleness:
    each stale pseudo-gradient accumulates momentum in a stale direction. DN
    decouples the gradient application from the momentum correction:

      - Between milestones: each push applies ``ε · g / N`` (pure gradient,
        no momentum) so workers always receive updated params.
      - Every N-th push (milestone): additionally applies ``ε · β · m_new``
        (Nesterov correction) where ``m_new = β·m + avg_grad``.

    Total parameter change over N pushes equals ``-ε · (avg_grad + β · m_new)``,
    identical to standard Nesterov applied once to the *average* of N gradients.
    Momentum is never amplified by individual stale updates.

    Setting ``nesterov_period=1`` recovers standard Nesterov-SGD.

    **Async multi-worker note**: in a multi-worker server, ``grad_buffer``
    accumulates pushes from all workers interleaved. With heterogeneous workers
    a fast worker may dominate the buffer before a slow one contributes, making
    ``avg_grad`` at a milestone a cross-worker mixture rather than a single
    worker's window average. Set ``nesterov_period`` ≥ number of workers so
    each milestone includes at least one push per worker on average.

    Reference: Algorithm 3, "Asynchronous Local SGD" (2024),
        https://arxiv.org/abs/2401.09135
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
        nesterov_period: int = 10,
    ) -> None:
        """
        Args:
            params: Model parameters (same signature as any ``optim.Optimizer``).
            lr: Outer learning rate ε.
            momentum: Nesterov momentum coefficient β.
            nesterov_period: Number of worker pushes N between momentum
                corrections. Higher values reduce staleness amplification
                at the cost of less frequent momentum updates.
        """
        if nesterov_period < 1:
            raise ValueError("nesterov_period must be >= 1")
        defaults = dict(lr=lr, momentum=momentum, nesterov_period=nesterov_period)
        super().__init__(params, defaults)
        self._push_count: int = 0

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """
        Process one worker pseudo-gradient push.

        Expects ``p.grad`` to be set to the worker's pseudo-gradient before
        calling (same contract as ``AsyncDiLoCoServer._handle_sync``).
        """
        self._push_count += 1

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            N: int = group["nesterov_period"]
            is_milestone: bool = (self._push_count % N == 0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]

                if "grad_buffer" not in state:
                    state["grad_buffer"] = torch.zeros_like(p)
                    # float32 avoids precision loss when accumulating momentum across many pushes
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)

                state["grad_buffer"].add_(g)

                if is_milestone:
                    avg_grad = state["grad_buffer"] / N
                    m: torch.Tensor = state["m"]
                    m.mul_(beta).add_(avg_grad)
                    p.add_(-(g / N + beta * m), alpha=lr)
                    state["grad_buffer"].zero_()
                else:
                    # Pure gradient descent: apply 1/N fraction, no momentum
                    p.add_(-g / N, alpha=lr)


@dataclasses.dataclass
class _GraceBatch:
    """One grace-period aggregation window.

    Created by the first worker to arrive. Subsequent workers that arrive
    before ``deadline`` append their pseudo-gradients to ``grads_list``.
    The first thread whose ``deadline`` has passed becomes the *processor*:
    at claim time the batch is detached from the server (late arrivals open
    a fresh batch), so the processor can safely iterate ``grads_list``. It
    applies each worker's update sequentially (matching paper Algorithm 2:
    θ ← sync(θ, w.update) for each w in arrival order), fills ``snapshot_flat``
    / ``revision`` / ``pool_speed``, then publishes so every other thread in
    the batch can continue. If processing fails, ``error`` is published
    instead so waiters fail fast rather than hanging.
    """
    grads_list: List[Dict[str, torch.Tensor]]  # per-worker pseudo_grads, arrival order
    speeds: List[float]                         # per-worker speeds (for DyLU pool)
    deadline: float                             # time.monotonic() deadline
    claimed: bool = False                       # processor has been elected
    done: bool = False                          # results (or error) are ready
    error: Optional[str] = None                 # set if the processor failed
    snapshot_flat: Optional[torch.Tensor] = None
    revision: int = 0                           # global-model revision of snapshot
    pool_speed: float = 0.0                     # DyLU pool speed after step


class AsyncDiLoCoServer:
    """
    Central parameter server for AsyncDiLoCo.

    Stores the authoritative global model weights and outer optimizer state.
    Each worker performs one push-pull cycle per sync:
      1. Worker sends pseudo-gradients (outer_params - local_params)
      2. Server applies outer optimizer step
      3. Server sends updated global params back to the worker

    Thread-safe: concurrent worker sessions serialize around the optimizer step.
    The model passed here acts as the global (outer) model and should be on CPU.
    The outer_optimizer must reference this model's parameters.

    Every committed outer step increments the server's global-model
    *revision*. Workers track the revision their pseudo-gradient is relative
    to; a push whose baseline revision is ahead of the server's (possible only
    after the server restored from an older checkpoint) is rejected and the
    worker resyncs instead of silently corrupting the outer trajectory.

    **Transport**: each sync is a single worker-initiated HTTP ``POST /sync``
    carrying length-prefixed, coalesced buffers — no per-tensor round trips
    and no side-channel process group. Because the server never dials the
    worker, only worker→server reachability is required (NAT'd workers work),
    and everything (sync, heartbeat, status) is served on **one** port.

    Wire format (all parameter data flattened and concatenated in
    ``named_parameters()`` order, little-endian):
      - Request body: one JSON line
        ``{"flag": 0|1, "speed": float, "baseline_revision": int,
        "dtype": "float32"|"bfloat16"|"int8", "numel": int}``
        followed, when ``flag == 1`` (full sync; ``flag == 0`` is
        pull-only), by the raw pseudo-gradient payload:
          - ``dtype == "float32"``: ``numel × 4`` bytes.
          - ``dtype == "int8"`` (``should_quantize``): one fp32 scale per
            parameter tensor (``num_params × 4`` bytes, blockwise symmetric
            quantization) then ``numel`` int8 bytes.
        Fragment-wise sync (``num_fragments`` > 1) adds ``"fragment": int``
        and ``"num_fragments": int`` to a push's header; the payload then
        covers only that fragment's parameters (its contiguous
        ``named_parameters()`` slice — see :func:`_fragment_bounds`), with
        one int8 scale per parameter IN the fragment. Both ends must agree
        on ``num_fragments`` (validated per push). Pull-only requests are
        always whole-model.
      - Response body (200): one JSON line
        ``{"new_steps": int, "revision": int, "applied": bool, "numel": int}``
        followed by ``numel × 4`` raw float32 bytes of the global params —
        the pushed fragment's slice for a fragment push, the whole model
        otherwise (the download is never quantized — see
        ``should_quantize``).
        Failures are plain HTTP errors (500 processing / 503 at capacity),
        so a broken sync fails fast instead of wedging the worker.

    **Security**: HTTP here is unauthenticated plaintext. Run on a trusted,
    isolated network (VPC / WireGuard or similar); anyone who can reach the
    port can read or poison the model.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: optim.Optimizer,
        port: int = 0,
        bind_host: str = "",
        advertise_host: Optional[str] = None,
        max_sessions: int = 128,
        request_timeout: float = 600.0,
        dylu_H: int = 0,
        dylu_timeout: float = 300.0,
        dylu_percentile: float = 0.9,
        heartbeat_timeout: float = 15.0,
        should_quantize: bool = False,
        grace_period: float = 0.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 10,
        num_fragments: int = 1,
    ) -> None:
        """
        Args:
            model: The global (outer) model on CPU. Its parameters are the
                authoritative weights shared across all workers.
            outer_optimizer: Outer optimizer bound to ``model.parameters()``.
            port: HTTP port serving /sync, /heartbeat and /status
                (0 = OS-assigned). Set explicitly so the single port can be
                pre-opened in firewalls / security groups and advertised
                statically.
            bind_host: interface to bind the HTTP server to (default: all).
            advertise_host: hostname/IP workers use to reach this server.
                Defaults to ``$TORCHFT_PS_ADVERTISE_HOST`` if set, otherwise
                ``socket.gethostname()`` — set explicitly for any multi-host
                deployment.
            max_sessions: cap on concurrently processing sync sessions;
                requests beyond it receive 503 so a flood of workers cannot
                exhaust server threads/RAM.
            request_timeout: socket timeout in seconds for each sync request;
                a dead peer occupies a handler thread for at most this long.
                This bounds a STALL, not the transfer: it must cover the
                longest the peer can go without moving a byte, which on a
                shared hub is a GIL-starvation window, not a bandwidth
                figure. A 4B model syncs ~21 GB per roundtrip (4.4 GB int8
                up, 16.3 GB fp32 down) at 350-500 MB/s while the relay
                fans out 8.8 GB checkpoints from the same process -- the
                old 60 s default timed out BOTH ends mid-body and killed
                the run. Defaults to 600 s.
            dylu_H: Maximum local steps H for Dynamic Local Updates (DyLU).
                Per the paper (Eq. 6), each worker w is assigned
                ``floor(v(w) / v_ref * H)`` steps (capped at H) so slower
                workers finish each window in roughly the same wall-clock
                time as the fastest workers, where ``v_ref`` is a high
                percentile of the recent speed pool (see ``dylu_percentile``).
                Set to 0 (default) to disable DyLU; workers keep their own
                ``sync_every`` unchanged.
            dylu_timeout: Seconds after which a worker that has not synced
                is removed from the active set W. Defaults to 300 s.
            dylu_percentile: Percentile of the speed pool used as the DyLU
                reference speed. Using a high percentile instead of the max
                keeps a single mis-measured outlier window from shrinking
                every worker's window until ``dylu_timeout`` expires it.
                Defaults to 0.9.
            heartbeat_timeout: Seconds without a heartbeat before a worker
                is considered departed and removed from the active set.
                Workers send heartbeats every ``heartbeat_interval`` seconds
                (configured on the worker side; default 2 s). Defaults to 15 s.
            should_quantize: If True, receive worker pseudo-gradients as
                blockwise symmetric int8 over the wire (~4× upload
                bandwidth reduction; one fp32 scale per parameter tensor,
                per-element error ≤ ``max|Δ_b|/254`` within each block).
                The server→worker parameter download always stays float32:
                quantizing the authoritative params would compound error
                into every worker's baseline each sync — only the
                worker→server pseudo-gradients are quantized, following the
                DiLoCo int8 practice. Must match the worker's
                ``should_quantize`` setting.
            grace_period: Seconds the server waits after the first worker
                delivers pseudo-gradients before applying the outer step.
                Workers arriving within the window have their gradients
                averaged into a single outer step (§3.3 of the paper).
                0.0 (default) disables grace-period aggregation.
            checkpoint_path: File path for periodic server-state checkpoints
                (global model, outer optimizer state, revision). If the file
                exists at construction time, state is restored from it.
                None (default) disables checkpointing.
            checkpoint_every: Outer steps between checkpoints when
                ``checkpoint_path`` is set. Defaults to 10.
            num_fragments: Fragment-wise sync (Decoupled DiLoCo,
                arXiv 2604.21428). The model is partitioned into this many
                contiguous, numel-balanced fragments (:func:`_fragment_bounds`)
                and workers push/pull ONE fragment per (shortened) window on a
                staggered rotation — every transfer and per-push transient is
                model/num_fragments sized. Each fragment's outer step is exact:
                momentum, HeLoCo block correction and look-ahead are all
                per-parameter, so P fragment pushes commit bitwise the same
                state as one whole-model push of the same deltas. Must match
                the workers' ``num_fragments``. 1 (default) is the legacy
                whole-model protocol. Incompatible with ``grace_period`` > 0
                (grace batches whole-model gradient dicts).
        """
        if num_fragments > 1 and grace_period > 0.0:
            raise ValueError(
                "fragment-wise sync (num_fragments > 1) is incompatible with "
                "grace_period batching"
            )
        self._lock = threading.Lock()
        self._model = model
        self._outer_optimizer = outer_optimizer
        self._param_names: List[str] = []
        self._param_shapes: List[torch.Size] = []
        self._param_numels: List[int] = []
        for name, p in model.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(p.shape)
            self._param_numels.append(p.numel())
        self._total_numel: int = sum(self._param_numels)
        self._params_by_name: Dict[str, nn.Parameter] = dict(
            model.named_parameters()
        )
        self._num_fragments: int = num_fragments
        self._frag_bounds: List[Tuple[int, int]] = _fragment_bounds(
            self._param_numels, num_fragments
        )
        self._frag_numels: List[int] = [
            sum(self._param_numels[a:b]) for a, b in self._frag_bounds
        ]

        self._quantize: bool = should_quantize
        self._grace_period: float = grace_period
        self._grace_batch: Optional[_GraceBatch] = None
        self._grace_cond: threading.Condition = threading.Condition()
        self._dylu_H: int = dylu_H
        self._dylu_timeout: float = dylu_timeout
        self._dylu_percentile: float = dylu_percentile
        # DyLU speed pool: list of (v(w), timestamp) from all recent sessions.
        # Per-worker identity not needed — a pool percentile is sufficient.
        self._worker_speeds: List[Tuple[float, float]] = []
        # Heartbeat registry: worker_id → last_seen monotonic timestamp
        self._heartbeats: Dict[str, float] = {}
        # Workers that announced completion via /done. Kept apart from _heartbeats so a
        # finished worker can never be mistaken for a lost one (see the /done handler).
        self._finished: set[str] = set()
        self._heartbeat_timeout: float = heartbeat_timeout

        # Global-model revision: incremented on every committed outer step.
        self._revision: int = 0
        # Whole-model snapshots by revision, retained for delta downloads: a
        # worker asking for a delta names the revision it holds
        # (baseline_revision), and the delta is computed against the snapshot
        # SERVED at that revision. Bounded ring -- a baseline older than the
        # ring falls back to a full download, so eviction is never a
        # correctness problem, only a bandwidth one. 8 revisions of a 0.6B
        # model is ~19 GB on the hub.
        self._served: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._served_max = 8
        self._applied_pushes: int = 0
        self._last_step_time: Optional[float] = None  # wall clock, for /status
        # One snapshot per FRAGMENT shared by all sessions until that
        # fragment's next commit (K concurrent syncs no longer cost K
        # model-size clones; the caches sum to at most one model copy).
        # P=1: key 0 is the whole model — the legacy behavior exactly.
        self._snapshot_cache: Dict[int, Tuple[int, torch.Tensor]] = {}

        self._checkpoint_path: Optional[str] = checkpoint_path
        self._checkpoint_every: int = checkpoint_every
        self._last_checkpoint_revision: int = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self._advertise_host: str = _resolve_advertise_host(advertise_host)
        self._session_slots = threading.BoundedSemaphore(max_sessions)
        self._shutdown_event = threading.Event()
        # Streaming request path (grace_period == 0 only): pushes are read
        # chunk-by-chunk into ONE persistent, lazily-allocated buffer set instead
        # of fresh whole-model allocations per push. The buffers are shared, so
        # read+apply is serialized by this lock; that bounds the server's
        # per-push memory at ONE model copy total — independent of how many
        # workers push concurrently — at the cost of queueing uploads, a bounded
        # delay HeLoCo's staleness tolerance absorbs (it is also exactly what
        # max_sessions=1 would impose). Grace batching needs private per-worker
        # gradients, so it keeps the materializing path.
        self._stream_lock = threading.Lock()
        self._stream_bufs: Optional[Dict[str, torch.Tensor]] = None

        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            # Socket timeout for each request: bounds how long a dead peer
            # can park a handler thread (R7).
            timeout = request_timeout

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/sync":
                    self._respond(400, b"unknown path")
                    return
                if not server_ref._session_slots.acquire(blocking=False):
                    self.send_response(503)
                    self.send_header("Retry-After", "1")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                try:
                    header = json.loads(
                        self.rfile.readline(_MAX_HEADER_BYTES)
                    )
                    is_full_sync = bool(header["flag"])
                    flat_grads: Optional[torch.Tensor] = None
                    stream_bufs: Optional[Dict[str, torch.Tensor]] = None
                    # Streaming read (grace off): land the body in the shared
                    # persistent buffers instead of fresh whole-model
                    # allocations — see _stream_body_into_bufs. Grace batching
                    # holds several workers' gradients at once, so it keeps the
                    # materializing read below.
                    use_streaming = (
                        is_full_sync and server_ref._grace_period == 0.0
                    )
                    fragment: Optional[int] = None
                    if is_full_sync:
                        wire_dtype = header.get("dtype", "float32")
                        numel = int(header["numel"])
                        # A fragmented server accepts only fragment pushes at
                        # its own P; a P=1 server only the legacy whole-model
                        # header. One spec field drives both ends, so any
                        # mismatch is a config bug — fail loudly.
                        if "fragment" in header:
                            client_p = int(header.get("num_fragments", 0))
                            if client_p != server_ref._num_fragments:
                                raise ValueError(
                                    f"num_fragments mismatch: client "
                                    f"{client_p}, server "
                                    f"{server_ref._num_fragments}"
                                )
                            fragment = int(header["fragment"])
                            if not 0 <= fragment < server_ref._num_fragments:
                                raise ValueError(
                                    f"fragment index {fragment} out of range "
                                    f"(num_fragments="
                                    f"{server_ref._num_fragments})"
                                )
                            expected_numel = server_ref._frag_numels[fragment]
                        else:
                            if server_ref._num_fragments != 1:
                                raise ValueError(
                                    "whole-model push to a fragmented server "
                                    f"(num_fragments="
                                    f"{server_ref._num_fragments})"
                                )
                            expected_numel = server_ref._total_numel
                        if numel != expected_numel:
                            raise ValueError(
                                f"pseudo-gradient numel mismatch: got {numel}, "
                                f"expected {expected_numel}"
                            )
                        if use_streaming:
                            pass  # body is read under the stream lock below
                        elif wire_dtype == "int8":
                            scales = _bytes_to_tensor(
                                _read_exact(
                                    self.rfile,
                                    len(server_ref._param_numels) * 4,
                                ),
                                torch.float32,
                            )
                            q = _bytes_to_tensor(
                                _read_exact(self.rfile, numel), torch.int8
                            )
                            flat_grads = _dequantize_int8(
                                q, scales, server_ref._param_numels
                            )
                        elif wire_dtype == "bfloat16":
                            flat_grads = _bytes_to_tensor(
                                _read_exact(self.rfile, numel * 2),
                                torch.bfloat16,
                            ).to(torch.float32)
                        elif wire_dtype == "float32":
                            flat_grads = _bytes_to_tensor(
                                _read_exact(self.rfile, numel * 4),
                                torch.float32,
                            )
                        else:
                            raise ValueError(
                                f"unsupported wire dtype {wire_dtype!r}"
                            )

                    if use_streaming:
                        # Shared buffers: read + apply as one exclusive
                        # section. The lock is released before the response is
                        # written — the snapshot is an immutable per-revision
                        # copy, so a slow reader can't stall the next apply.
                        with server_ref._stream_lock:
                            stream_bufs = server_ref._stream_body_into_bufs(
                                self.rfile, wire_dtype, fragment
                            )
                            resp, snapshot_flat = server_ref._handle_sync(
                                is_full_sync=is_full_sync,
                                worker_speed=float(header.get("speed", 0.0)),
                                baseline_revision=int(
                                    header.get("baseline_revision", 0)
                                ),
                                flat_grads=None,
                                pseudo_grads=stream_bufs,
                                fragment=fragment,
                            )
                    else:
                        resp, snapshot_flat = server_ref._handle_sync(
                            is_full_sync=is_full_sync,
                            worker_speed=float(header.get("speed", 0.0)),
                            baseline_revision=int(
                                header.get("baseline_revision", 0)
                            ),
                            flat_grads=flat_grads,
                            fragment=fragment,
                        )

                    resp["numel"] = snapshot_flat.numel()
                    # bf16 download, sent only to a client that
                    # advertised `accept_dtype` in its push header, and the
                    # response always names its encoding in `dtype` (a client
                    # also uses the key's presence to learn this server takes
                    # bf16 uploads).
                    # fp32 on the wire carried precision the worker's bf16
                    # inner compute immediately discarded; the server's own
                    # master copy stays fp32, so each download's rounding is
                    # one bf16 ulp of the authoritative value, never
                    # cumulative.
                    # Delta download: the worker names the revision it holds
                    # and we ship only the CHANGE since then, int8-quantized --
                    # a delta is small-magnitude and zero-centered, so it earns
                    # the same treatment the pseudo-gradient upload gets. Only
                    # for whole-model exchanges (a fragment worker's baseline
                    # is mutated slice-by-slice and never matches a retained
                    # whole-model snapshot), and only while the baseline is
                    # still in the ring -- otherwise fall through to full.
                    base_rev = int(header.get("baseline_revision", -1))
                    delta_base = None
                    if (
                        header.get("accept_delta")
                        and fragment is None
                        and server_ref._num_fragments == 1
                    ):
                        with server_ref._lock:
                            delta_base = server_ref._served.get(base_rev)
                    if delta_base is not None:
                        resp["dtype"] = "delta_int8"
                        resp["delta_from"] = base_rev
                        q, scales = _quantize_int8(
                            snapshot_flat - delta_base,
                            server_ref._param_numels,
                        )
                        body_bytes = (
                            _tensor_to_bytes(scales) + _tensor_to_bytes(q)
                        )
                        payload = memoryview(body_bytes)
                    elif header.get("accept_dtype") == "bfloat16":
                        resp["dtype"] = "bfloat16"
                        body_t = (
                            snapshot_flat.contiguous()
                            .to(torch.bfloat16).view(torch.uint16)
                        )
                        payload = memoryview(body_t.numpy()).cast("B")
                    else:
                        resp["dtype"] = "float32"
                        # Zero-copy body: a memoryview over the snapshot's own
                        # storage; converting would duplicate the whole-model
                        # response (2.2 GiB at 0.6B) per in-flight reply. The
                        # snapshot is a per-revision immutable copy, so writing
                        # from it directly is safe even after the cache moves
                        # on — our reference keeps this revision alive.
                        payload = memoryview(
                            snapshot_flat.contiguous().numpy()
                        ).cast("B")
                    if fragment is None and server_ref._num_fragments == 1:
                        with server_ref._lock:
                            server_ref._served[resp["revision"]] = snapshot_flat
                            while len(server_ref._served) > server_ref._served_max:
                                server_ref._served.popitem(last=False)
                    head = (json.dumps(resp) + "\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header(
                        "Content-Length", str(len(head) + payload.nbytes)
                    )
                    self.end_headers()
                    self.wfile.write(head)
                    self.wfile.write(payload)
                except Exception as exc:
                    # Fail the sync fast with a plain HTTP error: the worker
                    # drops the push and resyncs instead of wedging.
                    logger.exception("sync session failed")
                    try:
                        self._respond(
                            500, f"{type(exc).__name__}: {exc}".encode()
                        )
                    except Exception:
                        pass  # peer already gone
                finally:
                    server_ref._session_slots.release()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/heartbeat":
                    wids = parse_qs(parsed.query).get("worker_id", [])
                    if not wids:
                        self._respond(400, b"missing worker_id")
                        return
                    worker_id = wids[0]
                    with server_ref._lock:
                        is_new = worker_id not in server_ref._heartbeats
                        server_ref._heartbeats[worker_id] = time.monotonic()
                        n = len(server_ref._heartbeats)
                    if is_new:
                        logger.info(f"Worker joined: {worker_id} ({n} active)")
                    self._respond(200, b"ok")
                elif parsed.path == "/done":
                    # A worker that finished its planned steps is NOT a lost worker.
                    # Without this the two are indistinguishable -- both simply stop
                    # heartbeating -- so a fast island that completed its run made the
                    # slower island's cohort gate block until it timed out.
                    wids = parse_qs(parsed.query).get("worker_id", [])
                    if not wids:
                        self._respond(400, b"missing worker_id")
                        return
                    with server_ref._lock:
                        server_ref._finished.add(wids[0])
                        server_ref._heartbeats.pop(wids[0], None)
                        n = len(server_ref._finished)
                    logger.info(f"Worker finished: {wids[0]} ({n} done)")
                    self._respond(200, b"ok")
                elif parsed.path == "/status":
                    self._respond(
                        200,
                        json.dumps(server_ref.status()).encode(),
                        content_type="application/json",
                    )
                else:
                    self._respond(400, b"unknown path")

            def _respond(
                self, code: int, body: bytes, content_type: str = "text/plain"
            ) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress default access-log noise

        self._server = _IPv6HTTPServer((bind_host, port), _Handler)
        self._server.daemon_threads = True
        threading.Thread(
            target=self._server.serve_forever, daemon=True
        ).start()
        logger.info(f"Started AsyncDiLoCoServer on {self.address()}")

        # Daemon monitor that logs join/leave events and evicts stale entries
        threading.Thread(
            target=self._run_heartbeat_monitor, daemon=True
        ).start()

    def _port(self) -> int:
        return self._server.socket.getsockname()[1]

    def address(self) -> str:
        """URL workers POST syncs to. Pass to :class:`AsyncDiLoCo`."""
        return f"http://{self._advertise_host}:{self._port()}/sync"

    def heartbeat_address(self) -> str:
        """
        Return the URL workers should send heartbeats to (same port as the
        sync endpoint — one port to open and advertise).

        Pass this to :class:`AsyncDiLoCo` as ``heartbeat_address``::

            server = AsyncDiLoCoServer(model, outer_opt)
            with AsyncDiLoCo(server.address(), model, inner_opt, sync_every=100,
                             heartbeat_address=server.heartbeat_address()):
                ...
        """
        return f"http://{self._advertise_host}:{self._port()}/heartbeat"

    def status_address(self) -> str:
        """URL of the JSON status endpoint (see :meth:`status`)."""
        return f"http://{self._advertise_host}:{self._port()}/status"

    def status(self) -> Dict[str, Any]:
        """
        Liveness/progress snapshot for external supervisors (also served as
        JSON at ``/status``).
        """
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._heartbeat_timeout
            active = {
                wid: round(now - ts, 3)
                for wid, ts in self._heartbeats.items()
                if ts >= cutoff
            }
            return {
                "active_workers": active,  # worker_id → heartbeat staleness (s)
                "worker_count": len(active),
                # Workers that completed their planned steps and left cleanly. A
                # cohort gate must count these as present: they are done, not lost.
                "finished_count": len(self._finished),
                "revision": self._revision,
                "applied_pushes": self._applied_pushes,
                "last_outer_step_time": self._last_step_time,
                "dylu_pool_size": len(self._worker_speeds),
                "num_fragments": self._num_fragments,
            }

    def shutdown(self) -> None:
        """Stop the HTTP server and the heartbeat monitor, releasing their
        threads and socket."""
        self._shutdown_event.set()
        self._server.shutdown()
        self._server.server_close()

    def _run_heartbeat_monitor(self) -> None:
        """
        Background daemon: evict workers whose heartbeats have expired and
        log the departure. Runs at half the heartbeat_timeout cadence so
        departures are detected within ~1.5× heartbeat_timeout of the last
        heartbeat (same trade-off as the Lighthouse eviction loop).
        """
        while not self._shutdown_event.wait(self._heartbeat_timeout / 2):
            departed: List[str] = []
            with self._lock:
                cutoff = time.monotonic() - self._heartbeat_timeout
                for wid, ts in list(self._heartbeats.items()):
                    if ts < cutoff:
                        del self._heartbeats[wid]
                        departed.append(wid)
                n = len(self._heartbeats)
            for wid in departed:
                logger.info(
                    f"Worker departed (no heartbeat): {wid} ({n} active)"
                )

    def active_workers(self) -> Dict[str, float]:
        """
        Return ``{worker_id: last_seen_timestamp}`` for all workers whose
        most recent heartbeat arrived within ``heartbeat_timeout`` seconds.

        The timestamp is from ``time.monotonic()``; subtract from the current
        ``time.monotonic()`` to get the staleness in seconds.
        """
        with self._lock:
            cutoff = time.monotonic() - self._heartbeat_timeout
            return {
                wid: ts
                for wid, ts in self._heartbeats.items()
                if ts >= cutoff
            }

    def worker_count(self) -> int:
        """Number of workers currently sending heartbeats."""
        return len(self.active_workers())

    # ------------------------------------------------------------------ #
    # Checkpointing (R3)                                                  #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str) -> None:
        """
        Atomically persist the server's authoritative state: global model,
        outer optimizer state, and revision. Tensors are cloned under the
        lock; the (slow) disk write happens outside it.
        """
        with self._lock:
            state = {
                "model": {
                    k: v.detach().clone()
                    for k, v in self._model.state_dict().items()
                },
                "outer_optimizer": self._outer_optimizer.state_dict(),
                "revision": self._revision,
                "applied_pushes": self._applied_pushes,
            }
            # Optimizer state tensors are references — clone before leaving the lock
            state["outer_optimizer"] = _clone_tensors(state["outer_optimizer"])
        tmp = f"{path}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info(f"Checkpointed server state at revision {state['revision']} to {path}")

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self._model.load_state_dict(state["model"])
        self._outer_optimizer.load_state_dict(state["outer_optimizer"])
        self._revision = state["revision"]
        self._applied_pushes = state["applied_pushes"]
        self._last_checkpoint_revision = self._revision
        logger.info(f"Restored server state at revision {self._revision} from {path}")

    def _maybe_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            return
        with self._lock:
            due = (
                self._revision - self._last_checkpoint_revision
                >= self._checkpoint_every
            )
            if due:
                # Claim before saving so concurrent sessions don't double-save
                self._last_checkpoint_revision = self._revision
        if due:
            try:
                self.save_checkpoint(self._checkpoint_path)
            except Exception:
                logger.exception("periodic checkpoint failed; training continues")

    # ------------------------------------------------------------------ #
    # Flat-buffer helpers (R1: one coalesced transfer per direction)      #
    # ------------------------------------------------------------------ #

    def _frag_names(self, fragment: Optional[int]) -> List[str]:
        """The parameter names a push covers: one fragment's contiguous slice,
        or every parameter for whole-model (``fragment=None``; P=1 legacy)."""
        if fragment is None:
            return self._param_names
        a, b = self._frag_bounds[fragment]
        return self._param_names[a:b]

    def _stream_body_into_bufs(
        self, rfile: BinaryIO, wire_dtype: str, fragment: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Read one push's body parameter-by-parameter into the persistent
        streaming buffers, returning them shaped like :meth:`_unflatten`'s dict.

        Caller must hold ``self._stream_lock`` (the buffers are shared) and have
        validated the header's numel. The values produced are BITWISE IDENTICAL
        to the materializing path: fp32 bytes land directly in the destination
        via readinto, and the int8 path applies the same per-parameter-block
        ``q.float() * scale`` that ``_dequantize_int8`` does — only the buffer
        they land in changes, from a fresh whole-model allocation per push to
        one reused set. Peak transient drops from O(model) per concurrent push
        to O(largest parameter) on the int8 path and O(socket chunk) on fp32.

        A fragment push fills (and returns) only that fragment's buffers; the
        rest of the lazily-allocated set is untouched.
        """
        if self._stream_bufs is None:
            self._stream_bufs = {
                name: torch.empty(shape, dtype=torch.float32)
                for name, shape in zip(self._param_names, self._param_shapes)
            }
        names = self._frag_names(fragment)
        numels = [self._params_by_name[n].numel() for n in names]
        bufs = {name: self._stream_bufs[name] for name in names}
        if wire_dtype == "int8":
            scales = _bytes_to_tensor(
                _read_exact(rfile, len(numels) * 4), torch.float32
            )
            for i, (name, n) in enumerate(zip(names, numels)):
                q = _bytes_to_tensor(_read_exact(rfile, n), torch.int8)
                torch.mul(q.float(), scales[i], out=bufs[name].view(-1))
        elif wire_dtype == "bfloat16":
            for name, n in zip(names, numels):
                bufs[name].view(-1).copy_(
                    _bytes_to_tensor(
                        _read_exact(rfile, n * 2), torch.bfloat16
                    )
                )
        elif wire_dtype == "float32":
            for name in names:
                flat = bufs[name].view(-1)
                _read_exact_into(
                    rfile, memoryview(flat.numpy()).cast("B")
                )
        else:
            raise ValueError(f"unsupported wire dtype {wire_dtype!r}")
        return bufs

    def _unflatten(
        self, flat: torch.Tensor, fragment: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Split one flat buffer into per-parameter views (no copies). With
        ``fragment`` set, the buffer covers only that fragment's slice."""
        a, b = (
            (0, len(self._param_names))
            if fragment is None
            else self._frag_bounds[fragment]
        )
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, numel in zip(
            self._param_names[a:b],
            self._param_shapes[a:b],
            self._param_numels[a:b],
        ):
            out[name] = flat[offset : offset + numel].view(shape)
            offset += numel
        return out

    def _frag_snapshot_locked(self, fragment: int) -> torch.Tensor:
        """One fragment's flat snapshot, built at most once per commit TO THAT
        FRAGMENT (a commit only moves its own fragment's params and momentum,
        so other fragments' cached snapshots stay valid — the caches sum to at
        most one model copy). Must hold ``self._lock``."""
        cached = self._snapshot_cache.get(fragment)
        if cached is not None:
            return cached[1]
        names = self._frag_names(fragment)
        snap = self._build_snapshot_locked(names)
        flat = torch.cat(
            [snap[name].detach().reshape(-1).float() for name in names]
        )
        self._snapshot_cache[fragment] = (self._revision, flat)
        return flat

    def _snapshot_flat(
        self, fragment: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Return ``(flat_params, revision)`` — one fragment's slice, or the
        whole model when ``fragment`` is None (pull-only requests and the
        P=1 legacy protocol, where fragment 0 IS the whole model).
        """
        with self._lock:
            if fragment is not None or self._num_fragments == 1:
                return self._frag_snapshot_locked(fragment or 0), self._revision
            flat = torch.cat(
                [
                    self._frag_snapshot_locked(f)
                    for f in range(self._num_fragments)
                ]
            )
            return flat, self._revision

    def _build_snapshot_locked(
        self, names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters to send back to workers, restricted to ``names`` (one
        fragment's slice, or all parameters). Must be called with
        ``self._lock`` held. Subclasses may override (e.g. HeLoCo's
        look-ahead shift).
        """
        return {name: self._params_by_name[name].data for name in names}

    # ------------------------------------------------------------------ #
    # Outer-step application                                              #
    # ------------------------------------------------------------------ #

    def _commit_step_locked(
        self, grads: Dict[str, torch.Tensor], fragment: int = 0
    ) -> None:
        """Apply one outer step over exactly the parameters in ``grads`` (one
        fragment's, or all of them). Must be called with ``self._lock`` held.

        ``zero_grad()`` leaves every other parameter's ``.grad`` as None and
        both outer optimizers skip None grads, so a single optimizer steps one
        fragment natively — momentum is per-parameter state.
        """
        with torch.no_grad():
            for name, g in grads.items():
                p = self._params_by_name[name]
                p.grad = g.to(p.dtype)
        self._outer_optimizer.step()
        self._outer_optimizer.zero_grad()
        self._revision += 1
        self._applied_pushes += 1
        self._last_step_time = time.time()
        self._snapshot_cache.pop(fragment, None)

    def _apply_one(
        self, pseudo_grads: Dict[str, torch.Tensor], fragment: int = 0
    ) -> None:
        """
        Apply one worker's pseudo-gradient as one outer step. Subclasses may
        override to transform the gradient first (e.g. HeLoCo block
        correction) as long as they end with ``_commit_step_locked``.
        """
        with self._lock:
            self._commit_step_locked(pseudo_grads, fragment)

    def _record_speeds_locked(self, speeds: List[float]) -> None:
        """Add worker speeds to the DyLU pool. Must hold ``self._lock``."""
        if self._dylu_H <= 0:
            return
        now = time.monotonic()
        for spd in speeds:
            if spd > 0:
                self._worker_speeds.append((spd, now))

    def _pool_speed_locked(self) -> float:
        """
        DyLU reference speed: expire stale entries, then take the configured
        percentile of the pool (robust to a single outlier, unlike max).
        Must hold ``self._lock``.
        """
        cutoff = time.monotonic() - self._dylu_timeout
        self._worker_speeds = [
            (s, ts) for s, ts in self._worker_speeds if ts >= cutoff
        ]
        if not self._worker_speeds:
            return 0.0
        speeds = sorted(s for s, _ in self._worker_speeds)
        idx = math.ceil(self._dylu_percentile * (len(speeds) - 1))
        return speeds[idx]

    def _dylu_steps(self, worker_speed: float, pool_speed: float) -> int:
        """Recommended local steps for a worker (paper Eq. 6, capped at H)."""
        if self._dylu_H > 0 and worker_speed > 0 and pool_speed > 0:
            return min(
                self._dylu_H,
                max(1, int(worker_speed / pool_speed * self._dylu_H)),
            )
        return self._dylu_H  # 0 → disabled

    # ------------------------------------------------------------------ #
    # Grace-period aggregation                                            #
    # ------------------------------------------------------------------ #

    def _grace_accumulate_and_wait(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
        worker_speed: float,
    ) -> Tuple[_GraceBatch, bool]:
        """Accumulate pseudo-grads into the current grace window and wait.

        Returns ``(batch, is_processor)``.  If ``is_processor`` is True the
        caller must process the batch and call :meth:`_grace_batch_publish`
        (with an error on failure) to unblock all waiting threads. At claim
        time the batch is detached from ``self._grace_batch`` so workers
        arriving after the processor election open a fresh batch instead of
        racing the processor's iteration of ``grads_list``.

        Non-processor threads return only after the batch is published; they
        must check ``batch.error``.
        """
        i_am_processor = False
        with self._grace_cond:
            now = time.monotonic()
            if self._grace_batch is None:
                self._grace_batch = _GraceBatch(
                    grads_list=[pseudo_grads],
                    speeds=[worker_speed],
                    deadline=now + self._grace_period,
                )
            else:
                self._grace_batch.grads_list.append(pseudo_grads)
                self._grace_batch.speeds.append(worker_speed)

            batch = self._grace_batch

            while not (batch.done or batch.claimed):
                remaining = batch.deadline - time.monotonic()
                if remaining <= 0:
                    batch.claimed = True
                    # Detach: late arrivals open a fresh batch; grads_list is
                    # now safe for the processor to iterate without the lock.
                    self._grace_batch = None
                    i_am_processor = True
                    break
                self._grace_cond.wait(timeout=remaining)

            # Non-processor: another thread claimed it — wait for it to publish
            if not i_am_processor:
                while not batch.done:
                    self._grace_cond.wait()

        return batch, i_am_processor

    def _grace_batch_publish(
        self, batch: _GraceBatch, error: Optional[str] = None
    ) -> None:
        """Mark batch done (optionally with an error) and wake all waiters."""
        with self._grace_cond:
            batch.error = error
            batch.done = True
            self._grace_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Sync processing                                                     #
    # ------------------------------------------------------------------ #

    @torch.profiler.record_function("async_diloco.handle_sync")
    def _handle_sync(
        self,
        is_full_sync: bool,
        worker_speed: float,
        baseline_revision: int,
        flat_grads: Optional[torch.Tensor],
        pseudo_grads: Optional[Dict[str, torch.Tensor]] = None,
        fragment: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Process one worker sync (transport-independent core; the HTTP handler
        does the framing — see the class docstring for the wire format).

        Subclasses reuse this unchanged and customize behavior via
        :meth:`_apply_one` and :meth:`_build_snapshot_locked`.

        ``pseudo_grads`` short-circuits the unflatten for the streaming path,
        whose per-parameter dict already exists (the shared streaming buffers —
        caller holds ``_stream_lock`` for the duration of this call).

        ``fragment`` scopes a fragment push: the gradients cover that
        fragment's parameters and the returned ``flat_params`` is its slice.
        ``None`` is a whole-model sync (the P=1 protocol, and every
        pull-only request).

        Returns ``({"new_steps", "revision", "applied"}, flat_params)``.
        """
        applied = False
        if is_full_sync:
            if pseudo_grads is None:
                assert flat_grads is not None
                # The HTTP handler already dequantized to fp32.
                pseudo_grads = self._unflatten(flat_grads, fragment)

            with self._lock:
                stale = baseline_revision > self._revision
            if stale:
                # Only possible after this server restored from an older
                # checkpoint: the pseudo-gradient is relative to params we no
                # longer have continuity with. Reject; the worker resyncs.
                logger.warning(
                    f"Rejecting push with baseline revision {baseline_revision} "
                    f"ahead of server revision {self._revision} "
                    "(server restored from checkpoint?)"
                )
                new_steps = self._dylu_H
                snapshot_flat, revision = self._snapshot_flat(fragment)
            elif self._grace_period > 0.0:
                batch, i_am_processor = self._grace_accumulate_and_wait(
                    pseudo_grads, worker_speed
                )

                if i_am_processor:
                    try:
                        # Update DyLU pool once for all workers in the batch
                        with self._lock:
                            self._record_speeds_locked(batch.speeds)
                            batch.pool_speed = self._pool_speed_locked()

                        # Apply each worker's update sequentially (paper
                        # Algorithm 2: θ ← sync(θ, w.update) in arrival order)
                        for grads in batch.grads_list:
                            self._apply_one(grads)

                        batch.snapshot_flat, batch.revision = self._snapshot_flat()
                    except Exception as exc:
                        self._grace_batch_publish(
                            batch, error=f"{type(exc).__name__}: {exc}"
                        )
                        raise
                    self._grace_batch_publish(batch)
                    self._maybe_checkpoint()

                if batch.error is not None:
                    # Fail this session too (HTTP 500) so its worker drops
                    # the push and resyncs.
                    raise RuntimeError(
                        f"grace batch processing failed: {batch.error}"
                    )

                applied = True
                snapshot_flat, revision = batch.snapshot_flat, batch.revision
                new_steps = self._dylu_steps(worker_speed, batch.pool_speed)
            else:
                with self._lock:
                    self._record_speeds_locked([worker_speed])
                    pool_speed = self._pool_speed_locked()
                self._apply_one(pseudo_grads, fragment or 0)
                applied = True
                snapshot_flat, revision = self._snapshot_flat(fragment)
                new_steps = self._dylu_steps(worker_speed, pool_speed)
                self._maybe_checkpoint()
        else:
            snapshot_flat, revision = self._snapshot_flat()
            new_steps = self._dylu_H

        return (
            {"new_steps": new_steps, "revision": revision, "applied": applied},
            snapshot_flat,
        )


def _clone_tensors(obj: Any) -> Any:
    """Recursively clone all tensors in a state-dict-like structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tensors(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tensors(v) for v in obj)
    return obj


@dataclasses.dataclass
class _InflightPush:
    """One background fragment exchange: launched at a fragment boundary,
    joined and adopted at the NEXT boundary (Decoupled DiLoCo's overlap —
    the roundtrip has a full fragment-window of inner steps to complete
    before anything blocks on it). Exactly one is outstanding at a time."""

    fragment: int
    thread: Optional[threading.Thread] = None
    # (flat_params, new_steps, revision, applied) from _session_roundtrip
    result: Optional[Tuple[torch.Tensor, int, int, bool]] = None
    error: Optional[BaseException] = None


class AsyncDiLoCo:
    """
    AsyncDiLoCo worker trainer.

    Wraps an inner training loop via a context manager. After every
    ``sync_every`` inner optimizer steps the worker:
      1. Computes pseudo-gradients: ``global_params - local_params``
      2. Pushes them to :class:`AsyncDiLoCoServer`
      3. Pulls the updated global parameters
      4. Resets the local model to the new global parameters

    Workers operate fully independently — no cross-worker communication. Each
    sync is a single worker-initiated HTTP request, so only worker→server
    reachability is required (workers may sit behind NAT).

    Fault tolerance: a failed sync never kills the training loop. The push is
    dropped, inner training continues on the current params, and the worker
    retries at subsequent window boundaries (with exponential backoff) using
    a pull-only resync once the server is reachable again.

    Example::

        server = AsyncDiLoCoServer(global_model, outer_optimizer)
        server_addr = server.address()

        with AsyncDiLoCo(server_addr, model, inner_optimizer, sync_every=100):
            for inputs, labels in dataloader:
                inner_optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                inner_optimizer.step()
    """

    def __init__(
        self,
        server_address: str,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        fragment_update_alpha: float = 0.0,
        heartbeat_address: Optional[str] = None,
        heartbeat_interval: float = 2.0,
        should_quantize: bool = False,
        reset_inner_state: bool = False,
        resync_backoff_max: float = 60.0,
        sync_timeout: float = 600.0,
        busy_retries: int = 10,
        min_replicas: int = 0,
        min_replicas_timeout: float = 600.0,
        replica_pg: Optional[dist.ProcessGroup] = None,
        num_fragments: int = 1,
        wire_bf16: bool = True,
    ) -> None:
        """
        Args:
            server_address: HTTP address returned by
                :py:meth:`AsyncDiLoCoServer.address`.
            model: The local worker model (may live on GPU).
            inner_optimizer: Inner optimizer applied every step.
            sync_every: Number of inner steps between server syncs.
            backup_device: Device for storing the global parameter backup.
                Defaults to CPU.
            fragment_update_alpha: Blends local and global params after each
                sync: ``p = (1 - alpha) * global + alpha * local``.
                ``alpha=0`` (default): full reset to global params (standard DiLoCo).
                ``alpha=1``: keep fully local params (outer optimizer has no effect).
            heartbeat_address: URL returned by
                :py:meth:`AsyncDiLoCoServer.heartbeat_address`.
                When provided, a daemon thread pings this endpoint every
                ``heartbeat_interval`` seconds so the server can track
                which workers are active. Pass ``None`` (default) to
                disable heartbeats.
            heartbeat_interval: Seconds between heartbeat pings to the server.
                Must be well below the server's ``heartbeat_timeout``.
                Defaults to 2 s.
            should_quantize: If True, upload pseudo-gradients as blockwise
                symmetric int8 (~4× upload bandwidth reduction; the
                parameter download stays float32 — see
                ``AsyncDiLoCoServer.should_quantize``). Must match the
                server's setting.
            reset_inner_state: If True, clear the inner optimizer state after
                every sync. Standard DiLoCo persists inner AdamW state across
                windows (that persistence is load-bearing for convergence),
                so this defaults to False; enable only if you have evidence
                the reset helps for your workload.
            resync_backoff_max: Cap in seconds on the exponential backoff
                between resync attempts while the server is unreachable.
            sync_timeout: Socket timeout in seconds for each sync request.
                Must exceed the server's ``grace_period`` (the server holds
                the response while aggregating the batch) AND the longest
                the server can go without writing a byte -- see
                ``AsyncDiLoCoServer.request_timeout`` for why that is a
                GIL-starvation window on a shared hub, not a bandwidth
                figure. Defaults to 600 s.
            busy_retries: How many times to re-send a push the server refused
                with 503 (all ``max_sessions`` slots busy), waiting the
                advertised ``Retry-After`` between attempts. Retrying is what
                makes a session cap safe: without it the refusal reaches the
                sync catch-all, which DROPS the push and re-baselines, losing
                the window's pseudo-gradient. Set 0 to restore that old
                behavior.
            replica_pg: REPLICA MODE — a (gloo) process group spanning every
                rank of one multi-GPU/multi-node replica whose model may be
                DTensor-sharded (FSDP/TP/2-D; never pipeline-split — PP
                changes ``named_parameters()`` itself). The replica then
                behaves as ONE parameter-server worker: rank 0 of the group
                owns the HTTP session, the heartbeat identity, and the
                full-model CPU backup; window boundaries gather full
                parameter values (``DTensor.full_tensor()``), rank 0 does the
                roundtrip, broadcasts a status word all control flow keys off
                (so ranks can never diverge on adopt/resync decisions), then
                broadcasts the pulled params parameter-by-parameter for each
                rank to slice its own shard from. Every rank of the replica
                must construct and drive this object in lockstep (same
                ``sync_every``, same step cadence). ``None`` (default) is the
                single-process behavior, byte-for-byte.
            num_fragments: Fragment-wise sync with communication overlap
                (Decoupled DiLoCo, arXiv 2604.21428). ``sync_every`` stays the
                FULL cycle: the model is split into this many contiguous,
                numel-balanced fragments and every ``sync_every /
                num_fragments`` inner steps ONE fragment (rotating) is pushed
                in a BACKGROUND thread; its merged result is adopted at the
                NEXT fragment boundary while the next fragment's exchange is
                in flight — the roundtrip has a full fragment-window of steps
                to complete before anything blocks on it. Requires
                ``sync_every % num_fragments == 0`` and must match the
                server's ``num_fragments``. 1 (default) is the legacy
                synchronous whole-model sync, byte-for-byte.
        """
        if num_fragments < 1:
            raise ValueError(
                f"num_fragments must be >= 1, got {num_fragments}"
            )
        if num_fragments > 1 and sync_every % num_fragments != 0:
            raise ValueError(
                f"sync_every ({sync_every}) must be divisible by "
                f"num_fragments ({num_fragments})"
            )
        self._server_address = server_address
        self._model = model
        self._inner_optimizer = inner_optimizer
        self._sync_every = sync_every
        self._fragment_update_alpha = fragment_update_alpha
        self._quantize = should_quantize
        # bf16 wire, negotiated: advertised in every header; a response that
        # NAMES its dtype proves the server bf16-capable, and uploads switch
        # too. Default on -- the exchange is link-limited so halving the bytes
        # halves the boundary, and the server's master copy stays fp32 so the 
        # rounding is one bf16 ulp
        # of the authoritative value, never cumulative. `wire_bf16=False`
        # keeps the wire bitwise fp32 (tests that assert exact equality;
        # debugging).
        self._wire_bf16 = wire_bf16
        self._server_bf16 = False
        # Delta downloads ride the same knob as bf16 (wire_bf16=False means a
        # bitwise-fp32 wire, full stop). _have_baseline flips once
        # _global_params holds a WHOLE adopted model; the refresh counter
        # forces a full download every _delta_refresh_every-th exchange, which
        # bounds the drift a chain of quantized deltas can accumulate (each
        # delta is exact against the snapshot the SERVER holds, but the worker
        # holds the quantized reconstruction of it -- see _session_roundtrip).
        self._have_baseline = False
        self._delta_refresh_every = 8
        self._deltas_since_full = 0
        self._reset_inner_state = reset_inner_state
        self._sync_timeout = sync_timeout
        self._local_step = 0
        self._hooks: List[Any] = []
        self._window_start: float = 0.0
        # Replica mode (see the docstring): rank 0 of `replica_pg` is the
        # replica's LEAD — the only rank that speaks to the server.
        self._replica_pg = replica_pg
        self._is_lead: bool = (
            replica_pg is None or dist.get_rank(replica_pg) == 0
        )
        self._lead_rank: Optional[int] = (
            None if replica_pg is None else dist.get_global_rank(replica_pg, 0)
        )
        backup = backup_device or torch.device("cpu")
        # The wire layout: DTensor .shape/.numel() are the GLOBAL shape, so
        # these match the server's own named_parameters() layout even when
        # this rank only holds a shard.
        self._param_names: List[str] = [
            name for name, _ in model.named_parameters()
        ]
        self._param_numels: List[int] = [
            p.numel() for _, p in model.named_parameters()
        ]
        self._total_numel: int = sum(self._param_numels)
        # Fragment-wise sync state: the SAME deterministic partition the
        # server derives (the wire carries only the fragment index).
        self._num_fragments: int = num_fragments
        self._frag_bounds: List[Tuple[int, int]] = _fragment_bounds(
            self._param_numels, num_fragments
        )
        self._frag_numels: List[int] = [
            sum(self._param_numels[a:b]) for a, b in self._frag_bounds
        ]
        self._frag_idx: int = 0
        self._inflight: Optional[_InflightPush] = None
        self._params_by_name: Dict[str, torch.Tensor] = dict(
            model.named_parameters()
        )
        # Full-model backup of the last-adopted global params: pseudo-gradients
        # are computed against it. Only the lead ever reads it, so followers
        # skip it entirely (a sharded rank couldn't cheaply fill it anyway).
        # Its CONTENT before the first pull is never read — __enter__ adopts
        # the server's params into it — so replica mode allocates empty; the
        # single-process path keeps the value snapshot for drivers that skip
        # __enter__ (e.g. the decentralized_rl HeLoCoRLClient pattern).
        self._global_params: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if replica_pg is None:
                    self._global_params[name] = p.detach().to(backup).clone()
                elif self._is_lead:
                    self._global_params[name] = torch.empty(
                        tuple(p.shape), dtype=p.dtype, device=backup
                    )

        # Revision of the global model our params are based on (see
        # AsyncDiLoCoServer: lets the server detect pushes computed against a
        # baseline it no longer has continuity with).
        self._baseline_revision: int = 0

        # Failed-sync recovery state (see _step_post_hook)
        self._pending_resync: bool = False
        self._resync_at: float = 0.0
        self._resync_backoff: float = 1.0
        self._resync_backoff_max: float = resync_backoff_max
        self._busy_retries: int = busy_retries
        # The window right after a resync started from stale params and an
        # unusual boundary — exclude it from DyLU speed measurement.
        self._skip_speed_report: bool = False

        # Heartbeat: persistent daemon thread pinging the server while in context.
        # Disabled if heartbeat_address is None.
        self._heartbeat_interval = heartbeat_interval
        # Unique per instance (uuid, not a module counter): every worker
        # process must register under a distinct id, hostname-prefixed for
        # readable logs.
        self._worker_id: str = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        # Replica mode: the LEAD is the replica's one worker identity — a
        # follower heartbeat would register the replica K times, and DyLU /
        # grace batching / rho=1/sqrt(K) all assume workers are replicas.
        if heartbeat_address is not None and self._is_lead:
            self._heartbeat_url: Optional[str] = (
                f"{heartbeat_address}?worker_id={self._worker_id}"
            )
        else:
            self._heartbeat_url = None
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        # Cohort gate (see _await_cohort). The status URL is the /sync address
        # with its path swapped -- the worker is handed /sync and /heartbeat,
        # never /status.
        self._min_replicas = int(min_replicas or 0)
        self._min_replicas_timeout = float(min_replicas_timeout)
        self._status_url: Optional[str] = (
            urlunparse(urlparse(server_address)._replace(path="/status", query=""))
            if server_address
            else None
        )

    def _await_cohort(self, startup: bool = True) -> None:
        """Block until ``min_replicas`` replicas are registered with the server.

        A registered "worker" in this module IS a replica -- only a replica's lead
        holds the HTTP session and heartbeat identity -- so the name matches
        torchft's lighthouse knob and the ``ft.min_replica_size`` spec field that
        feeds it. One vocabulary end to end.

        The parameter-server equivalent of torchft's lighthouse ``min_replicas``:
        heloco is asynchronous by design and a lone worker will happily train and
        push by itself, so a run whose whole point is cross-island synchronization
        can go green having synchronized NOTHING.

        Checked at STARTUP and again at every sync boundary, mirroring the
        lighthouse: ``min_replicas`` there gates every quorum, not just the first,
        so a cohort that shrinks mid-run stops training instead of quietly
        degrading to solo pushes. ``startup`` only changes the log wording.

        Counts replicas, not ranks: a 4-GPU island registers once.

        CAVEAT for multi-GPU replicas: the gate is lead-only, so while the lead
        polls here its FOLLOWERS are already waiting in the boundary collective.
        Keep ``min_replicas_timeout`` below the replica process-group timeout, or
        the followers abort the collective before the lead gives up. Single-GPU
        replicas have no followers and are unaffected.

        Raises on timeout rather than training alone. A verification run must fail
        loudly when its premise does not hold; an operator who prefers a solo run
        sets min_replicas=0 (the default) and gets exactly the old behavior.
        """
        if self._min_replicas <= 1 or not self._is_lead:
            return
        if self._status_url is None:
            logger.warning(
                "min_replicas=%d requested but the server address yields no /status "
                "URL; starting without waiting for the cohort", self._min_replicas
            )
            return
        deadline = time.monotonic() + self._min_replicas_timeout
        last_seen = -1
        while True:
            count = -1
            try:
                with urllib.request.urlopen(self._status_url, timeout=10.0) as resp:
                    snap = json.loads(resp.read())
                    # finished_count matters as much as worker_count: an island that
                    # completed its steps has left for a GOOD reason, and blocking on
                    # it deadlocks the slower island's last boundary (run 9a64a0588dc3).
                    count = (int(snap.get("worker_count", 0))
                             + int(snap.get("finished_count", 0)))
            except Exception as e:  # noqa: BLE001 - the server may still be coming up
                logger.debug("cohort probe failed: %s", e)
            if count >= self._min_replicas:
                if not startup or last_seen >= 0:
                    logger.info(
                        "cohort ready: %d/%d workers registered; %s",
                        count, self._min_replicas,
                        "starting training" if startup else "resuming training",
                    )
                return
            if count != last_seen and count >= 0:
                # Mirrors the lighthouse's "New quorum not ready, only have N
                # participants" line, so the wait is visible in the run's logs.
                logger.info(
                    "%s cohort: only have %d worker(s), need min_replicas %d",
                    "waiting for" if startup else "training paused, waiting for",
                    count, self._min_replicas,
                )
                last_seen = count
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"waited {self._min_replicas_timeout:.0f}s for {self._min_replicas} "
                    f"workers to register with the parameter server at "
                    f"{self._status_url}; only {max(count, 0)} present. Training alone "
                    f"would silently defeat the point of a multi-island run -- raise "
                    f"min_replicas_timeout if provisioning is just slow, or set "
                    f"min_replicas=0 to allow a solo run."
                )
            time.sleep(min(2.0, self._heartbeat_interval))

    def __enter__(self) -> "AsyncDiLoCo":
        # Start heartbeats before the initial pull: on a large model the pull
        # is the worker's longest silent phase and it should be visible to
        # the server for all of it.
        if self._heartbeat_url is not None:
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._run_heartbeat, daemon=True
            )
            self._heartbeat_thread.start()
        try:
            # Order matters: our own heartbeat must be running or we would never
            # count ourselves and every worker would wait for the others forever.
            self._await_cohort()
            self._initial_pull()
        except Exception:
            self._stop_heartbeat()
            raise
        self._hooks.append(
            self._inner_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Announce completion BEFORE the heartbeat stops: a peer still training reads
        # the cohort from /status, and a worker that simply goes quiet is
        # indistinguishable from one that crashed. Only on a clean exit -- an
        # exception here is a real loss and must keep blocking the cohort.
        if exc_type is None:
            self._announce_done()
        self._stop_heartbeat()
        # Fragment mode: drain (never adopt) an in-flight exchange — the
        # process is shutting down and replica followers couldn't join the
        # adopt broadcast anyway; losing the final fragment window is the
        # same cost as any dropped push.
        inflight, self._inflight = self._inflight, None
        if inflight is not None and inflight.thread is not None:
            # Capped independently of _sync_timeout: that bounds a stall
            # mid-transfer (minutes), this is a discard on the way out.
            inflight.thread.join(timeout=min(self._sync_timeout, 60.0))
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return False

    def _announce_done(self) -> None:
        """Tell the parameter server this worker finished its planned steps.

        Best-effort: a failure here costs a peer a cohort-timeout at worst, never
        correctness, so it must not raise on the way out of a successful run.
        """
        if self._status_url is None or not self._is_lead:
            return
        url = self._status_url.replace("/status", "/done")
        sep = "&" if "?" in url else "?"
        try:
            with urllib.request.urlopen(
                f"{url}{sep}worker_id={self._worker_id}", timeout=10.0
            ):
                pass
        except Exception as e:  # noqa: BLE001 - never fail a completed run on this
            logger.debug("done announcement failed: %s", e)

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval * 2)
            self._heartbeat_thread = None

    def _run_heartbeat(self) -> None:
        """
        Daemon thread: ping the server's /heartbeat endpoint every
        ``heartbeat_interval`` seconds while this worker is in context.

        Mirrors the ``_run_heartbeat`` loop in ``manager.rs`` (Lighthouse).
        Failures are logged at DEBUG and retried on the next tick — transient
        network hiccups should not kill the training loop.
        """
        assert self._heartbeat_stop is not None
        assert self._heartbeat_url is not None
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                with urllib.request.urlopen(
                    self._heartbeat_url, timeout=5.0
                ) as resp:
                    resp.read()
            except Exception as exc:
                logger.debug("Heartbeat failed (will retry): %s", exc)

    def _step_post_hook(
        self,
        _optim: optim.Optimizer,
        _args: Tuple[Any, ...],
        _kwargs: Dict[str, Any],
    ) -> None:
        self._local_step += 1
        # Fragment mode shortens the boundary cadence: one fragment syncs per
        # sync_every/P window, so a full rotation still moves the whole model
        # every sync_every steps (P=1: the legacy whole-model boundary).
        if self._local_step < self._sync_every // self._num_fragments:
            return

        # Lighthouse parity: min_replicas gates EVERY quorum there, so re-check the
        # cohort at every sync boundary here. Pushing into a cohort that has shrunk
        # below the floor is the silent-solo-training failure this exists to stop.
        self._await_cohort(startup=False)

        if self._replica_pg is not None:
            self._boundary_replica()
        elif self._pending_resync:
            # Server was unreachable on a previous boundary: the dropped
            # push's window is gone, so just try to re-baseline (pull-only)
            # with backoff and keep training locally in the meantime.
            if time.monotonic() >= self._resync_at:
                self._try_resync()
        elif self._num_fragments > 1:
            self._boundary_fragment()
        else:
            try:
                self._sync()
            except Exception as exc:
                # A transient server/network failure must not kill the
                # training loop (and must not leave a half-applied push: the
                # server may have committed our step before the return
                # transfer failed, so the push is dropped and we re-baseline
                # via a pull-only resync instead of retrying it).
                logger.warning(
                    "AsyncDiLoCo sync failed; dropping push and continuing "
                    "local training (will resync): %s",
                    exc,
                )
                self._pending_resync = True
                self._resync_backoff = 1.0
                self._resync_at = time.monotonic()

        self._local_step = 0
        self._window_start = time.monotonic()

    def _try_resync(self) -> None:
        """Attempt a pull-only re-baseline after a failed sync."""
        try:
            self._pull_global()
        except Exception as exc:
            self._resync_at = time.monotonic() + self._resync_backoff
            self._resync_backoff = min(
                self._resync_backoff * 2, self._resync_backoff_max
            )
            logger.warning(
                "AsyncDiLoCo resync failed (next attempt in %.0fs): %s",
                self._resync_at - time.monotonic(),
                exc,
            )
            return
        self._pending_resync = False
        self._resync_backoff = 1.0
        self._skip_speed_report = True
        logger.info(
            "AsyncDiLoCo resynced to server revision %d", self._baseline_revision
        )

    # ------------------------------------------------------------------ #
    # Fragment-wise sync (num_fragments > 1): staggered rotation with     #
    # overlapped communication (Decoupled DiLoCo, arXiv 2604.21428)       #
    # ------------------------------------------------------------------ #

    def _enter_resync(self) -> None:
        """Schedule a pull-only whole-model re-baseline (fragment paths: a
        failed or rejected fragment exchange invalidates the pipeline, and
        the pull refreshes every fragment's baseline at once)."""
        self._pending_resync = True
        self._resync_backoff = 1.0
        self._resync_at = time.monotonic()

    def _window_speed(self) -> float:
        """Inner steps/sec over the window just ended (0.0 right after a
        resync — that window started from stale params, exclude it from
        DyLU measurement)."""
        if self._skip_speed_report:
            self._skip_speed_report = False
            return 0.0
        elapsed = time.monotonic() - self._window_start
        return self._local_step / elapsed if elapsed > 0 else 0.0

    def _apply_dylu(self, new_steps: int) -> None:
        """Adopt a DyLU window-length recommendation. ``new_steps`` always
        means the FULL cycle; fragment mode rounds it to a multiple of P so
        the boundary cadence stays integral."""
        if new_steps > 0 and self._num_fragments > 1:
            new_steps = max(
                self._num_fragments,
                new_steps // self._num_fragments * self._num_fragments,
            )
        if new_steps > 0 and new_steps != self._sync_every:
            logger.info(
                f"AsyncDiLoCo DyLU: sync_every updated "
                f"{self._sync_every} → {new_steps}"
            )
            self._sync_every = new_steps

    def _fragment_pseudo_grad(self, fragment: int) -> torch.Tensor:
        """Δ = θ_baseline − θ_local over ONE fragment's parameters, flat fp32
        CPU (the fragment-scoped form of :meth:`_sync`'s whole-model loop)."""
        a, b = self._frag_bounds[fragment]
        grad_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for name in self._param_names[a:b]:
                local_cpu = self._params_by_name[name].detach().cpu()
                grad_chunks.append(
                    (self._global_params[name] - local_cpu).reshape(-1).float()
                )
        return torch.cat(grad_chunks)

    def _launch_push(
        self, fragment: int, speed: float, flat_grads: torch.Tensor
    ) -> None:
        """Start one fragment exchange in the background. The 503 busy-retry
        loop lives inside _session_roundtrip and works unchanged there."""
        inflight = _InflightPush(fragment=fragment)

        def _run() -> None:
            try:
                inflight.result = self._session_roundtrip(
                    flag=1.0, speed=speed, flat_grads=flat_grads,
                    fragment=fragment,
                )
            except BaseException as exc:  # surfaced at join, never raised here
                inflight.error = exc

        inflight.thread = threading.Thread(
            target=_run, daemon=True, name=f"asyncdiloco-push-f{fragment}"
        )
        self._inflight = inflight
        inflight.thread.start()

    def _join_and_adopt_inflight(self) -> bool:
        """Join the outstanding fragment exchange (backpressure: blocks only
        if the roundtrip was slower than one fragment-window of training) and
        adopt its merged fragment. Returns False when this boundary must not
        launch a new push (failure or rejection → whole-model resync)."""
        inflight, self._inflight = self._inflight, None
        if inflight is None:
            return True  # first boundary of the pipeline: nothing in flight
        inflight.thread.join()
        if inflight.error is not None:
            logger.warning(
                "AsyncDiLoCo fragment sync failed; dropping push and "
                "continuing local training (will resync): %s",
                inflight.error,
            )
            self._enter_resync()
            return False
        flat_params, new_steps, revision, applied = inflight.result
        if not applied:
            # Stale baseline (server checkpoint restore): EVERY fragment's
            # baseline is stale, so skip the fragment-sized response and
            # re-baseline the whole model via the resync path.
            logger.warning(
                "AsyncDiLoCo fragment push rejected by server (baseline "
                "revision %d); scheduling whole-model resync",
                self._baseline_revision,
            )
            self._skip_speed_report = True
            self._enter_resync()
            return False
        self._adopt_fragment(inflight.fragment, flat_params, revision, new_steps)
        return True

    def _adopt_fragment(
        self,
        fragment: int,
        flat_params: torch.Tensor,
        revision: int,
        new_steps: int,
    ) -> None:
        """Install one merged fragment into the model and the baseline backup.

        With ``fragment_update_alpha`` > 0 the merge lerps toward the
        parameter's CURRENT local value — the steps trained while the
        exchange was in flight — which is Streaming DiLoCo's merge semantics
        (at P=1 push-time and adopt-time locals coincide, so this matches the
        legacy blend exactly).
        """
        a, b = self._frag_bounds[fragment]
        alpha = self._fragment_update_alpha
        with torch.no_grad():
            offset = 0
            for name in self._param_names[a:b]:
                p = self._params_by_name[name]
                n = p.numel()
                chunk = flat_params[offset : offset + n].view(p.shape)
                offset += n
                self._global_params[name].copy_(chunk)
                if alpha > 0.0:
                    local_prev = p.data.clone()
                p.data.copy_(chunk.to(p.device))
                if alpha > 0.0:
                    p.data.lerp_(local_prev, alpha)
        self._baseline_revision = revision
        if self._reset_inner_state and fragment == self._num_fragments - 1:
            # Opt-in reset keeps its once-per-full-cycle cadence.
            self._inner_optimizer.state.clear()
        self._apply_dylu(new_steps)

    def _boundary_fragment(self) -> None:
        """One fragment boundary (single-process): adopt the previous
        fragment's merged result, then launch this fragment's exchange."""
        if not self._join_and_adopt_inflight():
            return
        fragment = self._frag_idx
        self._frag_idx = (fragment + 1) % self._num_fragments
        speed = self._window_speed()
        self._launch_push(fragment, speed, self._fragment_pseudo_grad(fragment))

    # ------------------------------------------------------------------ #
    # Replica mode (replica_pg): one PS session per multi-rank replica    #
    # ------------------------------------------------------------------ #

    # Broadcast words (lead -> followers). Action picks the boundary branch;
    # outcome picks the post-HTTP branch. _OUT_NONE: fragment mode's first
    # boundary — nothing in flight yet, nothing to adopt, proceed to launch.
    _ACT_SKIP, _ACT_SYNC, _ACT_RESYNC = 0, 1, 2
    _OUT_FAIL, _OUT_ADOPT, _OUT_ADOPT_BLEND, _OUT_NONE = 0, 1, 2, 3

    def _bcast_words(self, words: List[int]) -> List[int]:
        """Broadcast small control integers from the lead. Every branch the
        replica takes around an HTTP attempt keys off these words — per-rank
        decisions (clocks, exceptions the followers never saw) would pick
        different branches and deadlock the next collective."""
        t = torch.tensor(
            words if self._is_lead else [0] * len(words), dtype=torch.int64
        )
        dist.broadcast(t, src=self._lead_rank, group=self._replica_pg)
        return t.tolist()

    def _boundary_replica(self) -> None:
        """One window boundary in replica mode: the lead decides the action
        (its resync clock is the replica's), everyone follows the broadcast."""
        act = self._ACT_SKIP
        if self._is_lead:
            if self._pending_resync:
                act = (
                    self._ACT_RESYNC
                    if time.monotonic() >= self._resync_at
                    else self._ACT_SKIP
                )
            else:
                act = self._ACT_SYNC
        (act,) = self._bcast_words([act])
        if act == self._ACT_SYNC:
            if self._num_fragments > 1:
                self._boundary_fragment_replica()
            else:
                self._sync_replica()
        elif act == self._ACT_RESYNC:
            self._resync_replica()

    @torch.profiler.record_function("async_diloco.sync_replica")
    def _sync_replica(self) -> None:
        """Replica-mode :meth:`_sync`: gather full values -> lead HTTP ->
        outcome broadcast -> adopt. Failure semantics mirror the
        single-process path (drop the push, pull-only resync later)."""
        need_local = self._fragment_update_alpha > 0.0
        blend_local: Dict[str, torch.Tensor] = {}
        grad_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                if need_local:
                    local = p.to_local() if isinstance(p, DTensor) else p
                    # Per-rank snapshot of the rank's OWN slice: the blend
                    # lerps each rank's post-adopt shard toward it.
                    blend_local[name] = local.detach().cpu().clone()
                # Collective for sharded params — every rank participates,
                # only the lead consumes the value (one param at a time, so
                # the GPU transient is the largest parameter, not the model).
                full = _full_value(p)
                if self._is_lead:
                    grad_chunks.append(
                        (self._global_params[name] - full.detach().cpu())
                        .reshape(-1)
                        .float()
                    )
                del full

        outcome, new_steps, revision = self._OUT_FAIL, 0, 0
        flat_params: Optional[torch.Tensor] = None
        if self._is_lead:
            logger.info(
                f"AsyncDiLoCo syncing after {self._sync_every} inner steps"
            )
            if self._skip_speed_report:
                speed = 0.0
                self._skip_speed_report = False
            else:
                elapsed = time.monotonic() - self._window_start
                speed = self._local_step / elapsed if elapsed > 0 else 0.0
            try:
                flat_params, new_steps, revision, applied = (
                    self._session_roundtrip(
                        flag=1.0,
                        speed=speed,
                        flat_grads=torch.cat(grad_chunks),
                    )
                )
                if applied:
                    outcome = (
                        self._OUT_ADOPT_BLEND if need_local else self._OUT_ADOPT
                    )
                else:
                    # Stale baseline (e.g. server checkpoint restore): the
                    # response is a pure re-baseline, never blended.
                    logger.warning(
                        "AsyncDiLoCo push rejected by server (baseline "
                        "revision %d); re-baselining to server revision %d",
                        self._baseline_revision,
                        revision,
                    )
                    self._skip_speed_report = True
                    outcome = self._OUT_ADOPT
            except Exception as exc:
                logger.warning(
                    "AsyncDiLoCo sync failed; dropping push and continuing "
                    "local training (will resync): %s",
                    exc,
                )
                self._pending_resync = True
                self._resync_backoff = 1.0
                self._resync_at = time.monotonic()
        outcome, new_steps = self._bcast_words([outcome, new_steps])
        if outcome != self._OUT_FAIL:
            self._adopt_replica(
                flat_params,
                revision,
                new_steps,
                blend_local if outcome == self._OUT_ADOPT_BLEND else None,
            )

    def _boundary_fragment_replica(self) -> None:
        """One fragment boundary in replica mode. Phase 1 joins and adopts
        the previous fragment's exchange (every branch keys off broadcast
        words — per-rank decisions would diverge and deadlock the next
        collective). Phase 2 gathers this fragment's pseudo-gradient with
        collectives on the MAIN thread, then only the lead's HTTP roundtrip
        runs in the background."""
        out, new_steps, frag_prev = self._OUT_NONE, 0, 0
        revision = 0
        flat_params: Optional[torch.Tensor] = None
        if self._is_lead:
            inflight, self._inflight = self._inflight, None
            if inflight is not None:
                inflight.thread.join()
                frag_prev = inflight.fragment
                if inflight.error is not None:
                    logger.warning(
                        "AsyncDiLoCo fragment sync failed; dropping push and "
                        "continuing local training (will resync): %s",
                        inflight.error,
                    )
                    self._enter_resync()
                    out = self._OUT_FAIL
                else:
                    flat_params, new_steps, revision, applied = inflight.result
                    if applied:
                        out = self._OUT_ADOPT
                    else:
                        logger.warning(
                            "AsyncDiLoCo fragment push rejected by server "
                            "(baseline revision %d); scheduling whole-model "
                            "resync",
                            self._baseline_revision,
                        )
                        self._skip_speed_report = True
                        self._enter_resync()
                        out = self._OUT_FAIL
        out, new_steps, frag_prev = self._bcast_words(
            [out, new_steps, frag_prev]
        )
        if out == self._OUT_ADOPT:
            self._adopt_replica_fragment(
                frag_prev, flat_params, revision, new_steps
            )
        elif out == self._OUT_FAIL:
            return  # the resync path takes over at the next boundary

        fragment = self._frag_idx
        self._frag_idx = (fragment + 1) % self._num_fragments
        a, b = self._frag_bounds[fragment]
        grad_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for name in self._param_names[a:b]:
                # Collective for sharded params — every rank participates in
                # the same order, only the lead consumes the value.
                full = _full_value(self._params_by_name[name])
                if self._is_lead:
                    grad_chunks.append(
                        (self._global_params[name] - full.detach().cpu())
                        .reshape(-1)
                        .float()
                    )
                del full
        if self._is_lead:
            self._launch_push(
                fragment, self._window_speed(), torch.cat(grad_chunks)
            )

    def _adopt_replica_fragment(
        self,
        fragment: int,
        flat_params: Optional[torch.Tensor],
        revision: int,
        new_steps: int,
    ) -> None:
        """Fragment-scoped :meth:`_adopt_replica`: broadcast the merged
        fragment parameter-by-parameter; each rank installs its own slice.
        Blend (alpha > 0) lerps toward the rank's CURRENT local shard — the
        in-flight training progress (see :meth:`_adopt_fragment`)."""
        alpha = self._fragment_update_alpha
        a, b = self._frag_bounds[fragment]
        with torch.no_grad():
            offset = 0
            for name in self._param_names[a:b]:
                p = self._params_by_name[name]
                numel = p.numel()
                shape = tuple(p.shape)
                if self._is_lead:
                    full = flat_params[offset : offset + numel].view(shape)
                else:
                    full = torch.empty(shape, dtype=torch.float32)
                offset += numel
                dist.broadcast(
                    full, src=self._lead_rank, group=self._replica_pg
                )
                if isinstance(p, DTensor):
                    local = p.to_local()
                    chunk = (
                        full
                        if tuple(local.shape) == shape
                        else full[_local_shard_slices(p)]
                    )
                else:
                    local, chunk = p.data, full
                if alpha > 0.0:
                    prev = local.clone()
                local.copy_(chunk)
                if alpha > 0.0:
                    local.lerp_(prev, alpha)
                if self._is_lead:
                    self._global_params[name].copy_(full)
                del full
        if self._is_lead:
            self._baseline_revision = revision
        if self._reset_inner_state and fragment == self._num_fragments - 1:
            self._inner_optimizer.state.clear()
        self._apply_dylu(new_steps)

    def _resync_replica(self) -> None:
        """Replica-mode :meth:`_try_resync`: pull-only re-baseline; backoff
        state lives on the lead alone."""
        outcome, new_steps, revision = self._OUT_FAIL, 0, 0
        flat_params: Optional[torch.Tensor] = None
        if self._is_lead:
            try:
                flat_params, new_steps, revision, _ = self._session_roundtrip(
                    flag=0.0, speed=0.0, flat_grads=None
                )
                outcome = self._OUT_ADOPT
            except Exception as exc:
                self._resync_at = time.monotonic() + self._resync_backoff
                self._resync_backoff = min(
                    self._resync_backoff * 2, self._resync_backoff_max
                )
                logger.warning(
                    "AsyncDiLoCo resync failed (next attempt in %.0fs): %s",
                    self._resync_at - time.monotonic(),
                    exc,
                )
        outcome, new_steps = self._bcast_words([outcome, new_steps])
        if outcome == self._OUT_FAIL:
            return
        self._adopt_replica(flat_params, revision, new_steps, None)
        if self._is_lead:
            self._pending_resync = False
            self._resync_backoff = 1.0
            self._skip_speed_report = True
            logger.info("AsyncDiLoCo resynced to server revision %d", revision)

    def _adopt_replica(
        self,
        flat_params: Optional[torch.Tensor],
        revision: int,
        new_steps: int,
        blend_local: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Replica-mode :meth:`_adopt_global`: broadcast the pulled fp32
        params parameter-by-parameter (bounded follower memory — one full
        parameter at a time, never the flat model) and install each rank's
        own slice in place, sliced by the DTensor placement offsets."""
        with torch.no_grad():
            offset = 0
            for (name, p), numel in zip(
                self._model.named_parameters(), self._param_numels
            ):
                shape = tuple(p.shape)
                if self._is_lead:
                    full = flat_params[offset : offset + numel].view(shape)
                else:
                    full = torch.empty(shape, dtype=torch.float32)
                offset += numel
                dist.broadcast(
                    full, src=self._lead_rank, group=self._replica_pg
                )
                if isinstance(p, DTensor):
                    # to_local() shares storage: writes land in the model.
                    local = p.to_local()
                    chunk = (
                        full
                        if tuple(local.shape) == shape
                        else full[_local_shard_slices(p)]
                    )
                else:
                    local, chunk = p.data, full
                local.copy_(chunk)  # device move + dtype cast in one hop
                if blend_local is not None:
                    local.lerp_(
                        blend_local[name].to(local.device, local.dtype),
                        self._fragment_update_alpha,
                    )
                if self._is_lead:
                    self._global_params[name].copy_(full)
                del full
        if self._is_lead:
            self._baseline_revision = revision
        # A whole adopted model is what delta downloads reconstruct
        # against; fragment adopts never flip this.
        self._have_baseline = True
        if self._reset_inner_state:
            self._inner_optimizer.state.clear()
        # DyLU: new_steps arrives via the outcome broadcast, so every rank
        # moves to the same window length on the same boundary.
        self._apply_dylu(new_steps)

    def _pull_global_replica(self) -> None:
        """Replica-mode pull: failure raises on EVERY rank (so __enter__
        propagates consistently instead of stranding followers in a
        broadcast that will never come)."""
        outcome, new_steps, revision = self._OUT_FAIL, 0, 0
        flat_params: Optional[torch.Tensor] = None
        err: Optional[Exception] = None
        if self._is_lead:
            try:
                flat_params, new_steps, revision, _ = self._session_roundtrip(
                    flag=0.0, speed=0.0, flat_grads=None
                )
                outcome = self._OUT_ADOPT
            except Exception as exc:
                err = exc
        outcome, new_steps = self._bcast_words([outcome, new_steps])
        if outcome == self._OUT_FAIL:
            raise RuntimeError(
                "AsyncDiLoCo pull failed on the replica lead"
            ) from err
        self._adopt_replica(flat_params, revision, new_steps, None)

    # ------------------------------------------------------------------ #
    # Session plumbing                                                    #
    # ------------------------------------------------------------------ #

    def _session_roundtrip(
        self,
        flag: float,
        speed: float,
        flat_grads: Optional[torch.Tensor],
        fragment: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, bool]:
        """
        One push/pull cycle: a single HTTP POST to the server's /sync
        endpoint (see :class:`AsyncDiLoCoServer` for the wire format).
        ``fragment`` scopes the push (and the returned params) to one
        fragment's slice; pull-only requests are always whole-model.

        Fragment mode runs this on a background thread — it only reads
        construction-time state plus ``_baseline_revision`` (an int read,
        benign against the main thread's adopt), and exactly one exchange is
        in flight at a time.

        Returns ``(flat_params, new_steps, revision, applied)``.
        """
        header: Dict[str, Any] = {
            "flag": int(flag),
            "speed": speed,
            "baseline_revision": self._baseline_revision,
        }
        if self._wire_bf16:
            # Download negotiation: an old server ignores this key and replies
            # fp32 with no `dtype` field; a new one replies bf16 AND names it.
            header["accept_dtype"] = "bfloat16"
            # Delta download: only when this worker holds a whole adopted
            # baseline the server can still have (fragment workers mutate the
            # baseline slice-wise and never qualify), and not on a refresh
            # round. The server falls back to full on its own whenever the
            # named revision has left its ring, so this is an offer, not a
            # demand.
            if (
                self._have_baseline
                and fragment is None
                and getattr(self, "_num_fragments", 1) == 1
                and self._deltas_since_full < self._delta_refresh_every
            ):
                header["accept_delta"] = 1
        expected_numel = (
            self._total_numel
            if fragment is None
            else self._frag_numels[fragment]
        )
        body = b""
        if flat_grads is not None:
            header["numel"] = flat_grads.numel()
            numels = self._param_numels
            if fragment is not None:
                header["fragment"] = fragment
                header["num_fragments"] = self._num_fragments
                a, b = self._frag_bounds[fragment]
                numels = self._param_numels[a:b]
            if self._quantize:
                q, scales = _quantize_int8(flat_grads, numels)
                header["dtype"] = "int8"
                body = _tensor_to_bytes(scales) + _tensor_to_bytes(q)
            elif self._server_bf16:
                # The server named a dtype in an earlier response (the initial
                # pull, at the latest), so it understands bf16 uploads. Never
                # sent blind: a pre-bf16 server would 500 the push and the
                # catch-all in _step_post_hook would silently drop the window.
                header["dtype"] = "bfloat16"
                body = _bf16_bytes(flat_grads)
            else:
                header["dtype"] = "float32"
                body = _tensor_to_bytes(flat_grads)

        payload = (json.dumps(header) + "\n").encode() + body

        # 503 means "all session slots busy, come back" (the server's
        # max_sessions semaphore), NOT a failure: WAIT AND RETRY THE SAME PUSH.
        # Letting it escape would reach _step_post_hook's catch-all, which drops
        # the push and re-baselines — throwing away the whole window's
        # pseudo-gradient. That turns a capped-concurrency server (the memory fix
        # for a central PS) into silent training loss, so the cap is only safe
        # with this retry. Every other error still propagates untouched.
        for attempt in range(self._busy_retries + 1):
            request = urllib.request.Request(
                self._server_address,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self._sync_timeout
                ) as resp:
                    resp_header = json.loads(resp.readline(_MAX_HEADER_BYTES))
                    numel = int(resp_header["numel"])
                    if numel != expected_numel:
                        raise ValueError(
                            f"global param numel mismatch: got {numel}, "
                            f"expected {expected_numel} — model/server mismatch?"
                        )
                    # A response that names its dtype is from a server that
                    # also takes bf16 uploads; remember for the next push.
                    self._server_bf16 = (
                        self._wire_bf16 and "dtype" in resp_header
                    )
                    if resp_header.get("dtype") == "delta_int8":
                        if int(resp_header["delta_from"]) != self._baseline_revision:
                            raise ValueError(
                                "delta against revision "
                                f"{resp_header['delta_from']} but this worker "
                                f"holds {self._baseline_revision}"
                            )
                        scales = _bytes_to_tensor(
                            _read_exact(resp, len(self._param_numels) * 4),
                            torch.float32,
                        )
                        q = _bytes_to_tensor(
                            _read_exact(resp, numel), torch.int8
                        )
                        flat_params = self._baseline_flat() + _dequantize_int8(
                            q, scales, self._param_numels
                        )
                        self._deltas_since_full += 1
                    elif resp_header.get("dtype") == "bfloat16":
                        flat_params = _bytes_to_tensor(
                            _read_exact(resp, numel * 2), torch.bfloat16
                        ).to(torch.float32)
                    else:
                        flat_params = _bytes_to_tensor(
                            _read_exact(resp, numel * 4), torch.float32
                        )
                    if resp_header.get("dtype") != "delta_int8":
                        self._deltas_since_full = 0
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 503 or attempt == self._busy_retries:
                    raise
                # Honor Retry-After when the server sends it (it does: "1"),
                # else fall back to the same delay.
                try:
                    delay = float(exc.headers.get("Retry-After", 1.0))
                except (TypeError, ValueError):
                    delay = 1.0
                logger.debug(
                    "Server busy (503); retrying push in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self._busy_retries,
                )
                time.sleep(min(delay, self._sync_timeout))

        return (
            flat_params,
            int(resp_header["new_steps"]),
            int(resp_header["revision"]),
            bool(resp_header["applied"]),
        )

    def _baseline_flat(self) -> torch.Tensor:
        """This worker's adopted baseline as one flat fp32 tensor, in the
        canonical parameter order (the layout every wire payload uses)."""
        return torch.cat(
            [
                self._global_params[name].reshape(-1).float()
                for name in self._param_names
            ]
        )

    def _adopt_global(
        self,
        flat_params: torch.Tensor,
        revision: int,
        new_steps: int,
        blend_local: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Install newly pulled global params into the model and backup."""
        with torch.no_grad():
            offset = 0
            for name, p in self._model.named_parameters():
                n = p.numel()
                chunk = flat_params[offset : offset + n].view(p.shape)
                offset += n
                self._global_params[name].copy_(chunk)
                p.data.copy_(chunk.to(p.device))
                if blend_local is not None:
                    p.data.lerp_(
                        blend_local[name].to(p.device),
                        self._fragment_update_alpha,
                    )
        self._baseline_revision = revision
        # A whole adopted model is what delta downloads reconstruct
        # against; fragment adopts never flip this.
        self._have_baseline = True

        if self._reset_inner_state:
            # Optional deviation from DiLoCo (which persists inner state
            # across windows); see the constructor docstring.
            self._inner_optimizer.state.clear()

        self._apply_dylu(new_steps)

    def _pull_global(self) -> None:
        """Pull current global params (flag=0) and adopt them wholesale."""
        if self._replica_pg is not None:
            self._pull_global_replica()
            return
        flat_params, new_steps, revision, _ = self._session_roundtrip(
            flag=0.0, speed=0.0, flat_grads=None
        )
        self._adopt_global(flat_params, revision, new_steps)

    @torch.profiler.record_function("async_diloco.initial_pull")
    def _initial_pull(self) -> None:
        """Pull current global params from server without sending any pseudo-gradient.

        Called at __enter__ so the local model and _global_params are aligned
        with the server's authoritative weights before the first inner window starts.
        Also receives the server's DyLU H value as the initial sync_every hint.
        """
        self._pull_global()
        self._window_start = time.monotonic()

    @torch.profiler.record_function("async_diloco.sync")
    def _sync(self) -> None:
        """Push pseudo-gradients to server and pull new global params.

        Note: the outer step is committed on the server before the response is
        delivered. If the return transfer fails, the caller
        (:meth:`_step_post_hook`) drops the push and re-baselines via a
        pull-only resync — it never retries the push, so a committed-but-
        unacknowledged step can't be applied twice.
        """
        logger.info(f"AsyncDiLoCo syncing after {self._sync_every} inner steps")

        if self._skip_speed_report:
            speed = 0.0
            self._skip_speed_report = False
        else:
            elapsed = time.monotonic() - self._window_start
            speed = self._local_step / elapsed if elapsed > 0 else 0.0

        # Snapshot local params for alpha blend (only needed when alpha > 0)
        need_local = self._fragment_update_alpha > 0.0
        local_params: Dict[str, torch.Tensor] = {}
        grad_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            # self._param_names (fixed insertion-order list) guarantees the
            # flat layout matches the server's named_parameters() order.
            for name, p in self._model.named_parameters():
                local_cpu = p.detach().cpu()
                if need_local:
                    local_params[name] = local_cpu
                grad_chunks.append(
                    (self._global_params[name] - local_cpu).reshape(-1).float()
                )
        flat_grads = torch.cat(grad_chunks)

        flat_params, new_steps, revision, applied = self._session_roundtrip(
            flag=1.0, speed=speed, flat_grads=flat_grads
        )

        if not applied:
            # Server rejected the push (stale baseline, e.g. after a server
            # checkpoint restore): treat the response as a pure resync.
            logger.warning(
                "AsyncDiLoCo push rejected by server (baseline revision %d); "
                "re-baselining to server revision %d",
                self._baseline_revision,
                revision,
            )
            self._adopt_global(flat_params, revision, new_steps)
            self._skip_speed_report = True
            return

        self._adopt_global(
            flat_params,
            revision,
            new_steps,
            blend_local=local_params if need_local else None,
        )
