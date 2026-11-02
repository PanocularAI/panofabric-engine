# Copyright (c) Panocular AI.
#
# Checkpoint distribution for the async-inference swarm: in three pieces 
# that share one module because they share one wire format:
#
#   - Sharding: split a *serialized checkpoint* into size-balanced byte
#     pieces with a SHA-256-per-shard manifest, so a relay tier can
#     store/stream them independently and a fetcher can verify each piece
#     before reassembling. Unrelated to model/optimizer sharding (FSDP/TP) --
#     operates on a plain CPU state dict, after it's already been gathered
#     full (e.g. into a BackgroundPublisher's host buffers).
#   - RelayServer: a CDN-like CPU relay node sitting between one trainer and
#     many inference workers, so the trainer never serves every worker's
#     weight pulls itself and workers never need the trainer's address
#     directly. Stores shards in files -- /dev/shm when RAM has room, else
#     disk -- with manifests in memory, and keeps only the last
#     ``retain_last`` versions (matching the paper).
#   - RelayClient: the publisher/fetcher counterpart. Tracks a per-relay
#     success_rate/bandwidth EMA and picks a relay *probabilistically
#     weighted by success_rate x bandwidth* rather than always the fastest --
#     the paper's exact rule, which keeps one flaky-but-occasionally-fast
#     relay from permanently starving the others.
#   - BackgroundPublisher: the trainer-side upload, run on a thread so
#     training pauses only for the device->host copy.
#   - Deltas: a fetcher that already holds version A asks for B as a lossless
#     delta (``GET /delta/{A}/{B}``): each tensor's bit pattern XOR A's,
#     zstd-compressed. RL updates are tiny next to bf16's resolution, so most
#     weights keep their exact bits between versions (Qwen3-0.6B's 1.5 GB
#     checkpoint ships as ~11 MB). Verified per chunk (SHA-256), so a worker
#     holds exactly the weights it would have downloaded whole; any doubt
#     falls back to the full download.
#
# Needs torch (state-dict tensors) but never the torchtitan training stack or
# vLLM, so the standalone relay-server process stays CPU-only, like heloco's
# parameter server (parameter_server.py). Run one relay node per box with:
#   python -m panoengine.decentralized.relay --port 8765

import argparse
import asyncio
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import socket
import struct
import tempfile
import threading
import time
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import torch
import zstandard
from aiohttp import web

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.3
_RELAY_KEY = web.AppKey("relay")
_UPLOAD_CHUNK = 4 << 20
# Site cache (RelayClient.site_cache_dir): how often a fetcher re-checks shards
# another fetcher is downloading, how often a downloader refreshes its claim, and
# how long an unrefreshed claim is trusted before another fetcher takes it over.
_SITE_POLL_S = 2.0
_SITE_HEARTBEAT_S = 10.0
_SITE_CLAIM_STALE_S = 120.0


# --------------------------------------------------------------------- #
# Sharding: checkpoint <-> verified byte shards.
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Describes one published checkpoint version: enough for a fetcher to
    know how many shards to ask for and verify each one it receives."""

    version: int
    num_shards: int
    shard_checksums: list[str] = field(default_factory=list)
    shard_sizes: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "num_shards": self.num_shards,
            "shard_checksums": self.shard_checksums,
            "shard_sizes": self.shard_sizes,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CheckpointManifest":
        return cls(
            version=data["version"],
            num_shards=data["num_shards"],
            shard_checksums=list(data["shard_checksums"]),
            shard_sizes=list(data["shard_sizes"]),
        )


def _partition_names(
    state_dict: dict[str, torch.Tensor], num_shards: int
) -> list[list[str]]:
    """Greedy size-balanced bin-packing of parameter names into num_shards
    groups: sort by tensor nbytes descending, always add to the currently
    lightest bin. Deterministic given a deterministic dict iteration order."""
    names = sorted(
        state_dict,
        key=lambda n: state_dict[n].numel() * state_dict[n].element_size(),
        reverse=True,
    )
    bins: list[list[str]] = [[] for _ in range(num_shards)]
    bin_bytes = [0] * num_shards
    for name in names:
        idx = min(range(num_shards), key=lambda i: bin_bytes[i])
        bins[idx].append(name)
        bin_bytes[idx] += state_dict[name].numel() * state_dict[name].element_size()
    return bins


def shard_state_dict(
    state_dict: dict[str, torch.Tensor], num_shards: int
) -> list[bytes]:
    """Split a full CPU state dict into ``num_shards`` size-balanced shards,
    each a torch.save blob of {name: tensor} for its assigned parameter
    names. A shard may be empty (fewer params than shards) -- serialized as
    an empty dict, valid and reassembles to nothing."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    name_groups = _partition_names(state_dict, num_shards)
    shards: list[bytes] = []
    for names in name_groups:
        buf = io.BytesIO()
        torch.save({name: state_dict[name] for name in names}, buf)
        shards.append(buf.getvalue())
    return shards


def build_manifest(version: int, shard_bytes: list[bytes]) -> CheckpointManifest:
    """Compute the SHA-256 checksum + size of each shard for a manifest a
    fetcher can verify against before trusting the reassembled state dict."""
    checksums = [hashlib.sha256(b).hexdigest() for b in shard_bytes]
    sizes = [len(b) for b in shard_bytes]
    return CheckpointManifest(
        version=version,
        num_shards=len(shard_bytes),
        shard_checksums=checksums,
        shard_sizes=sizes,
    )


class ShardIntegrityError(RuntimeError):
    """A fetched shard's checksum didn't match the manifest -- corrupted in
    transit or from a misbehaving/stale relay. Caller should retry a
    different relay rather than trust the payload."""


def verify_shard(shard_idx: int, data: bytes, manifest: CheckpointManifest) -> None:
    if shard_idx < 0 or shard_idx >= manifest.num_shards:
        raise ShardIntegrityError(
            f"shard index {shard_idx} out of range for manifest with "
            f"{manifest.num_shards} shards"
        )
    checksum = hashlib.sha256(data).hexdigest()
    expected = manifest.shard_checksums[shard_idx]
    if checksum != expected:
        raise ShardIntegrityError(
            f"shard {shard_idx} checksum mismatch (version={manifest.version}): "
            f"got {checksum}, expected {expected}"
        )


def reassemble_state_dict(
    shard_bytes: list[bytes], manifest: CheckpointManifest, *, consume: bool = False
) -> dict[str, torch.Tensor]:
    """Verify every shard against the manifest, then merge into one state
    dict. Raises ShardIntegrityError on the first checksum mismatch or a
    shard-count mismatch -- never returns a partially-trusted result.

    ``consume`` drops each shard's bytes from the list as soon as they are
    loaded, so the peak is the state dict plus ONE shard instead of both in
    full."""
    if len(shard_bytes) != manifest.num_shards:
        raise ShardIntegrityError(
            f"expected {manifest.num_shards} shards, got {len(shard_bytes)}"
        )
    merged: dict[str, torch.Tensor] = {}
    for idx in range(len(shard_bytes)):
        data = shard_bytes[idx]
        if consume:
            shard_bytes[idx] = None
        verify_shard(idx, data, manifest)
        merged.update(torch.load(io.BytesIO(data), weights_only=True))
        del data
    return merged


# --------------------------------------------------------------------- #
# Deltas: version B as a lossless XOR against a version A the fetcher holds.
# --------------------------------------------------------------------- #

#: Blob layout: this magic, the zstd frames, a JSON footer (versions plus one
#: entry per tensor), then the footer's length as a little-endian u64. Footer
#: last so the relay can stream frames to disk a shard at a time.
_DELTA_MAGIC = b"PFDELTA1"
#: zstd level 3 (its default): the XOR is mostly zero bytes, which every level
#: crushes; higher levels only cost relay CPU.
_DELTA_LEVEL = 3
#: Each tensor is cut into chunks of at most this many bytes, every chunk its
#: own zstd frame and SHA-256, so one big tensor (an embedding) spreads over
#: every core instead of pinning one -- on the relay encoding and on each
#: worker decoding and verifying.
_CHUNK_BYTES = 16 << 20
_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _codec_pool() -> ThreadPoolExecutor:
    """One pool per process, as wide as the machine: zstd, hashlib and torch's
    XOR all release the GIL, so threads scale with cores. Never submit to it
    from one of its own threads (a job waiting on jobs can starve the pool)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(os.cpu_count() or 1,
                                       thread_name_prefix="relay-codec")
        return _POOL


def manifest_tag(manifest: CheckpointManifest) -> str:
    """A published version's content identity. Version numbers restart every
    run while a relay can outlive it, so deltas name their base by version AND
    this tag."""
    return hashlib.sha256("|".join(manifest.shard_checksums).encode()).hexdigest()[:16]


def _flat_bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s raw bit pattern as a flat integer tensor of the same width --
    XOR on the bits, not arithmetic on the values, is what makes the delta
    exact (it preserves -0.0, NaN payloads, everything)."""
    flat = t.detach().contiguous().reshape(-1)
    return flat.view(_INT_VIEW[flat.element_size()])


def _chunk_ranges(numel: int, chunk_elems: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk_elems, numel)) for s in range(0, numel, chunk_elems)] or [(0, 0)]


def _digest(bits: torch.Tensor) -> str:
    return hashlib.sha256(bits.view(torch.uint8).numpy()).hexdigest()


def _encode_entries(
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    digests: dict[str, list[str]] | None = None,
) -> tuple[list[dict], list[bytes], dict[str, list[str]]]:
    """Every target tensor (sorted by name) as a delta entry plus its frames,
    all chunks of all tensors encoded in parallel. ``digests``: the target's
    chunk digests when already known -- the relay caches them per version, so
    cutting one target against several bases hashes it once. Returns
    ``(entries, frames in file order, the target's digests)``."""
    plan = []
    for name in sorted(target):
        t = target[name]
        t_bits = _flat_bits(t)
        b = base.get(name)
        if b is not None and b.dtype == t.dtype and b.shape == t.shape:
            kind, b_bits = "xor", _flat_bits(b)
        else:  # a tensor the base lacks, or reshaped: ship it whole
            kind, b_bits = "raw", None
        chunk_elems = max(1, _CHUNK_BYTES // t_bits.element_size())
        plan.append((name, t, kind, t_bits, b_bits, chunk_elems,
                     _chunk_ranges(t_bits.numel(), chunk_elems)))
    known = digests or {}

    def encode_chunk(job):
        _, _, _, t_bits, b_bits, _, _ = plan[job[0]]
        start, end, want_digest = job[1], job[2], job[3]
        piece = t_bits[start:end]
        payload = piece if b_bits is None else piece ^ b_bits[start:end]
        frame = zstandard.ZstdCompressor(level=_DELTA_LEVEL).compress(
            payload.view(torch.uint8).numpy())
        return frame, (_digest(piece) if want_digest else None)

    jobs = [(i, s, e, p[0] not in known) for i, p in enumerate(plan) for s, e in p[6]]
    results = iter(_codec_pool().map(encode_chunk, jobs))
    entries, frames, out_digests = [], [], {}
    for name, t, kind, _, _, chunk_elems, ranges in plan:
        chunk = [next(results) for _ in ranges]
        frames.extend(f for f, _ in chunk)
        out_digests[name] = known[name] if name in known else [d for _, d in chunk]
        entries.append({
            "name": name,
            "kind": kind,
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "chunk_elems": chunk_elems,
            "frames": [len(f) for f, _ in chunk],
            "sha256": out_digests[name],
        })
    return entries, frames, out_digests


def _delta_footer(entries: list[dict], base_version: int, target_version: int) -> bytes:
    footer = json.dumps(
        {"base": base_version, "target": target_version, "entries": entries}
    ).encode()
    return footer + struct.pack("<Q", len(footer))


def encode_delta(
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    base_version: int,
    target_version: int,
) -> bytes:
    """``target`` as a delta against ``base``; ``decode_delta`` inverts it
    bit-exactly given the same ``base``."""
    entries, frames, _ = _encode_entries(base, target)
    return b"".join([_DELTA_MAGIC, *frames,
                     _delta_footer(entries, base_version, target_version)])


def decode_delta(
    blob: bytes,
    base: dict[str, torch.Tensor],
    *,
    base_version: int,
    target_version: int,
) -> dict[str, torch.Tensor]:
    """Rebuild the target state dict from a delta and the ``base`` it was cut
    against. Every chunk is checked against the SHA-256 the relay computed
    from the real target, so a wrong base, a corrupt blob or a codec bug raise
    ShardIntegrityError -- never return different weights. ``base`` is only
    read: the result is all new tensors."""
    view = memoryview(blob)
    try:
        if len(view) < len(_DELTA_MAGIC) + 8 or bytes(view[:len(_DELTA_MAGIC)]) != _DELTA_MAGIC:
            raise ValueError("not a delta blob (or another version of the format)")
        (footer_len,) = struct.unpack("<Q", view[-8:])
        footer_start = len(view) - 8 - footer_len
        if footer_start < len(_DELTA_MAGIC):
            raise ValueError("truncated delta blob")
        meta = json.loads(bytes(view[footer_start:-8]))
        if (meta["base"], meta["target"]) != (base_version, target_version):
            raise ValueError(
                f"delta is v{meta['base']}->v{meta['target']}, "
                f"wanted v{base_version}->v{target_version}")
        outputs, jobs, offset = {}, [], len(_DELTA_MAGIC)
        for entry in meta["entries"]:
            name = entry["name"]
            dtype = getattr(torch, entry["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"unknown dtype {entry['dtype']!r}")
            int_dtype = _INT_VIEW[torch.empty((), dtype=dtype).element_size()]
            numel = math.prod(entry["shape"])
            ranges = _chunk_ranges(numel, int(entry["chunk_elems"]))
            if not len(ranges) == len(entry["frames"]) == len(entry["sha256"]):
                raise ValueError(f"{name!r}: chunk count does not match its shape")
            if entry["kind"] == "xor":
                prev = base.get(name)
                if prev is None or prev.dtype != dtype or prev.numel() != numel:
                    raise ShardIntegrityError(
                        f"delta entry {name!r} does not match the base it was asked against")
                prev_bits = _flat_bits(prev)
            elif entry["kind"] == "raw":
                prev_bits = None
            else:
                raise ValueError(f"unknown entry kind {entry['kind']!r}")
            out = torch.empty(numel, dtype=int_dtype)
            outputs[name] = (out, dtype, entry["shape"])
            for (start, end), size, digest in zip(ranges, entry["frames"], entry["sha256"]):
                jobs.append((name, out, prev_bits, start, end, view[offset:offset + size], digest))
                offset += size
        if offset != footer_start:
            raise ValueError("delta frames do not add up to the blob")
    except (ValueError, KeyError, TypeError, struct.error) as exc:
        raise ShardIntegrityError(f"malformed delta: {exc}") from exc

    def decode_chunk(job):
        name, out, prev_bits, start, end, frame, digest = job
        raw = zstandard.ZstdDecompressor().decompress(frame)
        if len(raw) != (end - start) * out.element_size():
            raise ShardIntegrityError(f"delta entry {name!r}: a chunk has the wrong length")
        piece = out[start:end]
        if raw:
            with warnings.catch_warnings():  # read-only buffer; never written to
                warnings.simplefilter("ignore", UserWarning)
                payload = torch.frombuffer(raw, dtype=out.dtype)
            if prev_bits is None:
                piece.copy_(payload)
            else:
                torch.bitwise_xor(prev_bits[start:end], payload, out=piece)
        if _digest(piece) != digest:
            raise ShardIntegrityError(f"delta entry {name!r} reconstructs to the wrong bytes")

    try:
        for _ in _codec_pool().map(decode_chunk, jobs):
            pass
    except zstandard.ZstdError as exc:
        raise ShardIntegrityError(f"malformed delta: {exc}") from exc
    return {name: out.view(dtype).reshape(shape) for name, (out, dtype, shape) in outputs.items()}


# --------------------------------------------------------------------- #
# RelayServer: one relay node.
# --------------------------------------------------------------------- #


_RAM_SPOOL = Path("/dev/shm")
#: A version is spooled in RAM only while RAM has this many times its size
#: free, so the retained versions never squeeze the host or its cgroup.
_RAM_HEADROOM = 4
#: A version some fetcher touched (manifest or shard) this recently is still
#: being downloaded, so eviction spares it beyond ``retain_last``.
_IN_USE_S = 300.0


def _ram_room() -> int:
    """Bytes a RAM-backed spool may take: the least of the tmpfs's free space,
    the host's available memory, and this cgroup's memory headroom (tmpfs
    pages count against the cgroup limit). 0 when there is no /dev/shm."""
    try:
        room = [shutil.disk_usage(_RAM_SPOOL).free]
        with open("/proc/meminfo") as f:
            room += [int(line.split()[1]) * 1024 for line in f
                     if line.startswith("MemAvailable:")]
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.exists() and (cap := limit.read_text().strip()) != "max":
            used = Path("/sys/fs/cgroup/memory.current").read_text()
            room.append(int(cap) - int(used))
    except (OSError, ValueError):
        return 0
    return min(room)


def _sweep_dead_spools(base: Path) -> None:
    """Remove spools left by relays that died without cleanup (SIGKILL, OOM):
    one in RAM pins up to ``retain_last`` checkpoints of memory until reboot.
    Judges only spools this host's relays wrote (``.owner`` = "host pid"), so
    a relay in another PID namespace sharing the dir is never touched."""
    host = socket.gethostname()
    for spool in base.glob("pf-relay-*"):
        try:
            owner_host, pid = (spool / ".owner").read_text().split()
            os.kill(int(pid), 0)
        except ProcessLookupError:
            if owner_host == host:
                shutil.rmtree(spool, ignore_errors=True)
        except (OSError, ValueError):
            pass  # alive under another user, or not a spool of ours


#: Bases each new version's deltas are pre-built against: the versions most
#: recently served, i.e. what fetchers hold. Fetchers chase the latest, so a
#: pool clusters on one or two; the third covers a straggler.
_PREBUILD_BASES = 3
#: A pre-build waiting this long for the next shard gives up (the publisher
#: died mid-upload); a fetcher asking later gets an on-demand build.
_PREBUILD_STALL_S = 600.0


class _DeltaWriter:
    """One delta file written shard by shard: frames stream to a temp file as
    each target shard is cut, the footer goes last, then it's renamed into
    place -- so a delta is visible only once complete."""

    def __init__(self, out: Path, base: int, target: int):
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".part")
        self._file = os.fdopen(fd, "wb")
        self._file.write(_DELTA_MAGIC)
        self._tmp, self._out = Path(tmp), out
        self._base, self._target = base, target
        self._entries: list[dict] = []

    def add(self, entries: list[dict], frames: list[bytes]) -> None:
        for frame in frames:
            self._file.write(frame)
        self._entries.extend(entries)

    def finish(self) -> Path:
        self._file.write(_delta_footer(self._entries, self._base, self._target))
        self._file.close()
        os.replace(self._tmp, self._out)
        return self._out

    def abort(self) -> None:
        self._file.close()
        self._tmp.unlink(missing_ok=True)


class _Prebuild:
    """State of one version's pipelined pre-build: the shard indices landing
    (fed by publish_shard_file) and one future per base, which ensure_delta
    awaits instead of starting a duplicate build."""

    def __init__(self, target: int, paths: dict[int, Path], loop: asyncio.AbstractEventLoop):
        self.target, self.paths = target, paths  # base -> its delta file
        self.landed: asyncio.Queue[int] = asyncio.Queue()
        self.ready = {base: loop.create_future() for base in paths}
        self.task: asyncio.Task | None = None
        self.stopped = False

    def stop(self) -> None:
        """Give up: the target was dropped. Not a task cancel -- that aborts the
        writers while a _cut_shard thread may still be writing to them -- so
        the build stops at its next await instead."""
        self.stopped = True
        self.landed.put_nowait(-1)

    def finish(self, base: int, path: Path) -> None:
        if not self.ready[base].done():
            self.ready[base].set_result(path)

    def fail(self, base: int, exc: Exception) -> None:
        if not self.ready[base].done():
            self.ready[base].set_exception(exc)
            self.ready[base].exception()  # retrieved: no "never retrieved" noise if unawaited


class RelayServer:
    """Shard/manifest store for one relay node: shards in files, manifests in
    memory.

    Files, not Python memory, because the store holds whole checkpoints: ~18 GB
    per version for a 9B model in bf16, ``retain_last`` versions of them. The
    in-memory store this replaced grew to 13.3 GB on a 16 GB hub and was
    OOM-killed mid-publish (run 817f7f419fab); the trainer only saw "Server
    disconnected". Uploads stream into a temp file renamed into place, so a
    shard is visible only once complete; downloads go out with sendfile.

    ``spool_dir=None`` picks per version: RAM (/dev/shm) when it has room
    (see ``_RAM_HEADROOM``), else a temp dir on disk -- both private and
    removed with the server. RAM matters on a big hub: on the dev box's root
    disk the trainer's upload ran at ~115 MB/s (164 s per 18 GB publish,
    longer than a 68 s window, so 2 of 3 publishes were skipped; run
    a8acf96f4daa) against ~1.5 GB/s into /dev/shm. A small hub can't fit
    the versions in RAM and keeps the disk. An explicit ``spool_dir`` holds
    every version.

    Not thread-safe by locking (aiohttp's default single-threaded event loop
    serializes handler bodies between awaits, which is enough here since
    nothing awaits mid-mutation) but IS safe under normal aiohttp concurrency
    for that reason.
    """

    def __init__(self, retain_last: int = 5, spool_dir: str | None = None):
        self.retain_last = retain_last
        self.spool_dir = Path(spool_dir) if spool_dir is not None else None
        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._roots: dict[Path, Path] = {}  # base -> this server's private spool
        self._dirs: dict[int, Path] = {}
        self._landed: dict[int, set[int]] = {}
        self._manifests: dict[int, CheckpointManifest] = {}
        self._touched: dict[int, float] = {}  # version -> last fetcher access
        self._delta_builds: dict[Path, asyncio.Future] = {}  # delta file -> its build
        self._served: dict[int, float] = {}  # version -> last time a fetcher took it
        self._digests: dict[int, dict[str, list[str]]] = {}  # version -> chunk digests
        self._prebuilds: dict[int, _Prebuild] = {}
        self.num_delta_builds = 0  # on demand
        self.num_delta_prebuilds = 0

    def _private_root(self, base: Path) -> Path:
        if base not in self._roots:
            _sweep_dead_spools(base)
            root = Path(tempfile.mkdtemp(prefix="pf-relay-", dir=base))
            (root / ".owner").write_text(f"{socket.gethostname()} {os.getpid()}")
            weakref.finalize(self, shutil.rmtree, root, True)
            self._roots[base] = root
        return self._roots[base]

    def _spool_root(self, nbytes: int) -> Path:
        if self.spool_dir is not None:
            return self.spool_dir
        if _ram_room() >= _RAM_HEADROOM * nbytes:
            return self._private_root(_RAM_SPOOL)
        return self._private_root(Path(tempfile.gettempdir()))

    def _version_dir(self, version: int) -> Path:
        return self._dirs[version]

    def latest_version(self) -> int | None:
        """The newest version whose shards have ALL landed. A publish posts its
        manifest before its shards, so the newest manifest can name shards still
        in flight (or never coming: a trainer that exits mid-publish); a fetcher
        handed it gets 404s and re-downloads the shards that did land on every
        retry."""
        complete = [v for v in self._manifests if self.complete_manifest(v)]
        return max(complete) if complete else None

    def complete_manifest(self, version: int) -> CheckpointManifest | None:
        """``version``'s manifest once all its shards have landed, else None."""
        m = self._manifests.get(version)
        return m if m is not None and len(self._landed.get(version, ())) == m.num_shards else None

    def touch(self, version: int) -> None:
        """Note a fetcher reading ``version``, which keeps it from eviction."""
        self._touched[version] = time.monotonic()

    def _evict_old(self) -> None:
        if len(self._manifests) <= self.retain_last:
            return
        keep = set(sorted(self._manifests, reverse=True)[: self.retain_last])
        now = time.monotonic()
        keep |= {v for v, t in self._touched.items() if now - t < _IN_USE_S}
        for version in list(self._manifests):
            if version not in keep:
                self._drop_version(version)

    def _drop_version(self, version: int) -> None:
        del self._manifests[version]
        self._landed.pop(version, None)
        self._touched.pop(version, None)
        self._served.pop(version, None)
        self._digests.pop(version, None)
        if (build := self._prebuilds.get(version)) is not None:
            build.stop()
        # A download already streaming one of these files keeps its
        # open fd, so unlinking never cuts a transfer short.
        if (vdir := self._dirs.pop(version, None)) is not None:
            shutil.rmtree(vdir, ignore_errors=True)

    def publish_manifest(self, version: int, manifest: CheckpointManifest) -> None:
        old = self._manifests.get(version)
        if old is not None and manifest_tag(old) != manifest_tag(manifest):
            # Other content under a version number this relay still holds (a
            # new run on a standing relay): start the version over, or its
            # landed shards, chunk digests and deltas would be the old content's.
            self._drop_version(version)
        self._manifests[version] = manifest
        self._landed.setdefault(version, set())
        # Evict first: the versions it frees count toward this one's room.
        self._evict_old()
        if version in self._manifests and version not in self._dirs:
            nbytes = sum(manifest.shard_sizes)
            root = self._spool_root(nbytes)
            self._dirs[version] = root / str(version)
            self._dirs[version].mkdir(exist_ok=True)
            logger.info("spooling v%d (%.2f GB) in %s", version, nbytes / 1e9, root)
            self._start_prebuild(version)

    def new_shard_file(self, version: int, idx: int) -> Path:
        """A temp file in ``version``'s spool dir to stream shard ``idx`` into;
        publish_shard_file then commits it."""
        if version not in self._manifests:
            raise KeyError(f"no manifest published for version {version}")
        fd, tmp = tempfile.mkstemp(dir=self._version_dir(version),
                                   prefix=f".{idx}.", suffix=".part")
        os.close(fd)
        return Path(tmp)

    def publish_shard_file(self, version: int, idx: int, tmp: Path) -> None:
        """Commit a fully written temp file as shard ``idx`` of ``version``."""
        if version not in self._manifests:  # evicted while it uploaded
            tmp.unlink(missing_ok=True)
            raise KeyError(f"no manifest published for version {version}")
        os.replace(tmp, self._version_dir(version) / str(idx))
        self._landed[version].add(idx)
        if (build := self._prebuilds.get(version)) is not None:
            build.landed.put_nowait(idx)

    def publish_shard(self, version: int, idx: int, data: bytes) -> None:
        tmp = self.new_shard_file(version, idx)
        tmp.write_bytes(data)
        self.publish_shard_file(version, idx, tmp)

    def get_manifest(self, version: int) -> CheckpointManifest | None:
        return self._manifests.get(version)

    def shard_path(self, version: int, idx: int) -> Path | None:
        if idx not in self._landed.get(version, ()):
            return None
        return self._version_dir(version) / str(idx)

    def get_shard(self, version: int, idx: int) -> bytes | None:
        path = self.shard_path(version, idx)
        return path.read_bytes() if path is not None else None

    def _load_shard(self, version: int, idx: int) -> dict[str, torch.Tensor]:
        # Not via shard_path: that is the serving accessor (what fetchers GET),
        # and the delta builds' own reads are not downloads.
        if idx not in self._landed.get(version, ()) or version not in self._dirs:
            raise KeyError(f"no shard {idx} for version {version}")
        return torch.load(self._dirs[version] / str(idx), weights_only=True)

    def delta_path(self, base: int, target: int) -> Path:
        # Inside the target's spool dir, so evicting the target removes it; named
        # by the base's content too, so a base republished as other content
        # never gets a delta cut against the old one.
        tag = manifest_tag(self._manifests[base])
        return self._version_dir(target) / f"delta-from-{base}-{tag}"

    def note_served(self, version: int) -> None:
        """A fetcher downloaded ``version`` (whole or as a delta), so it now
        holds it: the versions served most recently are the bases the next
        version's deltas get pre-built against."""
        self._served[version] = time.monotonic()

    def note_moved(self, base: int, target: int) -> None:
        """A fetcher took ``target`` as a delta from ``base``, so it holds
        ``base`` no longer. Dropping ``base`` keeps pre-builds to what is held,
        not every version ever served. Other fetchers still on ``base`` lose
        nothing for this target (its delta is built) and get the next one
        built on request."""
        self._served.pop(base, None)
        self.note_served(target)

    def _prebuild_bases(self, target: int) -> list[int]:
        recent = sorted(self._served, key=self._served.__getitem__, reverse=True)
        return [v for v in recent
                if v < target and self.complete_manifest(v) is not None][:_PREBUILD_BASES]

    def _cut_shard(
        self, target: int, idx: int, writers: dict[int, "_DeltaWriter"]
    ) -> list[int]:
        """Encode target shard ``idx`` against every base in ``writers`` and
        append the frames to each base's delta, loading the target shard once
        and hashing it once (cached per version). Blocking. Returns the bases
        that failed (evicted mid-build); their writers are left to the caller.

        Shard ``idx`` of one model's versions holds the same names (the
        partition is deterministic), so the base's same-index shard is all a
        cut needs; one whose names it lacks (the model changed) is cut against
        the whole base instead."""
        target_sd = self._load_shard(target, idx)
        digests = self._digests.setdefault(target, {})
        failed = []
        for base, writer in writers.items():
            try:
                base_m = self.complete_manifest(base)
                if base_m is None:
                    raise KeyError(f"base v{base} is gone")
                base_sd = self._load_shard(base, idx) if idx < base_m.num_shards else {}
                if not target_sd.keys() <= base_sd.keys():
                    base_sd = {}
                    for i in range(base_m.num_shards):
                        base_sd.update(self._load_shard(base, i))
                known = {n: digests[n] for n in target_sd if n in digests}
                entries, frames, computed = _encode_entries(base_sd, target_sd, known)
                digests.update(computed)
                writer.add(entries, frames)
            except (KeyError, OSError) as exc:
                logger.warning("delta v%d->v%d: shard %d failed: %s", base, target, idx, exc)
                failed.append(base)
        return failed

    def build_delta(self, base: int, target: int) -> Path:
        """Write ``target`` as a delta against ``base`` (both complete) into the
        target's spool dir -- the on-demand path, for a base no pre-build
        covered. Blocking, so callers run it on a thread; a shard at a time, so
        the peak is two shards, not two checkpoints."""
        target_m = self.complete_manifest(target)
        if target_m is None or self.complete_manifest(base) is None:
            raise KeyError(f"v{base} and v{target} must both be complete")
        t0 = time.perf_counter()
        writer = _DeltaWriter(self.delta_path(base, target), base, target)
        try:
            for idx in range(target_m.num_shards):
                if self._cut_shard(target, idx, {base: writer}):
                    raise KeyError(f"v{base} or v{target} was evicted mid-build")
            out = writer.finish()
        except BaseException:
            writer.abort()
            raise
        self.num_delta_builds += 1
        size, full = out.stat().st_size, sum(target_m.shard_sizes)
        logger.info("built delta v%d->v%d on demand: %.1f MB (%.0fx smaller than "
                    "%.2f GB) in %.1fs", base, target, size / 1e6,
                    full / max(size, 1), full / 1e9, time.perf_counter() - t0)
        return out

    def _start_prebuild(self, target: int) -> None:
        """Pre-build ``target``'s deltas against the versions fetchers hold,
        pipelined with its upload: each shard is cut as it lands (see
        publish_shard_file), so by the time the last one lands -- the moment a
        fetcher can see the version at all -- only that shard's work is left.
        Needs a running event loop (the HTTP handlers); skipped without one."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        bases = self._prebuild_bases(target)
        if not bases:
            return
        for base in bases:
            self.touch(base)  # keep the bases through the build
        build = _Prebuild(target, {b: self.delta_path(b, target) for b in bases}, loop)
        self._prebuilds[target] = build
        for base, path in build.paths.items():
            self._delta_builds[path] = build.ready[base]
        build.task = loop.create_task(self._run_prebuild(build))

    async def _run_prebuild(self, build: "_Prebuild") -> None:
        target = build.target
        writers: dict[int, _DeltaWriter] = {}
        try:
            for base, path in build.paths.items():
                writers[base] = _DeltaWriter(path, base, target)
            num_shards = self._manifests[target].num_shards
            done: set[int] = set()
            last_landed = time.perf_counter()
            while len(done) < num_shards and writers:
                idx = await asyncio.wait_for(build.landed.get(), _PREBUILD_STALL_S)
                last_landed = time.perf_counter()
                if build.stopped:
                    raise KeyError(f"v{target} was dropped mid-build")
                if idx in done:
                    continue
                done.add(idx)
                for base in await asyncio.to_thread(self._cut_shard, target, idx, writers):
                    build.fail(base, KeyError(f"v{base} was evicted mid-build"))
                    writers.pop(base).abort()
            if build.stopped:  # dropped during the last cut
                raise KeyError(f"v{target} was dropped mid-build")
            full = sum(self._manifests[target].shard_sizes)
            for base, writer in writers.items():
                out = await asyncio.to_thread(writer.finish)
                self.num_delta_prebuilds += 1
                size = out.stat().st_size
                logger.info("pre-built delta v%d->v%d: %.1f MB (%.0fx smaller than %.2f GB), "
                            "ready %.2fs after the last shard landed", base, target,
                            size / 1e6, full / max(size, 1), full / 1e9,
                            time.perf_counter() - last_landed)
                build.finish(base, out)
            writers.clear()
        except BaseException as exc:  # incl. a dropped target, a stalled upload, shutdown
            if not (build.stopped or isinstance(exc, asyncio.CancelledError)):
                logger.warning("pre-build of v%d's deltas stopped: %r", target, exc)
            for base, writer in writers.items():
                writer.abort()
                build.fail(base, exc if isinstance(exc, Exception) else KeyError(str(exc)))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            if self._prebuilds.get(target) is build:
                del self._prebuilds[target]
            for base, path in build.paths.items():
                if self._delta_builds.get(path) is build.ready[base]:
                    del self._delta_builds[path]

    async def ensure_delta(self, base: int, target: int) -> Path:
        """The delta file: already built, being pre-built (wait for it), or
        built now -- once, however many fetchers ask for the same pair."""
        path = self.delta_path(base, target)
        if path.exists():
            return path
        pending = self._delta_builds.get(path)
        if pending is not None:
            try:
                return await asyncio.shield(pending)
            except Exception:  # that build failed (logged there): build it here instead
                if self._delta_builds.get(path) is pending:
                    del self._delta_builds[path]
            if path.exists():
                return path
        build = self._delta_builds.get(path)
        if build is None:
            build = asyncio.ensure_future(asyncio.to_thread(self.build_delta, base, target))
            self._delta_builds[path] = build
            build.add_done_callback(
                lambda done: self._delta_builds.pop(path, None)
                if self._delta_builds.get(path) is done else None)
        return await asyncio.shield(build)

    def app(self) -> web.Application:
        # client_max_size=0 disables aiohttp's default 1MB body cap: a
        # checkpoint shard is state_dict_bytes / num_shards, routinely
        # hundreds of MB to low GBs -- far past the default even for the
        # smallest model this swarm supports. Relay workers are already
        # trusted (no TOPLOC-style admission check), so an unbounded body
        # size adds no new trust assumption.
        app = web.Application(client_max_size=0)
        app[_RELAY_KEY] = self
        app.add_routes(
            [
                web.post("/publish/{version}/manifest", _handle_publish_manifest),
                web.post("/publish/{version}/shard/{idx}", _handle_publish_shard),
                web.get("/manifest/latest", _handle_manifest_latest),
                web.get("/manifest/{version}", _handle_manifest_version),
                web.get("/shard/{version}/{idx}", _handle_get_shard),
                web.get("/delta/{base}/{target}", _handle_get_delta),
            ]
        )
        return app


async def _handle_publish_manifest(request: web.Request) -> web.Response:
    relay: RelayServer = request.app[_RELAY_KEY]
    version = int(request.match_info["version"])
    manifest = CheckpointManifest.from_json(await request.json())
    relay.publish_manifest(version, manifest)
    return web.Response(status=204)


async def _handle_publish_shard(request: web.Request) -> web.Response:
    relay: RelayServer = request.app[_RELAY_KEY]
    version = int(request.match_info["version"])
    idx = int(request.match_info["idx"])
    try:
        tmp = relay.new_shard_file(version, idx)
    except KeyError as exc:
        return web.Response(status=404, text=str(exc))
    # Streamed to disk, never buffered whole: one shard of a 9B checkpoint is
    # ~4.5 GB. Writes go to a thread so a slow disk never stalls the loop that
    # is also serving downloads.
    t0, size = time.perf_counter(), 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.content.iter_chunked(_UPLOAD_CHUNK):
                await asyncio.to_thread(f.write, chunk)
                size += len(chunk)
        relay.publish_shard_file(version, idx, tmp)
    except KeyError as exc:
        return web.Response(status=404, text=str(exc))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dt = time.perf_counter() - t0
    logger.info("stored v%d shard %d: %.2f GB in %.1fs (%.0f MB/s)",
                version, idx, size / 1e9, dt, size / 1e6 / max(dt, 1e-9))
    return web.Response(status=204)


async def _handle_manifest_latest(request: web.Request) -> web.Response:
    relay: RelayServer = request.app[_RELAY_KEY]
    version = relay.latest_version()
    if version is None:
        return web.Response(status=404, text="no checkpoint published yet")
    relay.touch(version)
    return web.json_response(relay.get_manifest(version).to_json())


async def _handle_manifest_version(request: web.Request) -> web.Response:
    """A retained, complete version's manifest: how a fetcher confirms that a
    version its site already started is this run's and still on the relay."""
    relay: RelayServer = request.app[_RELAY_KEY]
    version = int(request.match_info["version"])
    manifest = relay.complete_manifest(version)
    if manifest is None:
        return web.Response(status=404, text=f"no complete version {version}")
    relay.touch(version)
    return web.json_response(manifest.to_json())


async def _handle_get_shard(request: web.Request) -> web.Response:
    relay: RelayServer = request.app[_RELAY_KEY]
    version = int(request.match_info["version"])
    idx = int(request.match_info["idx"])
    path = relay.shard_path(version, idx)
    if path is None:
        return web.Response(status=404, text=f"no shard {idx} for version {version}")
    relay.touch(version)
    relay.note_served(version)
    # sendfile straight from the spool: no multi-GB copy through Python.
    # Returned unprepared: aiohttp prepares it, and a FileResponse prepared
    # here first asserts on that second prepare. Download rates are logged by
    # the fetching RelayClient instead.
    return web.FileResponse(path)


async def _handle_get_delta(request: web.Request) -> web.Response:
    """``target`` as a delta against ``base`` (see decode_delta). 404 when either
    version is gone or incomplete, 409 when ``?base_tag=`` says the fetcher's
    base is different content under the same version number (a relay that
    outlived a run) -- both mean: download the whole checkpoint instead."""
    relay: RelayServer = request.app[_RELAY_KEY]
    base = int(request.match_info["base"])
    target = int(request.match_info["target"])
    base_m, target_m = relay.complete_manifest(base), relay.complete_manifest(target)
    if base_m is None or target_m is None or base >= target:
        return web.Response(status=404, text=f"no delta v{base}->v{target}")
    if request.query.get("base_tag", manifest_tag(base_m)) != manifest_tag(base_m):
        return web.Response(status=409, text=f"v{base} here is different content")
    relay.touch(base)
    relay.touch(target)
    try:
        path = await relay.ensure_delta(base, target)
    except (KeyError, OSError) as exc:  # evicted mid-build, or the spool failed
        logger.warning("delta v%d->v%d failed: %s", base, target, exc)
        return web.Response(status=404, text=f"no delta v{base}->v{target}: {exc}")
    relay.note_moved(base, target)
    return web.FileResponse(path, headers={"X-Delta-Target-Tag": manifest_tag(target_m)})


async def run_relay_server(
    host: str = "0.0.0.0", port: int = 8765, retain_last: int = 5,
    spool_dir: str | None = None,
):
    """Start a relay server; returns the ``web.AppRunner`` (caller keeps it
    alive and calls ``.cleanup()`` to stop)."""
    relay = RelayServer(retain_last=retain_last, spool_dir=spool_dir)
    runner = web.AppRunner(relay.app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    logger.info(
        "relay server listening on %s:%d (retain_last=%d, spool_dir=%s)",
        host, port, retain_last,
        relay.spool_dir or "auto: /dev/shm when it fits, else disk",
    )
    return runner


# --------------------------------------------------------------------- #
# RelayClient: publisher/fetcher side.
# --------------------------------------------------------------------- #


class RelayClient:
    """Publishes to / fetches from a tier of relay servers.

    Tracks a per-relay ``success_rate``/``bandwidth`` EMA (optimistic init so
    untried relays get a fair first shot) and selects relays probabilistically
    weighted by ``success_rate * bandwidth`` -- SHARDCAST's exact rule.
    ``rng`` is injectable for deterministic tests.
    """

    def __init__(
        self,
        relay_urls: list[str],
        *,
        rng: random.Random | None = None,
        timeout_s: float = 30.0,
        stall_timeout_s: float = 120.0,
        site_cache_dir: str | None = None,
        use_deltas: bool = True,
    ):
        """``use_deltas``: ask for each checkpoint newer than the last one
        returned as a delta against it (``GET /delta``), falling back to the
        whole download whenever the relay can't serve one or it fails
        verification.

        ``site_cache_dir``: a directory shared with the other fetchers on this
        site (a cluster's shared filesystem). Fetchers then split each version's
        shards between them -- every shard crosses the link to the relay ONCE per
        site -- and read the rest from there (_fetch_shards_via_site). Four
        generators behind one Slurm login node pulled 4 x 18 GB per 9B version
        through its ~1 Gbps link: ~10 min a version. None: download directly."""
        if not relay_urls:
            raise ValueError("relay_urls must be non-empty")
        self.relay_urls = list(relay_urls)
        self._site_cache = Path(site_cache_dir) if site_cache_dir else None
        self._rng = rng or random.Random()
        # A checkpoint transfer is multi-GB, so `total` is the wrong bound: it
        # caps the whole manifest+shards exchange regardless of progress. At
        # Qwen3-4B (~8.8 GB bf16 across 4 shards) a 30 s total made publishing
        # arithmetically impossible -- run a033803b3539 logged 26 consecutive
        # `publish of checkpoint v1 failed ()` (an empty str(), i.e.
        # asyncio.TimeoutError), with shards 0-2 landing and shard 3 always
        # 404, so no worker ever assembled a checkpoint and generation never
        # started. 0.6B (~1.2 GB) fit inside 30 s, which is why it took a 4B
        # run to surface.
        #
        # Bound the things that indicate a BROKEN relay instead: connecting,
        # and going quiet mid-transfer. A slow-but-progressing upload now
        # finishes rather than being killed on a stopwatch.
        self._transfer_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=timeout_s,
            sock_read=stall_timeout_s,
        )
        self._success_rate = {url: 1.0 for url in self.relay_urls}
        self._bandwidth = {
            url: 1.0 for url in self.relay_urls
        }  # arbitrary units, EMA of bytes/sec
        self._use_deltas = use_deltas
        # (version, manifest_tag, state dict) of the last checkpoint returned:
        # the base the next delta is cut against. The client's own dict, so a
        # caller emptying the returned one (the worker does) leaves it intact.
        self._base: tuple[int, str, dict[str, torch.Tensor]] | None = None
        #: How the last fetch_latest arrived: {"kind": "delta"|"full", "bytes",
        #: "seconds"} -- for logs, metrics and tests.
        self.last_fetch: dict | None = None

    def stats(self, url: str) -> tuple[float, float]:
        return self._success_rate[url], self._bandwidth[url]

    def _weighted_choice(self, candidates: list[str]) -> str:
        weights = [self._success_rate[u] * self._bandwidth[u] for u in candidates]
        if sum(weights) <= 0:
            return self._rng.choice(candidates)
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def select_relay(self) -> str:
        return self._weighted_choice(self.relay_urls)

    def _record_success(self, url: str, num_bytes: int, elapsed_s: float) -> None:
        self._success_rate[url] = (
            _EMA_ALPHA + (1 - _EMA_ALPHA) * self._success_rate[url]
        )
        if elapsed_s > 0:
            observed_bw = num_bytes / elapsed_s
            self._bandwidth[url] = (
                _EMA_ALPHA * observed_bw + (1 - _EMA_ALPHA) * self._bandwidth[url]
            )

    def _record_failure(self, url: str) -> None:
        self._success_rate[url] = (1 - _EMA_ALPHA) * self._success_rate[url]

    async def publish(
        self,
        version: int,
        shard_bytes: list[bytes],
        manifest: CheckpointManifest,
        *,
        relays: list[str] | None = None,
    ) -> None:
        """Upload the manifest + every shard to every target relay (default:
        the whole configured tier), so any fetcher can reach any of them."""
        targets = relays if relays is not None else self.relay_urls
        total_bytes = sum(len(d) for d in shard_bytes)
        async with aiohttp.ClientSession(
            timeout=self._transfer_timeout
        ) as session:
            for url in targets:
                t0 = time.perf_counter()
                async with session.post(
                    f"{url}/publish/{version}/manifest", json=manifest.to_json()
                ) as resp:
                    resp.raise_for_status()
                for idx, data in enumerate(shard_bytes):
                    async with session.post(
                        f"{url}/publish/{version}/shard/{idx}", data=data
                    ) as resp:
                        resp.raise_for_status()
                # Measured, because we could not explain the failure without it:
                # the run that exposed the `total` timeout showed ~8.9 Gbps on
                # the hub link, at which 8.8 GB should take ~8 s, yet 30 s was
                # never enough. Log the real rate so the next slow publish is a
                # number instead of a guess.
                dt = time.perf_counter() - t0
                logger.info(
                    "published checkpoint v%d to %s: %d shards, %.2f GB in "
                    "%.1fs (%.0f MB/s)",
                    version, url, len(shard_bytes), total_bytes / 1e9, dt,
                    total_bytes / 1e6 / max(dt, 1e-9),
                )

    async def _fetch_manifest(
        self, session: aiohttp.ClientSession, url: str, min_version: int
    ) -> CheckpointManifest | None:
        try:
            async with session.get(f"{url}/manifest/latest") as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # TimeoutError too: aiohttp raises plain asyncio.TimeoutError (not
            # a ClientError) on a slow relay; treat it as this relay failing.
            self._record_failure(url)
            return None
        manifest = CheckpointManifest.from_json(data)
        return manifest if manifest.version > min_version else None

    async def _fetch_shards(
        self, session: aiohttp.ClientSession, url: str, manifest: CheckpointManifest
    ) -> list[bytes] | None:
        shard_bytes: list[bytes] = []
        total_bytes = 0
        t0 = time.monotonic()
        try:
            for idx in range(manifest.num_shards):
                async with session.get(f"{url}/shard/{manifest.version}/{idx}") as resp:
                    if resp.status != 200:
                        self._record_failure(url)
                        return None
                    data = await resp.read()
                total_bytes += len(data)
                shard_bytes.append(data)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            self._record_failure(url)
            return None
        dt = time.monotonic() - t0
        self._record_success(url, total_bytes, dt)
        logger.info(
            "fetched checkpoint v%d from %s: %d shards, %.2f GB in %.1fs (%.0f MB/s)",
            manifest.version, url, manifest.num_shards, total_bytes / 1e9, dt,
            total_bytes / 1e6 / max(dt, 1e-9),
        )
        return shard_bytes

    # ----------------------------------------------------------------- #
    # Site cache: split each version's shards across a site's fetchers.
    # ----------------------------------------------------------------- #

    def _site_version_dir(self, url: str, manifest: CheckpointManifest) -> Path:
        # Keyed by the manifest's checksums, not just the version: versions restart
        # at 1 every run, and a standing hub reuses its URL across runs.
        relay = hashlib.sha256(url.encode()).hexdigest()[:12]
        tag = hashlib.sha256("|".join(manifest.shard_checksums).encode()).hexdigest()[:12]
        return self._site_cache / relay / f"v{manifest.version}-{tag}"

    @staticmethod
    async def _read_cached_shard(
        path: Path, idx: int, manifest: CheckpointManifest
    ) -> bytes | None:
        """A complete shard another fetcher left, verified; a bad one is removed so
        it gets downloaded again."""
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            return None
        try:
            await asyncio.to_thread(verify_shard, idx, data, manifest)
        except ShardIntegrityError:
            logger.warning("site cache: %s fails its checksum; re-fetching", path)
            path.unlink(missing_ok=True)
            return None
        return data

    @staticmethod
    def _try_claim(claim: Path) -> bool:
        # O_EXCL create, not flock: flock is not dependable on BeeGFS.
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        os.write(fd, f"{os.uname().nodename}:{os.getpid()}".encode())
        os.close(fd)
        return True

    @staticmethod
    def _steal_if_stale(claim: Path, final: Path) -> None:
        """Drop a claim whose holder stopped refreshing it (killed mid-download)."""
        try:
            age = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            return
        if age > _SITE_CLAIM_STALE_S and not final.exists():
            logger.warning("site cache: claim %s idle %.0fs; taking the shard over",
                           claim, age)
            claim.unlink(missing_ok=True)

    async def _download_to_site(
        self, session: aiohttp.ClientSession, url: str, manifest: CheckpointManifest,
        idx: int, final: Path, claim: Path,
    ) -> bytearray | None:
        """Download a claimed shard into the site cache: streamed to a temp file,
        verified, renamed into place -- so the cache never holds a bad shard."""
        tmp = final.with_name(f".{idx}.{os.getpid()}.part")
        buf = bytearray()
        beat = time.monotonic()
        try:
            async with session.get(f"{url}/shard/{manifest.version}/{idx}") as resp:
                if resp.status != 200:
                    return None
                with open(tmp, "wb") as f:
                    async for chunk in resp.content.iter_chunked(_UPLOAD_CHUNK):
                        buf += chunk
                        await asyncio.to_thread(f.write, chunk)
                        if time.monotonic() - beat > _SITE_HEARTBEAT_S:
                            os.utime(claim, None)
                            beat = time.monotonic()
            await asyncio.to_thread(verify_shard, idx, buf, manifest)
            os.replace(tmp, final)
            return buf
        except (aiohttp.ClientError, asyncio.TimeoutError, ShardIntegrityError, OSError):
            return None
        finally:
            tmp.unlink(missing_ok=True)
            claim.unlink(missing_ok=True)

    async def _site_joinable_manifest(
        self, session: aiohttp.ClientSession, url: str, min_version: int
    ) -> CheckpointManifest | None:
        """The newest version newer than ``min_version`` that this site's other
        fetchers already started (or finished), so a site settles on ONE version
        per round and splits its download, instead of each fetcher taking the
        relay's latest at whatever moment it happens to be ready.

        A candidate is confirmed with the relay by version AND content tag:
        version numbers restart every run, and the site cache can still hold a
        previous run's dirs under the same relay URL."""
        root = self._site_cache / hashlib.sha256(url.encode()).hexdigest()[:12]
        try:
            names = [d.name for d in root.iterdir()]
        except OSError:
            return None
        found = sorted(((int(m[1]), m[2]) for n in names
                        if (m := re.fullmatch(r"v(\d+)-([0-9a-f]{12})", n))
                        and int(m[1]) > min_version), reverse=True)
        for version, tag in found:
            try:
                async with session.get(f"{url}/manifest/{version}") as resp:
                    if resp.status != 200:
                        continue
                    manifest = CheckpointManifest.from_json(await resp.json())
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return None
            if self._site_version_dir(url, manifest).name == f"v{version}-{tag}":
                return manifest
        return None

    @staticmethod
    def _prune_site_cache(current: Path) -> None:
        """Keep this version and the newest other one (a slower fetcher may still
        be reading it); drop the rest, including other runs' versions."""
        others = sorted((d for d in current.parent.iterdir()
                         if d.is_dir() and d != current),
                        key=lambda d: d.stat().st_mtime, reverse=True)
        for old in others[1:]:
            shutil.rmtree(old, ignore_errors=True)

    async def _fetch_shards_via_site(
        self, session: aiohttp.ClientSession, url: str, manifest: CheckpointManifest
    ) -> list[bytes] | None:
        """Every fetcher on the site walks the shards: it reads the ones already in
        the cache, downloads the ones it can claim, and waits for the ones another
        fetcher holds. Fetchers that start together each claim a different shard,
        so the relay link carries each shard once instead of once per fetcher."""
        vdir = self._site_version_dir(url, manifest)
        vdir.mkdir(parents=True, exist_ok=True)
        shards: dict[int, bytes] = {}
        downloaded: list[int] = []
        downloaded_bytes = 0
        t0 = time.monotonic()
        while len(shards) < manifest.num_shards:
            progressed = False
            for idx in range(manifest.num_shards):
                if idx in shards:
                    continue
                final, claim = vdir / str(idx), vdir / f".claim-{idx}"
                data = await self._read_cached_shard(final, idx, manifest)
                if data is None and self._try_claim(claim):
                    if final.exists():   # finished between our read and our claim
                        claim.unlink(missing_ok=True)
                        continue
                    data = await self._download_to_site(
                        session, url, manifest, idx, final, claim)
                    if data is None:
                        self._record_failure(url)
                        return None
                    downloaded.append(idx)
                    downloaded_bytes += len(data)
                elif data is None:
                    self._steal_if_stale(claim, final)
                    continue
                shards[idx] = data
                progressed = True
            if not progressed:
                await asyncio.sleep(_SITE_POLL_S)
        dt = time.monotonic() - t0
        if downloaded_bytes:
            self._record_success(url, downloaded_bytes, dt)
        total = sum(len(d) for d in shards.values())
        logger.info(
            "fetched checkpoint v%d via site cache: downloaded shards %s from %s "
            "(%.2f GB), read %d from the cache; %.2f GB in %.1fs",
            manifest.version, downloaded, url, downloaded_bytes / 1e9,
            manifest.num_shards - len(downloaded), total / 1e9, dt,
        )
        try:
            self._prune_site_cache(vdir)
        except OSError:
            pass
        return [shards[i] for i in range(manifest.num_shards)]

    async def _fetch_delta(
        self, session: aiohttp.ClientSession, url: str, manifest: CheckpointManifest
    ) -> dict[str, torch.Tensor] | None:
        """``manifest``'s checkpoint as a delta against the held base, or None
        (no base, relay can't serve it, transfer failed, or it failed
        verification) -- the caller then downloads it whole."""
        if not self._use_deltas or self._base is None:
            return None
        base_version, base_tag, base_sd = self._base
        if base_version >= manifest.version:
            return None
        t0 = time.monotonic()
        try:
            async with session.get(
                f"{url}/delta/{base_version}/{manifest.version}",
                params={"base_tag": base_tag},
            ) as resp:
                if resp.status != 200:
                    logger.info("no delta v%d->v%d from %s (status %d); fetching whole",
                                base_version, manifest.version, url, resp.status)
                    return None
                if resp.headers.get("X-Delta-Target-Tag") != manifest_tag(manifest):
                    logger.warning("delta v%d->v%d from %s is for other content; "
                                   "fetching whole", base_version, manifest.version, url)
                    return None
                blob = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.info("delta v%d->v%d from %s failed (%s); fetching whole",
                        base_version, manifest.version, url, exc)
            return None
        t_net = time.monotonic() - t0
        try:
            state_dict = await asyncio.to_thread(
                decode_delta, blob, base_sd,
                base_version=base_version, target_version=manifest.version)
        except ShardIntegrityError as exc:
            logger.warning("delta v%d->v%d from %s failed verification (%s); "
                           "fetching whole", base_version, manifest.version, url, exc)
            return None
        except Exception as exc:  # e.g. a base on the wrong device; never fail the fetch
            logger.warning("could not apply delta v%d->v%d from %s (%r); fetching whole",
                           base_version, manifest.version, url, exc)
            return None
        dt = time.monotonic() - t0
        full = sum(manifest.shard_sizes)
        # Not fed to the bandwidth EMA: a ~10 MB transfer measures latency, and
        # would make a fast relay look slow for the next whole download.
        self._success_rate[url] = _EMA_ALPHA + (1 - _EMA_ALPHA) * self._success_rate[url]
        self.last_fetch = {"kind": "delta", "bytes": len(blob), "seconds": dt}
        logger.info(
            "fetched checkpoint v%d from %s as a delta from v%d: %.1f MB instead of "
            "%.2f GB (%.0fx) in %.1fs (%.1fs transfer)",
            manifest.version, url, base_version, len(blob) / 1e6, full / 1e9,
            full / max(len(blob), 1), dt, t_net,
        )
        return state_dict

    def _remember(self, manifest: CheckpointManifest, state_dict: dict) -> None:
        if self._use_deltas:
            self._base = (manifest.version, manifest_tag(manifest), dict(state_dict))

    async def latest_version(self) -> int:
        """The newest complete version any relay serves (0 when none, or none
        reachable) -- read from the manifests alone, nothing downloaded."""
        best = 0
        async with aiohttp.ClientSession(timeout=self._transfer_timeout) as session:
            for url in self.relay_urls:
                manifest = await self._fetch_manifest(session, url, 0)
                if manifest is not None:
                    best = max(best, manifest.version)
        return best

    async def fetch_latest(self, min_version: int = 0) -> tuple[int, dict] | None:
        """Try relays (probabilistically ordered, without replacement) for a
        checkpoint newer than ``min_version``, verifying checksums; a
        connection error or checksum mismatch decays that relay's
        success_rate and moves to the next. Returns ``(version, state_dict)``
        or ``None`` if no relay has anything newer / every attempt failed."""
        remaining = list(self.relay_urls)
        async with aiohttp.ClientSession(
            timeout=self._transfer_timeout
        ) as session:
            while remaining:
                url = self._weighted_choice(remaining)
                remaining.remove(url)

                manifest = None
                if self._site_cache is not None:
                    manifest = await self._site_joinable_manifest(
                        session, url, min_version)
                if manifest is None:
                    manifest = await self._fetch_manifest(session, url, min_version)
                if manifest is None:
                    continue
                state_dict = await self._fetch_delta(session, url, manifest)
                if state_dict is not None:
                    self._remember(manifest, state_dict)
                    return manifest.version, state_dict
                t0 = time.monotonic()
                fetch = (self._fetch_shards_via_site if self._site_cache is not None
                         else self._fetch_shards)
                shard_bytes = await fetch(session, url, manifest)
                if shard_bytes is None:
                    continue
                try:
                    state_dict = reassemble_state_dict(
                        shard_bytes, manifest, consume=True)
                except ShardIntegrityError:
                    logger.warning(
                        "relay %s served a corrupted shard for version %d; "
                        "trying another relay",
                        url,
                        manifest.version,
                    )
                    self._record_failure(url)
                    continue
                self.last_fetch = {"kind": "full", "bytes": sum(manifest.shard_sizes),
                                   "seconds": time.monotonic() - t0}
                self._remember(manifest, state_dict)
                return manifest.version, state_dict
        return None


# --------------------------------------------------------------------- #
# BackgroundPublisher: the trainer-side upload, off the training path.
# --------------------------------------------------------------------- #


class BackgroundPublisher:
    """Publishes a trainer's weights to the relay tier from a background
    thread, so training pauses only for the device->host copy.

    ``stage()`` copies each tensor into a host buffer allocated once and then
    reused -- pinned when CUDA is up. Measured on an H200: 53 GB/s into reused
    pinned buffers against ~4 GB/s into fresh pageable memory, i.e. ~0.35 s
    instead of ~4.6 s for a 9B bf16 checkpoint. ``start()`` then shards,
    hashes and uploads them on a thread, in the trainer process itself: the
    weights never cross to the controller.

    One upload at a time, because the buffers are reused: callers must not
    ``stage()`` while ``busy()``. Outcomes (success or the exception text) are
    collected for ``take_reports()``, never raised on the thread."""

    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}
        self._staged: list[str] = []
        self._thread: threading.Thread | None = None
        self._reports: list[dict] = []
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stage(self, name: str, tensor: torch.Tensor) -> None:
        if self.busy():
            raise RuntimeError("stage() while the previous upload still reads the buffers")
        dst = self._buffers.get(name)
        if dst is None or dst.shape != tensor.shape or dst.dtype != tensor.dtype:
            dst = self._buffers[name] = torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=torch.cuda.is_available()
            )
        dst.copy_(tensor, non_blocking=dst.is_pinned())
        self._staged.append(name)

    def start(self, version: int, relay_urls: list[str], num_shards: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()  # the non-blocking copies landed
        state = {name: self._buffers[name] for name in self._staged}
        self._staged = []
        self._thread = threading.Thread(
            target=self._upload, args=(version, state, relay_urls, num_shards),
            name=f"relay-publish-v{version}", daemon=True,
        )
        self._thread.start()

    def _upload(self, version, state, relay_urls, num_shards) -> None:
        t0 = time.perf_counter()
        try:
            shards = shard_state_dict(state, num_shards)
            manifest = build_manifest(version, shards)
            asyncio.run(RelayClient(relay_urls).publish(version, shards, manifest))
            report = {"version": version, "num_shards": manifest.num_shards,
                      "bytes": sum(manifest.shard_sizes),
                      "seconds": time.perf_counter() - t0}
        except Exception as exc:  # reported to the controller, not lost on the thread
            report = {"version": version, "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self._reports.append(report)

    def take_reports(self) -> list[dict]:
        with self._lock:
            reports, self._reports = self._reports, []
        return reports

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()


# --------------------------------------------------------------------- #
# Standalone relay-node entrypoint.
# --------------------------------------------------------------------- #


async def _serve(host: str, port: int, retain_last: int,
                 spool_dir: str | None) -> None:
    runner = await run_relay_server(host=host, port=port, retain_last=retain_last,
                                    spool_dir=spool_dir)
    print(f"ASYNC_INFERENCE_RELAY_ADDR=http://{host}:{port}", flush=True)
    # A job cancel SIGTERMs us; die through the normal exit so the spool's
    # finalizers run (default SIGTERM skips them and leaked a 34 GB spool).
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    logger.info("relay server serving; ctrl-c to stop")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="async-inference relay server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--retain_last",
        type=int,
        default=5,
        help="checkpoint versions kept before eviction (SHARDCAST's number)",
    )
    parser.add_argument(
        "--spool_dir",
        type=str,
        default=None,
        help="where shards are stored (needs retain_last x checkpoint size); "
        "default: per version, /dev/shm when RAM has room, else a temp dir on "
        "disk, removed on exit",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.host, args.port, args.retain_last, args.spool_dir))
    except KeyboardInterrupt:
        logger.info("relay server shutting down")


if __name__ == "__main__":
    main()
