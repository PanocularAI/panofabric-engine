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
#
# Needs torch (state-dict tensors) but never the torchtitan training stack or
# vLLM, so the standalone relay-server process stays CPU-only, like heloco's
# parameter server (parameter_server.py). Run one relay node per box with:
#   python -m panoengine.decentralized.relay --port 8765

import argparse
import asyncio
import hashlib
import io
import logging
import os
import random
import shutil
import signal
import socket
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import torch
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
    shard_bytes: list[bytes], manifest: CheckpointManifest
) -> dict[str, torch.Tensor]:
    """Verify every shard against the manifest, then merge into one state
    dict. Raises ShardIntegrityError on the first checksum mismatch or a
    shard-count mismatch -- never returns a partially-trusted result."""
    if len(shard_bytes) != manifest.num_shards:
        raise ShardIntegrityError(
            f"expected {manifest.num_shards} shards, got {len(shard_bytes)}"
        )
    merged: dict[str, torch.Tensor] = {}
    for idx, data in enumerate(shard_bytes):
        verify_shard(idx, data, manifest)
        shard_sd = torch.load(io.BytesIO(data), weights_only=True)
        merged.update(shard_sd)
    return merged


# --------------------------------------------------------------------- #
# RelayServer: one relay node.
# --------------------------------------------------------------------- #


_RAM_SPOOL = Path("/dev/shm")
#: A version is spooled in RAM only while RAM has this many times its size
#: free, so the retained versions never squeeze the host or its cgroup.
_RAM_HEADROOM = 4


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
        complete = [v for v, m in self._manifests.items()
                    if len(self._landed.get(v, ())) == m.num_shards]
        return max(complete) if complete else None

    def _evict_old(self) -> None:
        if len(self._manifests) <= self.retain_last:
            return
        keep = set(sorted(self._manifests, reverse=True)[: self.retain_last])
        for version in list(self._manifests):
            if version not in keep:
                del self._manifests[version]
                self._landed.pop(version, None)
                # A download already streaming one of these files keeps its
                # open fd, so unlinking never cuts a transfer short.
                if (vdir := self._dirs.pop(version, None)) is not None:
                    shutil.rmtree(vdir, ignore_errors=True)

    def publish_manifest(self, version: int, manifest: CheckpointManifest) -> None:
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
                web.get("/shard/{version}/{idx}", _handle_get_shard),
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
    return web.json_response(relay.get_manifest(version).to_json())


async def _handle_get_shard(request: web.Request) -> web.Response:
    relay: RelayServer = request.app[_RELAY_KEY]
    version = int(request.match_info["version"])
    idx = int(request.match_info["idx"])
    path = relay.shard_path(version, idx)
    if path is None:
        return web.Response(status=404, text=f"no shard {idx} for version {version}")
    # sendfile straight from the spool: no multi-GB copy through Python.
    # Returned unprepared: aiohttp prepares it, and a FileResponse prepared
    # here first asserts on that second prepare. Download rates are logged by
    # the fetching RelayClient instead.
    return web.FileResponse(path)


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
    ):
        """``site_cache_dir``: a directory shared with the other fetchers on this
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

                manifest = await self._fetch_manifest(session, url, min_version)
                if manifest is None:
                    continue
                fetch = (self._fetch_shards_via_site if self._site_cache is not None
                         else self._fetch_shards)
                shard_bytes = await fetch(session, url, manifest)
                if shard_bytes is None:
                    continue
                try:
                    state_dict = reassemble_state_dict(shard_bytes, manifest)
                except ShardIntegrityError:
                    logger.warning(
                        "relay %s served a corrupted shard for version %d; "
                        "trying another relay",
                        url,
                        manifest.version,
                    )
                    self._record_failure(url)
                    continue
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
