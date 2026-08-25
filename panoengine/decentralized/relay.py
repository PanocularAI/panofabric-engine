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
#     full (e.g. SnapshotPolicyTrainer.get_full_state_dict_cpu).
#   - RelayServer: a CDN-like CPU relay node sitting between one trainer and
#     many inference workers, so the trainer never serves every worker's
#     weight pulls itself and workers never need the trainer's address
#     directly. Stores shards + a per-version manifest and keeps only the
#     last ``retain_last`` versions (disk/memory limits, matching the paper).
#   - RelayClient: the publisher/fetcher counterpart. Tracks a per-relay
#     success_rate/bandwidth EMA and picks a relay *probabilistically
#     weighted by success_rate x bandwidth* rather than always the fastest --
#     the paper's exact rule, which keeps one flaky-but-occasionally-fast
#     relay from permanently starving the others.
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
import random
import time
from dataclasses import dataclass, field

import aiohttp
import torch
from aiohttp import web

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.3
_RELAY_KEY = web.AppKey("relay")


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


class RelayServer:
    """In-memory shard/manifest store for one relay node.

    Not thread-safe by locking (aiohttp's default single-threaded event loop
    serializes handler bodies between awaits, which is enough here since
    nothing awaits mid-mutation) but IS safe under normal aiohttp concurrency
    for that reason.
    """

    def __init__(self, retain_last: int = 5):
        self.retain_last = retain_last
        self._shards: dict[int, dict[int, bytes]] = {}
        self._manifests: dict[int, CheckpointManifest] = {}

    def latest_version(self) -> int | None:
        return max(self._manifests) if self._manifests else None

    def _evict_old(self) -> None:
        if len(self._manifests) <= self.retain_last:
            return
        keep = set(sorted(self._manifests, reverse=True)[: self.retain_last])
        for version in list(self._manifests):
            if version not in keep:
                del self._manifests[version]
                self._shards.pop(version, None)

    def publish_manifest(self, version: int, manifest: CheckpointManifest) -> None:
        self._manifests[version] = manifest
        self._shards.setdefault(version, {})
        self._evict_old()

    def publish_shard(self, version: int, idx: int, data: bytes) -> None:
        if version not in self._manifests:
            raise KeyError(f"no manifest published for version {version}")
        self._shards[version][idx] = data

    def get_manifest(self, version: int) -> CheckpointManifest | None:
        return self._manifests.get(version)

    def get_shard(self, version: int, idx: int) -> bytes | None:
        return self._shards.get(version, {}).get(idx)

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
    data = await request.read()
    try:
        relay.publish_shard(version, idx, data)
    except KeyError as exc:
        return web.Response(status=404, text=str(exc))
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
    data = relay.get_shard(version, idx)
    if data is None:
        return web.Response(status=404, text=f"no shard {idx} for version {version}")
    return web.Response(body=data, content_type="application/octet-stream")


async def run_relay_server(
    host: str = "0.0.0.0", port: int = 8765, retain_last: int = 5
):
    """Start a relay server; returns the ``web.AppRunner`` (caller keeps it
    alive and calls ``.cleanup()`` to stop)."""
    relay = RelayServer(retain_last=retain_last)
    runner = web.AppRunner(relay.app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    logger.info(
        "relay server listening on %s:%d (retain_last=%d)", host, port, retain_last
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
    ):
        if not relay_urls:
            raise ValueError("relay_urls must be non-empty")
        self.relay_urls = list(relay_urls)
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
        self._record_success(url, total_bytes, time.monotonic() - t0)
        return shard_bytes

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
                shard_bytes = await self._fetch_shards(session, url, manifest)
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
# Standalone relay-node entrypoint.
# --------------------------------------------------------------------- #


async def _serve(host: str, port: int, retain_last: int) -> None:
    runner = await run_relay_server(host=host, port=port, retain_last=retain_last)
    print(f"ASYNC_INFERENCE_RELAY_ADDR=http://{host}:{port}", flush=True)
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
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.host, args.port, args.retain_last))
    except KeyboardInterrupt:
        logger.info("relay server shutting down")


if __name__ == "__main__":
    main()
