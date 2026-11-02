"""Lossless checkpoint deltas on the relay (relay.encode_delta/decode_delta, GET /delta).

The contract that matters for training: a worker that fetched a checkpoint as a
delta holds EXACTLY the weights it would have downloaded whole -- bit for bit,
including -0.0 and NaN payloads -- and anything short of that (a corrupt delta, a
base the relay no longer has, a base that is different content under the same
version number) falls back to the whole download instead of handing the worker
wrong weights. Every comparison here is on raw bits, never a tolerance.
"""
import asyncio
import json
import threading

import pytest
import torch
from aiohttp.test_utils import TestServer

import panoengine.decentralized.relay as relay_mod
from panoengine.decentralized.relay import (
    RelayClient,
    RelayServer,
    ShardIntegrityError,
    build_manifest,
    decode_delta,
    encode_delta,
    shard_state_dict,
)


def _bits(t):
    return relay_mod._flat_bits(t)


def _assert_bit_identical(got: dict, want: dict):
    assert got.keys() == want.keys()
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert got[name].shape == want[name].shape, name
        assert torch.equal(_bits(got[name]), _bits(want[name])), f"{name} differs in bits"


class _Trainer:
    """An fp32 master copy published as bf16 -- how the RL trainer's weights
    reach the relay -- stepped at an RL-sized learning rate."""

    def __init__(self, sizes=((512, 256), (256,), (1024, 64)), lr=2e-6, seed=0):
        self.gen = torch.Generator().manual_seed(seed)
        self.master = {f"p{i}": torch.randn(s, generator=self.gen) * 0.02
                       for i, s in enumerate(sizes)}
        self.lr = lr

    def step(self):
        for p in self.master.values():
            # Adam-like: the update's magnitude is ~lr regardless of the gradient's.
            p.add_(torch.randn(p.shape, generator=self.gen).sign() * self.lr)

    def published(self):
        return {n: p.to(torch.bfloat16) for n, p in self.master.items()}


# --------------------------------------------------------------------- #
# The codec.
# --------------------------------------------------------------------- #


def test_delta_round_trip_is_bit_exact_for_every_dtype_and_edge_case():
    special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-40])
    nan_payload = torch.tensor([0x7FC1, 0x7FFF, -1], dtype=torch.int16).view(torch.bfloat16)
    base = {
        "bf16": torch.randn(64, 32).to(torch.bfloat16),
        "fp32": torch.randn(17),
        "int64": torch.arange(10),
        "bool": torch.tensor([True, False, True]),
        "scalar": torch.tensor(3.5),
        "special": special.clone(),
        "nan_payload": torch.zeros(3, dtype=torch.bfloat16),
        "reshaped": torch.randn(4, 6),
        "dropped": torch.randn(5),  # not in target: must not appear in the result
    }
    target = {k: v.clone() for k, v in base.items() if k != "dropped"}
    target["bf16"][::7] += 0.25
    target["fp32"][3] = -0.0
    target["int64"][-1] = 2**40
    target["bool"][1] = True
    target["scalar"] = torch.tensor(-1.25)
    target["special"] = special.flip(0)
    target["nan_payload"] = nan_payload
    target["reshaped"] = torch.randn(6, 4)  # shape changed: shipped whole
    target["new"] = torch.randn(2, 2, dtype=torch.float16)  # base lacks it
    target["empty"] = torch.empty(0, 3)
    base_snapshot = {k: v.clone() for k, v in base.items()}

    blob = encode_delta(base, target, base_version=1, target_version=2)
    restored = decode_delta(blob, base, base_version=1, target_version=2)

    _assert_bit_identical(restored, target)
    _assert_bit_identical(base, base_snapshot)  # decode never writes into the base


def test_rl_sized_update_ships_as_a_small_exact_delta():
    """The reason this exists: at RL learning rates most bf16 weights keep their
    exact bits between versions, so the delta is a few percent of the checkpoint
    -- and it must still reconstruct exactly."""
    trainer = _Trainer()
    v1 = trainer.published()
    for _ in range(3):
        trainer.step()
    v2 = trainer.published()
    changed = sum(int((_bits(v1[k]) != _bits(v2[k])).sum()) for k in v1)
    total = sum(t.numel() for t in v1.values())
    assert 0 < changed < 0.2 * total  # a real but sparse update

    blob = encode_delta(v1, v2, base_version=1, target_version=4)
    full = sum(t.numel() * t.element_size() for t in v2.values())
    assert len(blob) < 0.25 * full, f"delta {len(blob)} B vs checkpoint {full} B"
    _assert_bit_identical(decode_delta(blob, v1, base_version=1, target_version=4), v2)


def test_tensors_split_into_many_chunks_still_round_trip_exactly(monkeypatch):
    """Big tensors are cut into chunks (one frame + digest each) so they spread
    over every core; with a tiny chunk size every tensor here spans many."""
    monkeypatch.setattr(relay_mod, "_CHUNK_BYTES", 64)
    trainer = _Trainer()
    v1 = trainer.published()
    trainer.step()
    v2 = trainer.published()
    v2["odd"] = torch.randn(37)  # numel not a multiple of the chunk
    blob = encode_delta(v1, v2, base_version=1, target_version=2)
    footer_len = int.from_bytes(blob[-8:], "little")
    meta = json.loads(blob[-8 - footer_len:-8])
    assert all(len(e["frames"]) > 1 for e in meta["entries"])
    _assert_bit_identical(decode_delta(blob, v1, base_version=1, target_version=2), v2)

    # Corrupt a chunk in the MIDDLE of a tensor: the per-chunk digest catches it.
    entry = meta["entries"][0]
    offset = len(relay_mod._DELTA_MAGIC) + sum(entry["frames"][: len(entry["frames"]) // 2])
    corrupt = bytearray(blob)
    corrupt[offset + 6] ^= 0x01
    with pytest.raises(ShardIntegrityError):
        decode_delta(bytes(corrupt), v1, base_version=1, target_version=2)


def test_decode_refuses_anything_but_the_exact_target():
    trainer = _Trainer()
    v1 = trainer.published()
    trainer.step()
    v2 = trainer.published()
    blob = encode_delta(v1, v2, base_version=1, target_version=2)

    corrupt = bytearray(blob)
    corrupt[len(relay_mod._DELTA_MAGIC) + 20] ^= 0x40  # inside the first frame
    with pytest.raises(ShardIntegrityError):
        decode_delta(bytes(corrupt), v1, base_version=1, target_version=2)

    wrong_base = {k: v.clone() for k, v in v1.items()}
    wrong_base["p0"][0, 0] += 1.0
    with pytest.raises(ShardIntegrityError, match="wrong bytes"):
        decode_delta(blob, wrong_base, base_version=1, target_version=2)

    with pytest.raises(ShardIntegrityError):
        decode_delta(blob, {k: v for k, v in v1.items() if k != "p1"},
                     base_version=1, target_version=2)
    with pytest.raises(ShardIntegrityError):
        decode_delta(blob[:-30], v1, base_version=1, target_version=2)
    with pytest.raises(ShardIntegrityError):
        decode_delta(blob, v1, base_version=1, target_version=3)
    with pytest.raises(ShardIntegrityError):
        decode_delta(b"not a delta at all", v1, base_version=1, target_version=2)


# --------------------------------------------------------------------- #
# Relay + client over HTTP.
# --------------------------------------------------------------------- #


async def _start_relay(retain_last=5):
    relay = RelayServer(retain_last=retain_last)
    server = TestServer(relay.app())
    await server.start_server()
    return relay, server, str(server.make_url("")).rstrip("/")


async def _publish(url, version, sd, num_shards=2):
    shards = shard_state_dict(sd, num_shards)
    await RelayClient([url]).publish(version, shards, build_manifest(version, shards))


def test_worker_fetches_newer_versions_as_exact_deltas():
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            client = RelayClient([url])
            published = {1: trainer.published()}
            await _publish(url, 1, published[1])
            version, sd = await client.fetch_latest()
            assert version == 1 and client.last_fetch["kind"] == "full"
            _assert_bit_identical(sd, published[1])
            sd.clear()  # what the worker does after loading; the base must survive

            for version in (2, 3, 4):
                trainer.step()
                published[version] = trainer.published()
                await _publish(url, version, published[version])
                got_version, sd = await client.fetch_latest(min_version=version - 1)
                assert got_version == version
                assert client.last_fetch["kind"] == "delta"
                full = sum(relay.get_manifest(version).shard_sizes)
                assert client.last_fetch["bytes"] < 0.25 * full
                _assert_bit_identical(sd, published[version])
                sd.clear()
            # Every delta was pre-built while its version uploaded, against
            # exactly the version the worker held -- one per version, none
            # wasted on versions it had moved off, none built on request.
            assert relay.num_delta_builds == 0 and relay.num_delta_prebuilds == 3

            # A second worker at the same base reuses the cached delta.
            twin = RelayClient([url])
            twin._base = (3, relay_mod.manifest_tag(relay.get_manifest(3)), published[3])
            _, sd = await twin.fetch_latest(min_version=3)
            assert twin.last_fetch["kind"] == "delta" and relay.num_delta_builds == 0
            _assert_bit_identical(sd, published[4])
        finally:
            await server.close()

    asyncio.run(scenario())


def test_skipping_versions_still_deltas_against_what_the_worker_holds():
    """A worker that fell behind (generation outlasting several publishes) asks
    for latest-vs-what-it-holds, not a chain."""
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            for version in (2, 3, 4, 5):
                trainer.step()
                await _publish(url, version, trainer.published())
            version, sd = await client.fetch_latest(min_version=1)
            assert version == 5 and client.last_fetch["kind"] == "delta"
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_falls_back_to_whole_download_when_base_was_evicted(monkeypatch):
    monkeypatch.setattr(relay_mod, "_IN_USE_S", 0.0)  # count-based eviction only

    async def scenario():
        relay, server, url = await _start_relay(retain_last=2)
        try:
            trainer = _Trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            for version in (2, 3):
                trainer.step()
                await _publish(url, version, trainer.published())
            assert relay.get_manifest(1) is None  # the worker's base is gone
            version, sd = await client.fetch_latest(min_version=1)
            assert version == 3 and client.last_fetch["kind"] == "full"
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_falls_back_when_relay_has_different_content_under_the_base_version():
    """A relay that outlived a run: same version number, different weights.
    The base tag must catch it (409) before any delta is applied."""
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer(seed=1)
            client = RelayClient([url])
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            other_run = _Trainer(seed=2).published()
            client._base = (1, "0" * 16, other_run)  # held v1 is another run's
            trainer.step()
            await _publish(url, 2, trainer.published())
            version, sd = await client.fetch_latest(min_version=1)
            assert version == 2 and client.last_fetch["kind"] == "full"
            assert relay.num_delta_builds == 0
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_falls_back_when_the_served_delta_is_corrupt():
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            trainer.step()
            await _publish(url, 2, trainer.published())
            path = await relay.ensure_delta(1, 2)
            data = bytearray(path.read_bytes())
            data[len(relay_mod._DELTA_MAGIC) + 20] ^= 0x40
            path.write_bytes(bytes(data))

            version, sd = await client.fetch_latest(min_version=1)
            assert version == 2 and client.last_fetch["kind"] == "full"
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_falls_back_when_applying_the_delta_raises(monkeypatch):
    """Not only a failed check: anything applying a delta raises (a base on the
    wrong device, say) costs only the optimization, never the fetch."""
    def broken(*args, **kwargs):
        raise RuntimeError("Expected all tensors to be on the same device")

    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            trainer.step()
            await _publish(url, 2, trainer.published())
            monkeypatch.setattr(relay_mod, "decode_delta", broken)
            version, sd = await client.fetch_latest(min_version=1)
            assert version == 2 and client.last_fetch["kind"] == "full"
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_republished_versions_start_over_as_the_new_content():
    """A new run on a standing relay republishes v1 and v2 as other weights. The
    relay must drop the old content's shards, chunk digests and deltas -- served
    any of them, a fetcher of the new run would fail verification on every
    delta and silently download whole."""
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            old = _Trainer(seed=1)
            old_client = RelayClient([url])
            await _publish(url, 1, old.published())
            await old_client.fetch_latest()
            old.step()
            await _publish(url, 2, old.published())
            await old_client.fetch_latest(min_version=1)
            assert old_client.last_fetch["kind"] == "delta"  # old v1->v2 cached

            new = _Trainer(seed=2)
            new_v1 = new.published()
            await _publish(url, 1, new_v1)
            client = RelayClient([url])
            client._base = (1, relay_mod.manifest_tag(relay.get_manifest(1)), new_v1)
            new.step()
            await _publish(url, 2, new.published())
            version, sd = await client.fetch_latest(min_version=1)
            assert version == 2 and client.last_fetch["kind"] == "delta"
            _assert_bit_identical(sd, new.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_use_deltas_false_keeps_the_old_behaviour():
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            client = RelayClient([url], use_deltas=False)
            await _publish(url, 1, trainer.published())
            await client.fetch_latest()
            assert client._base is None  # no extra checkpoint held in RAM
            trainer.step()
            await _publish(url, 2, trainer.published())
            _, sd = await client.fetch_latest(min_version=1)
            assert client.last_fetch["kind"] == "full" and relay.num_delta_builds == 0
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# Pre-building: deltas cut while the version uploads, before anyone asks.
# --------------------------------------------------------------------- #


def _four_shard_trainer():
    return _Trainer(sizes=((512, 256), (256, 256), (1024, 64), (128, 512)))


def test_prebuild_is_pipelined_with_the_upload(monkeypatch):
    """The relay knows a fetcher holds v1, so v2's delta against it is cut
    shard by shard AS v2 UPLOADS -- each shard before the next one lands --
    and the fetch is served from it, with nothing built on request."""
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _four_shard_trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published(), num_shards=4)
            await client.fetch_latest()
            trainer.step()
            shards = shard_state_dict(trainer.published(), 4)

            cuts = []
            real_cut = relay._cut_shard

            def recording(target, idx, writers):
                cuts.append((idx, set(relay._landed[target])))
                return real_cut(target, idx, writers)

            monkeypatch.setattr(relay, "_cut_shard", recording)
            relay.publish_manifest(2, build_manifest(2, shards))
            for idx, data in enumerate(shards):
                relay.publish_shard(2, idx, data)
                for _ in range(500):  # give the pre-build its turn
                    if len(cuts) > idx:
                        break
                    await asyncio.sleep(0.01)
            assert [idx for idx, _ in cuts] == [0, 1, 2, 3]
            assert all(landed == set(range(idx + 1)) for idx, landed in cuts), \
                "a shard must be cut before the next one lands"

            await relay.ensure_delta(1, 2)
            assert relay.num_delta_prebuilds == 1 and relay.num_delta_builds == 0
            _, sd = await client.fetch_latest(min_version=1)
            assert client.last_fetch["kind"] == "delta" and relay.num_delta_builds == 0
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_prebuild_gives_up_on_a_stalled_upload_and_builds_on_request(monkeypatch):
    """A publisher that dies mid-upload must not wedge the pre-build: it gives
    up, and once the version does complete a fetch still gets an exact delta,
    built on request."""
    monkeypatch.setattr(relay_mod, "_PREBUILD_STALL_S", 0.2)

    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _four_shard_trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published(), num_shards=4)
            await client.fetch_latest()
            trainer.step()
            shards = shard_state_dict(trainer.published(), 4)
            relay.publish_manifest(2, build_manifest(2, shards))
            relay.publish_shard(2, 0, shards[0])
            await asyncio.sleep(0.6)  # the pre-build times out waiting for shard 1
            assert relay._prebuilds == {} and relay.num_delta_prebuilds == 0
            for idx in (1, 2, 3):
                relay.publish_shard(2, idx, shards[idx])

            _, sd = await client.fetch_latest(min_version=1)
            assert client.last_fetch["kind"] == "delta"
            assert relay.num_delta_builds == 1 and relay.num_delta_prebuilds == 0
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())


def test_evicting_a_version_mid_prebuild_stops_it_quietly(monkeypatch, caplog):
    """Eviction stops a pre-build at its next await rather than cancelling it:
    a cancel aborted the delta writers while a _cut_shard thread could still be
    writing to them. Stopping is routine, so it logs no warning."""
    monkeypatch.setattr(relay_mod, "_IN_USE_S", 3600.0)  # the touched base v1 stays

    async def scenario():
        relay, server, url = await _start_relay(retain_last=2)
        try:
            trainer = _four_shard_trainer()
            client = RelayClient([url])
            await _publish(url, 1, trainer.published(), num_shards=4)
            await client.fetch_latest()  # the relay now pre-builds against v1

            entered, release = threading.Event(), threading.Event()
            real_cut = relay._cut_shard

            def held_cut(*args):  # a cut still running on its thread
                entered.set()
                release.wait(5)
                return real_cut(*args)

            monkeypatch.setattr(relay, "_cut_shard", held_cut)
            trainer.step()
            shards = shard_state_dict(trainer.published(), 4)
            relay.publish_manifest(2, build_manifest(2, shards))
            relay.publish_shard(2, 0, shards[0])
            build = relay._prebuilds[2]
            assert await asyncio.to_thread(entered.wait, 5)
            for version in (3, 4):  # v2 never completes; v4 evicts it mid-cut
                trainer.step()
                later = shard_state_dict(trainer.published(), 4)
                relay.publish_manifest(version, build_manifest(version, later))
                for idx, data in enumerate(later):
                    relay.publish_shard(version, idx, data)
            assert relay.get_manifest(2) is None and relay.get_manifest(1) is not None
            release.set()
            await asyncio.wait_for(build.task, 5)
            assert not build.task.cancelled() and 2 not in relay._prebuilds
            assert all(f.done() and f.exception() for f in build.ready.values())
            assert not [r for r in caplog.records if "pre-build of v2" in r.getMessage()]
        finally:
            await server.close()

    asyncio.run(scenario())


def test_no_prebuild_until_some_fetcher_holds_a_version():
    """Nothing to cut against before the first download: the first publish
    builds nothing, and the first fetch is a whole download."""
    async def scenario():
        relay, server, url = await _start_relay()
        try:
            trainer = _Trainer()
            await _publish(url, 1, trainer.published())
            trainer.step()
            await _publish(url, 2, trainer.published())
            await asyncio.sleep(0.1)
            assert relay.num_delta_prebuilds == 0
            client = RelayClient([url])
            _, sd = await client.fetch_latest()
            assert client.last_fetch["kind"] == "full"
            _assert_bit_identical(sd, trainer.published())
        finally:
            await server.close()

    asyncio.run(scenario())
