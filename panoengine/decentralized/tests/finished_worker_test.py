"""A worker that COMPLETED its steps must not look like a worker that was LOST.

Both simply stop heartbeating, so a cohort gate counting only live workers blocks on a
peer that already did its job. Live on 2026-08-30 (run 9a64a0588dc3): the H100 island
finished 300 steps while the slower L40S island still had its final sync boundary to
cross, and that boundary died with "waited 600s for 2 workers ... only 1 present".
It bites precisely at a large sync interval, where compute dominates and unequal
islands finish far apart -- i.e. every realistic operating point.
"""
import json
import urllib.request
from unittest import TestCase

import torch.optim as optim

from panoengine.decentralized.async_diloco import AsyncDiLoCoServer
from panoengine.decentralized.tests.async_diloco_test import _make_model


def _base(server) -> str:
    return server.address().rsplit("/", 1)[0]      # address() ends in /sync


def _status(server) -> dict:
    url = _base(server) + "/status"
    with urllib.request.urlopen(url, timeout=10.0) as resp:
        return json.loads(resp.read())


def _hit(server, path: str, worker_id: str) -> None:
    url = f"{_base(server)}{path}?worker_id={worker_id}"
    with urllib.request.urlopen(url, timeout=10.0):
        pass


class TestFinishedWorkerAccounting(TestCase):
    def setUp(self) -> None:
        model = _make_model()
        self.server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1),
                                        port=0)

    def test_done_moves_a_worker_out_of_active_without_losing_it(self) -> None:
        _hit(self.server, "/heartbeat", "a")
        _hit(self.server, "/heartbeat", "b")
        self.assertEqual(_status(self.server)["worker_count"], 2)

        _hit(self.server, "/done", "a")
        s = _status(self.server)
        self.assertEqual(s["worker_count"], 1, "a is no longer training")
        self.assertEqual(s["finished_count"], 1, "but a is accounted for, not lost")

    def test_cohort_total_survives_a_peer_finishing(self) -> None:
        """What the gate actually checks: active + finished still meets min_replicas=2,
        so the island still training crosses its last boundary instead of timing out."""
        _hit(self.server, "/heartbeat", "fast")
        _hit(self.server, "/heartbeat", "slow")
        _hit(self.server, "/done", "fast")
        s = _status(self.server)
        self.assertGreaterEqual(s["worker_count"] + s["finished_count"], 2)

    def test_a_lost_worker_is_still_lost(self) -> None:
        """The gate must keep protecting the real case: a peer that crashed never
        announces /done, so the cohort total drops and the gate blocks."""
        _hit(self.server, "/heartbeat", "a")
        _hit(self.server, "/heartbeat", "b")
        self.server._heartbeats.pop("b")          # crashed: heartbeat just stops
        s = _status(self.server)
        self.assertEqual(s["worker_count"] + s["finished_count"], 1)

    def test_done_is_idempotent(self) -> None:
        _hit(self.server, "/heartbeat", "a")
        _hit(self.server, "/done", "a")
        _hit(self.server, "/done", "a")
        self.assertEqual(_status(self.server)["finished_count"], 1)

    def test_done_without_worker_id_is_rejected(self) -> None:
        url = _base(self.server) + "/done"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url, timeout=10.0)
        self.assertEqual(ctx.exception.code, 400)
