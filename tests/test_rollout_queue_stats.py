# Copyright (c) Panocular AI.
#
# GET /stats on the rollout queue: the control plane polls it to size the
# generator pool. Needs only aiohttp -- no torch, vLLM or Monarch.

import asyncio

import aiohttp
from aiohttp.test_utils import TestServer

from panoengine.decentralized.rollout_queue import (
    VERSION_HEADER,
    WORKER_ID_HEADER,
    RolloutQueueServer,
)


def test_stats_counts_pushes_rejections_and_pops():
    async def scenario():
        server = RolloutQueueServer(maxsize=2)
        test_server = TestServer(server.app())
        await test_server.start_server()
        base = f"http://{test_server.host}:{test_server.port}"
        headers = {WORKER_ID_HEADER: "0", VERSION_HEADER: "1"}
        try:
            async with aiohttp.ClientSession() as session:
                async def stats():
                    async with session.get(f"{base}/stats") as resp:
                        assert resp.status == 200
                        return await resp.json()

                assert await stats() == {"qsize": 0, "maxsize": 2, "received": 0,
                                         "rejected": 0, "popped": 0}
                codes = []
                for _ in range(3):   # the third push finds the queue full
                    async with session.post(f"{base}/rollouts", data=b"x",
                                            headers=headers) as resp:
                        codes.append(resp.status)
                assert codes == [204, 204, 503]
                async with session.post(f"{base}/rollouts/pop") as resp:
                    assert resp.status == 200
                assert await stats() == {"qsize": 1, "maxsize": 2, "received": 2,
                                         "rejected": 1, "popped": 1}
        finally:
            await test_server.close()

    asyncio.run(scenario())
