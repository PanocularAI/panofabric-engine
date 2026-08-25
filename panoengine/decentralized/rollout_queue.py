# Copyright (c) Panocular AI.
#
# The rollout queue: a standalone CPU process buffering rollout batches
# between the generator workers (push) and the trainer replicas (pop), used
# by BOTH decoupled strategies:
#   - async_inference: one trainer pops (the queue used to be embedded in the
#     trainer process; standalone, a trainer stall can never back up worker
#     pushes and queue traffic never shares the trainer's event loop).
#   - heloco_async_inference: N trainers pop from the same queue (any trainer
#     consumes any worker's rollouts, at-most-once).
#
# Deliberately dumb: a bounded FIFO of pickled ``(worker_id, version,
# groups)`` batches. Full -> push rejected (503; the worker drops the batch
# -- the trainer's max_staleness bound is designed around lost rollouts).
# Empty -> pop returns 204 (the trainer polls). Runs next to the relay
# process (relay.py) but separately from it, so multi-GB
# checkpoint traffic and rollout traffic never queue behind each other.
#
# Run as:
#   python -m panoengine.decentralized.rollout_queue \
#     --port 8767

import argparse
import asyncio
import logging
import pickle

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


class RolloutQueueServer:
    """Bounded queue of ``(worker_id, version, rollout_groups)`` batches.

    Push (``POST /rollouts``, pickled tuple) and pop (``POST /rollouts/pop``)
    are both at-most-once: a popped batch is claimed and removed in one call
    -- a lost batch is cheaper than a trainer that stalls waiting on a wedged
    one, the same dropping philosophy the rest of this swarm uses. Safe under
    concurrent trainers despite no explicit lock: aiohttp's single-threaded
    event loop serializes handler bodies between awaits, and
    ``asyncio.Queue.get_nowait`` has no await in it, so two concurrent pops
    can't claim the same batch."""

    def __init__(self, maxsize: int = 256):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.num_received = 0
        self.num_rejected = 0
        self.num_popped = 0

    def qsize(self) -> int:
        return self.queue.qsize()

    def routes(self) -> list:
        return [
            web.post("/rollouts", self._handle_push),
            web.post("/rollouts/pop", self._handle_pop),
        ]

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.add_routes(self.routes())
        return app

    async def _handle_push(self, request: web.Request) -> web.Response:
        data = await request.read()
        try:
            worker_id, version, groups = pickle.loads(data)
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            return web.Response(status=400, text=f"malformed rollout payload: {exc}")
        try:
            self.queue.put_nowait((worker_id, version, groups))
        except asyncio.QueueFull:
            self.num_rejected += 1
            return web.Response(status=503, text="rollout queue full; consumer stalled")
        self.num_received += 1
        return web.Response(status=204)

    async def _handle_pop(self, request: web.Request) -> web.Response:
        del request
        try:
            batch = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return web.Response(status=204)
        self.num_popped += 1
        return web.Response(
            body=pickle.dumps(batch), content_type="application/octet-stream"
        )


class RolloutQueuePushClient:
    """The generator workers' side: one pickle+HTTP POST per batch.

    A rejection (503, queue full) or transport failure is logged and the
    batch is dropped rather than retried: blocking generation to retry a
    stuck queue defeats the point of decoupled generation, and the trainer's
    max_staleness bound already tolerates -- and is designed around -- losing
    some rollouts."""

    def __init__(self, queue_address: str, *, timeout_s: float = 30.0):
        if not queue_address.strip():
            raise ValueError(
                "queue_address is required (set $ASYNC_INFERENCE_ROLLOUT_QUEUE_ADDR)"
            )
        self.queue_address = queue_address.rstrip("/")
        self._timeout_s = timeout_s

    async def send(self, worker_id: int, version: int, groups: list) -> bool:
        """Returns True if the queue accepted the batch, False otherwise
        (rejected or unreachable) -- never raises on a transport failure."""
        payload = pickle.dumps((worker_id, version, groups))
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s)
            ) as session:
                async with session.post(
                    f"{self.queue_address}/rollouts", data=payload
                ) as resp:
                    if resp.status == 204:
                        return True
                    logger.warning(
                        "rollout push to %s rejected (status=%d)",
                        self.queue_address,
                        resp.status,
                    )
                    return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            # TimeoutError too: aiohttp raises plain asyncio.TimeoutError (not
            # a ClientError) on a slow queue; a push must never raise.
            logger.warning("rollout push to %s failed: %s", self.queue_address, exc)
            return False


class RolloutQueuePopClient:
    """The trainer replicas' side: pops one batch at a time.

    A non-blocking poll: :meth:`pop` returns ``None`` immediately if the
    queue is empty or unreachable, never raises (the pure-learner consumer
    then waits ``queue_poll_interval_s`` and retries). Multiple trainer
    replicas pop from the SAME queue concurrently -- an at-most-once claim
    per call, so no two trainers ever consume the same batch."""

    def __init__(self, queue_address: str, *, timeout_s: float = 30.0):
        if not queue_address.strip():
            raise ValueError("queue_address is required (set $ROLLOUT_QUEUE_ADDR)")
        self.queue_address = queue_address.rstrip("/")
        self._timeout_s = timeout_s

    async def pop(self):
        """Returns ``(worker_id, version, groups)`` or ``None`` (empty queue
        or unreachable/slow server)."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s)
            ) as session:
                async with session.post(f"{self.queue_address}/rollouts/pop") as resp:
                    if resp.status == 204:
                        return None
                    if resp.status != 200:
                        logger.warning(
                            "rollout pop from %s failed (status=%d)",
                            self.queue_address,
                            resp.status,
                        )
                        return None
                    data = await resp.read()
                    return pickle.loads(data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            # TimeoutError too: one slow response must not kill the consumer.
            logger.warning("rollout pop from %s failed: %s", self.queue_address, exc)
            return None


async def _serve(*, host: str, port: int, advertise_host: str, maxsize: int) -> None:
    server = RolloutQueueServer(maxsize=maxsize)
    runner = web.AppRunner(server.app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    # `host` is the bind interface (often "0.0.0.0"); remote trainers/workers
    # need a real, connectable hostname/IP, so this is printed separately.
    print(f"ROLLOUT_QUEUE_ADDR=http://{advertise_host}:{port}", flush=True)
    logger.info(
        "rollout queue listening on %s:%d, advertised as %s (maxsize=%d)",
        host,
        port,
        advertise_host,
        maxsize,
    )
    await asyncio.Event().wait()  # serve until killed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="standalone rollout queue")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--advertise_host",
        type=str,
        default=None,
        help="hostname/IP trainers and workers use to reach this process "
        "(default: $TORCHFT_PS_ADVERTISE_HOST if set, else this machine's "
        "hostname -- NOT --host, which is only the local bind interface)",
    )
    parser.add_argument("--maxsize", type=int, default=256)
    args = parser.parse_args()

    from torchft.parameter_server import _resolve_advertise_host

    try:
        asyncio.run(
            _serve(
                host=args.host,
                port=args.port,
                advertise_host=_resolve_advertise_host(args.advertise_host),
                maxsize=args.maxsize,
            )
        )
    except KeyboardInterrupt:
        logger.info("rollout queue shutting down")


if __name__ == "__main__":
    main()
