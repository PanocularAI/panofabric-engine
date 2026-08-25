# Copyright (c) Panocular AI.
#
# Distributed-inference gateway: an OpenAI-compatible reverse proxy in
# front of a set of vLLM replica islands. One public port; requests
# are routed to the healthy target with the fewest in-flight requests
# (least-loaded — decode time varies wildly with sequence length, so
# round-robin skews badly). Streaming responses pass through chunk-by-chunk.
#
# Optional bearer-key auth (`--api-keys k1,k2`): when set, every /v1 request
# must carry `Authorization: Bearer <key>`. With `--upstream-key` the backends
# are dialed with THAT bearer instead of the client's (the control plane starts
# them with VLLM_API_KEY set to it), so a replica whose port is publicly
# reachable — as it is on any cloud launch — cannot be used to skip the check
# above. Health: each target's /health is
# polled in the background; unhealthy targets are skipped (and everything is
# eligible again if every target looks down, so a cold-starting deployment
# still converges instead of 503ing forever).
#
# Launched by the control plane as a CoordinatorPlan (same pattern as the RL
# relay); runs standalone too:
#   python -m panoengine.serve.gateway --port 8800 \
#       --targets http://127.0.0.1:8801,http://127.0.0.1:8802

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
                "keep-alive", "te", "upgrade", "proxy-authorization"}

# Max silence from a backend mid-response before the proxy gives up. Generous:
# a spliced pipeline's inter-token gap is seconds at WAN latency, and the first
# token can trail a long prefill.
_SOCK_READ_S = 300.0


class Gateway:
    """Routing/health state for one deployment. Counters are mutated only from the
    single aiohttp event loop (no awaits between read and write), so no locks."""

    def __init__(self, targets: list[str], *, api_keys: list[str] | None = None,
                 admin_token: str = "", upstream_key: str = "",
                 health_interval_s: float = 5.0):
        if not targets:
            raise ValueError("gateway needs at least one target")
        self.api_keys = set(api_keys or [])
        self.admin_token = admin_token
        self.upstream_key = upstream_key
        self.health_interval_s = health_interval_s
        self.num_requests = 0
        self.num_rejected = 0
        self._session: aiohttp.ClientSession | None = None
        self.targets: list[str] = []
        self.inflight: dict[str, int] = {}
        self.healthy: dict[str, bool] = {}
        self.retarget(targets)

    def retarget(self, targets: list[str]) -> None:
        """Atomically replace the backend set — the control plane pushes the
        REAL island addresses here once they're provisioned (plan-time
        targets are loopback placeholders on cloud launches) and re-points
        after a recovery moves an island to a new node. State for kept
        targets survives; new ones start unproven (health loop promotes)."""
        clean = [t.rstrip("/") for t in targets]
        if not clean:
            raise ValueError("gateway needs at least one target")
        self.inflight = {t: self.inflight.get(t, 0) for t in clean}
        self.healthy = {t: self.healthy.get(t, False) for t in clean}
        self.targets = clean

    def _http(self) -> aiohttp.ClientSession:
        """One long-lived session (connection pool / keep-alive to the
        backends); a per-request session costs a TCP handshake per request."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ---------------------------- routing ---------------------------- #

    def pick(self) -> str:
        """Least-in-flight among healthy targets; if none are healthy (cold
        start, or every target flapped) fall back to all targets so we
        forward and let the real error surface instead of synthesizing 503s."""
        pool = [t for t in self.targets if self.healthy[t]] or self.targets
        return min(pool, key=lambda t: self.inflight[t])

    def _authorized(self, request: web.Request) -> bool:
        if not self.api_keys:
            return True
        auth = request.headers.get("Authorization", "")
        return auth.startswith("Bearer ") and auth[7:] in self.api_keys

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            self.num_rejected += 1
            return web.json_response({"error": "invalid or missing API key"},
                                     status=401)
        target = self.pick()
        url = f"{target}{request.path_qs}"
        # The client's own bearer stops HERE (it was just checked against
        # api_keys); with an upstream key the backend gets the internal one it
        # was launched with instead — that key is what makes a replica
        # reachable only THROUGH this gateway, since on a cloud launch its own
        # port is public. Drop every casing of the inbound header, or aiohttp
        # would forward the client's alongside ours.
        drop = _HOP_HEADERS | ({"authorization"} if self.upstream_key else set())
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in drop}
        if self.upstream_key:
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        body = await request.read()
        self.inflight[target] = self.inflight.get(target, 0) + 1
        self.num_requests += 1
        resp: web.StreamResponse | None = None
        try:
            async with self._http().request(
                request.method, url, headers=headers, data=body,
                # sock_read bounds a backend that accepts and then goes
                # silent (a wedged engine — the splice's own failure mode):
                # without it the request and its inflight slot pin forever.
                timeout=aiohttp.ClientTimeout(total=None, connect=10,
                                              sock_read=_SOCK_READ_S),
            ) as upstream:
                out_headers = {k: v for k, v in upstream.headers.items()
                               if k.lower() not in _HOP_HEADERS
                               and k.lower() != "content-encoding"}
                resp = web.StreamResponse(status=upstream.status,
                                          headers=out_headers)
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.healthy[target] = False   # re-probed by the health loop
            logger.warning("target %s failed mid-request: %s", target, exc)
            if resp is not None:
                # response already started (streaming): a 502 body can no
                # longer be sent — close the stream and let the client see a
                # truncated response rather than raising inside aiohttp.
                return resp
            return web.json_response(
                {"error": f"upstream {target} unavailable"}, status=502)
        finally:
            # A retarget mid-request drops this target from the counters
            # (the scheduler re-points the fleet after every recovery, which
            # is exactly when requests to a dead island are in flight), so
            # decrement defensively and never resurrect a removed key.
            if target in self.inflight:
                self.inflight[target] = max(0, self.inflight[target] - 1)

    async def handle_health(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({
            "targets": {t: {"healthy": self.healthy[t],
                            "inflight": self.inflight[t]}
                        for t in self.targets},
            "requests": self.num_requests,
            "rejected": self.num_rejected,
        })

    async def handle_retarget(self, request: web.Request) -> web.Response:
        """Control-plane push of the live backend set (requires the launch's
        admin token — the plan-time targets are placeholders on cloud
        launches, where island addresses exist only after provisioning)."""
        auth = request.headers.get("Authorization", "")
        if not self.admin_token or auth != f"Bearer {self.admin_token}":
            return web.json_response({"error": "admin token required"},
                                     status=401)
        body = await request.json()
        try:
            self.retarget(list(body["targets"]))
        except (KeyError, TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        logger.info("retargeted -> %s", ", ".join(self.targets))
        return web.json_response({"targets": self.targets})

    # ---------------------------- health ----------------------------- #

    async def probe_once(self) -> None:
        for t in list(self.targets):
            try:
                async with self._http().get(
                    f"{t}/health", timeout=aiohttp.ClientTimeout(total=4)
                ) as resp:
                    up = resp.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError):
                up = False
            # a retarget during the await may have dropped this target:
            # don't resurrect it, and don't abort the rest of the pass
            if t not in self.healthy:
                continue
            if up != self.healthy[t]:
                logger.info("target %s -> %s", t, "up" if up else "down")
            self.healthy[t] = up

    async def health_loop(self) -> None:
        while True:
            try:
                await self.probe_once()
            except Exception:   # never let the prober die
                logger.exception("health probe failed")
            await asyncio.sleep(self.health_interval_s)

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes([
            web.get("/gateway/health", self.handle_health),
            web.post("/gateway/targets", self.handle_retarget),
            web.route("*", "/{path:.*}", self.handle),
        ])

        async def _start_prober(app_):
            app_["prober"] = asyncio.create_task(self.health_loop())

        async def _stop_prober(app_):
            app_["prober"].cancel()
            if self._session is not None and not self._session.closed:
                await self._session.close()

        app.on_startup.append(_start_prober)
        app.on_cleanup.append(_stop_prober)
        return app


async def _serve(host: str, port: int, gw: Gateway) -> None:
    runner = web.AppRunner(gw.app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    print(f"GATEWAY_ADDR=http://{host}:{port}", flush=True)
    logger.info("gateway on %s:%d -> %s", host, port, ", ".join(gw.targets))
    await asyncio.Event().wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="distributed-inference gateway")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--targets", required=True,
                   help="comma-separated backend base URLs")
    p.add_argument("--api-keys", default="",
                   help="comma-separated bearer keys; empty = no auth")
    p.add_argument("--admin-token", default="",
                   help="bearer for POST /gateway/targets (control-plane "
                        "retargeting); empty disables the route")
    p.add_argument("--upstream-key", default="",
                   help="bearer sent to the BACKENDS in place of the client's "
                        "(they are launched with VLLM_API_KEY set to it), so a "
                        "publicly-reachable replica port cannot bypass this "
                        "gateway's api-key check")
    args = p.parse_args()
    # Secrets come from the environment (argv is visible in `ps` and lands in
    # cloud job scripts/logs); the flags stay for standalone use.
    api_keys = os.environ.get("PANOFABRIC_GATEWAY_API_KEYS", args.api_keys)
    admin_token = os.environ.get("PANOFABRIC_GATEWAY_ADMIN_TOKEN",
                                 args.admin_token)
    gw = Gateway(
        [t for t in args.targets.split(",") if t.strip()],
        api_keys=[k for k in api_keys.split(",") if k.strip()],
        admin_token=admin_token,
        upstream_key=os.environ.get("PANOFABRIC_GATEWAY_UPSTREAM_KEY",
                                    args.upstream_key),
    )
    started = time.monotonic()
    try:
        asyncio.run(_serve(args.host, args.port, gw))
    except KeyboardInterrupt:
        logger.info("gateway shutting down after %.0fs",
                    time.monotonic() - started)


if __name__ == "__main__":
    main()
