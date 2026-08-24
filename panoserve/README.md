# panoserve

The serving plane for [PanoFabric](https://github.com/PanocularAI/panofabric):
distributed inference that serves one model spliced across multiple GPU
islands — potentially in different datacenters — behind a single
OpenAI-compatible endpoint.

Each module is a `python -m` entrypoint launched by the PanoFabric control
plane:

| Module | Role |
| --- | --- |
| `panoserve.engine_stage` | One stage of a spliced pipeline: an independent vLLM engine holding a layer range, chained to its neighbors by the stage transport. Stage 0 drives generation and exposes the OpenAI server plus `/admin/reload` weight hot-swap. |
| `panoserve.gateway` | Replica-fleet gateway: one OpenAI endpoint fanned out least-in-flight over N replica servers, with bearer-key auth and an admin plane for membership. |
| `panoserve.sharder` | Splits an HF safetensors checkpoint into per-stage layer-range shards. |
| `panoserve.weights` | Verified fetch-then-exec: downloads a sha256-manifest-checked weight set, then execs the real server argv. Stdlib-only. |
| `panoserve.stage_transport` | The tensor wire protocol between stages: framed dial/listen links plus a `LinkProfile` that simulates WAN latency/jitter/bandwidth for local testing. |

## How a splice serves

Each stage runs a full vLLM engine on its own layer slice. Stages execute in
lockstep: stage 0 broadcasts a control frame (admits/step/aborts), hidden
states flow stage-to-stage, and the last stage's sampled token rides the ring
back to stage 0, which streams it to the client. Per-sequence speed is
therefore bound by the ring latency — one round trip per token.

**Wave interleaving** (`--waves N`, default 32 from the PanoFabric spec's
`serving.waves`) recovers aggregate throughput: concurrent requests are
partitioned into up to N independent wave groups, and a stage computes one
wave while another wave's frames are on the wire. Requests join the emptiest
wave and idle waves are skipped, so low concurrency — even a single request —
behaves exactly like lockstep. `--waves` must match on every stage;
`0` = legacy lockstep.

## Installation

`torch` and `vllm` are **not** declared as dependencies: the runtimes that
launch these modules provide their own matched torch+vllm builds, and a plain
resolve of this package must not replace them. Only lightweight dependencies
(aiohttp, safetensors, huggingface_hub, numpy, transformers) are declared.

```bash
pip install panoserve @ git+https://github.com/PanocularAI/panoserve.git@main
```

For a standalone install where a stable torch is acceptable, the `vllm`
extra pulls the engine too: `pip install 'panoserve[vllm]'`.

PanoFabric installs this package into the interpreter its daemon runs from
(the control plane launches the modules as subprocesses) — an editable
sibling checkout for development, or the git URL via the
`panofabric-controld[serve]` extra on provisioned nodes.

## Tests

```
pytest panoserve/tests
```

`test_pipeline.py` is fast and GPU-free. `test_serve.py` needs a
torch-capable environment (it round-trips real tensors through the transport
and gateway) and includes deliberately slow WAN-pacing tests.
