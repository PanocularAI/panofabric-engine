# RL for a tool-calling agent — your agent loop, our trainer

This is the template for `kind: rl` when the agent loop must stay **your** code:
a harness that talks to an OpenAI-compatible endpoint, calls tools across several
turns, and is scored afterwards. The package is uploaded as a workspace **code
overlay**, like any custom model.

The task itself is a toy: each episode names 3-10 parcels, the agent looks up
their weights (in mixed units) with one tool and submits the total in kilograms
with another, and the reward is how close the total is, minus a penalty per
malformed tool call.

```
models/rl_tool_agent/
  __init__.py           what the overlay is                       (required member)
  config_registry.py    the presets: engine Qwen3.5 presets + this task  (required member)
  parcels.py            the task, its rewards, and the agent loop
spec.yaml               the run (Qwen3.5-9B, one H100:8 island, rl_solo)
```

## How the episode runs

```
your agent loop ── openai.AsyncOpenAI(base_url=<interception>) ──┐
   (ParcelHarness.run_agent)                                     │  every turn recorded
                                                                 ▼  token-exact
                         Verifiers interception server ──► the live policy (vLLM, current weights)
                                                                 │
finished trace ──► @vf.reward methods (Reward, tool errors) ──► GRPO update ──► new weights
```

- The loop is ordinary OpenAI-client code. It never sees a URL of yours: the
  `client` it is handed already points at the interception server, which records
  each request/response as tokens and forwards it to the generator. A call made
  through any other client is invisible to training.
- Tool calls in the replies are parsed by the Qwen3.5 renderer (its XML
  `<function=…><parameter=…>` format), so `reply.tool_calls` is populated the same
  way a `--tool-call-parser qwen3_coder` server would populate it. No vLLM parser
  is involved in training.
- Rewards are `@vf.reward` methods on the task. They read what the loop stored in
  `trace.info` (here: the submitted total and the number of malformed calls). The
  episode's reward is their weighted sum.

## Adapting it

1. Replace `ParcelTaskset.load()` with your tasks (one `TaskData` per episode; put
   anything your loop or rewards need on it as fields).
2. Replace `ParcelHarness.run_agent` with your loop. Keep using the `client` and
   `model` it is given. Tools run wherever your loop runs them — calls out to your
   own services work as long as the island can reach them.
3. Replace the `@vf.reward` methods. They run after the loop, on the same trace.
4. Keep `__all__ = [<Taskset>, <Harness>]` in that module: Verifiers loads the
   harness from the taskset's module, and the engine refuses a taskset module
   that exports none (Verifiers would otherwise fall back to a shell harness).

Budget: `SEQ_LEN` bounds the whole conversation (prompt, tool schemas, every turn
and tool result). Raise it with the turn count, and the generator's
`sampling.max_tokens` with the length of one reply.

## Strategies

`config_registry.py` defines the single-island (`rl_solo_*`), colocated multi-island
(`rl_heloco_*`) and decoupled (`rl_heloco_async_inference_*` + `_worker_`)
variants. controld derives the strategy segment from the spec: `sync: {method:
none}` on one island runs `rl_solo_*` (what `spec.yaml` uses — no sync hub; at 9B it
measured ~46 GB of host RAM against heloco's ~400-500 GB, with 5-10x shorter
windows), `method: heloco` runs `rl_heloco_*`, and adding `role: generator`
islands runs the decoupled pair, all from the same `preset:` value.

## Run it

```bash
panofabric run submit spec.yaml --code models/rl_tool_agent
```

The 0.8B presets (`rl_solo_tool_agent_qwen3_5_0_8b`, `hf_model:
Qwen/Qwen3.5-0.8B`, `accelerators: "H100:2"`) are the cheap end-to-end check.
