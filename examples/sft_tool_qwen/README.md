# Tool-calling SFT of Qwen3-4B — a standalone panofabric workload

This is a standalone SFT training example that can be used for submitting to PanoFabric; the package is uploaded as a workspace **code overlay**,
installed into the engine's `models` package on every island, and named by
`model.module` in the spec. This is the template for "bring your own training
code and your own data".

```
models/sft_tool_qwen/
  __init__.py           what the overlay is        (required member)
  config_registry.py    the preset                 (required member)
  tool_chat.py          the only real logic: multi-span assistant loss masking
  data.json             12 demo tool-calling trajectories
spec.yaml               the run
test_tool_chat.py       runnable check of the mask against the real Qwen3 template
```

## What SFT for tool calling actually needs

One row is a whole trajectory, and loss lands **only** on the assistant's own
tokens:

```
system   (tool schemas)                          masked
user     "weather in Lisbon?"                    masked
assistant <tool_call>{...}</tool_call><|im_end|>  TRAINED
user     <tool_response>{...}</tool_response>     masked   <- the environment's output
assistant <think></think>It's 19 °C.<|im_end|>    TRAINED
```

Training on tool results teaches the model to hallucinate results instead of
waiting for them, so they stay masked. Each trained span **includes** its
`<|im_end|>` terminator, or the model never learns to stop a tool call.

torchtitan's own `ChatDataset` cannot express this: it hard-rejects anything but
`[user, assistant]` and masks a single prompt prefix. `tool_chat.py` renders the
whole conversation once through the model's own chat template (tool schemas
included), tokenizes once, and unmasks every span following the template's
assistant header — derived from the template itself, so no hardcoded token ids.

Header scanning rather than per-turn prefix re-rendering is deliberate: Qwen3's
template inserts an empty `<think>` block into an assistant message only when it
is the *last* one, so `render(msgs[:k])` is not a token prefix of `render(msgs)`
and the length deltas silently misalign the mask.

## Run it

Local check of the mask (no GPU):

```bash
<engine-venv>/bin/python test_tool_chat.py
# 12 rows, 28 assistant spans, 726/4984 tokens trained (14.6%)
```

Local smoke test on one GPU with a 2-layer stand-in — proves the data pipeline
and the packed-sequence mask, not the model:

```bash
torchrun --nproc_per_node=1 -m torchtitan.train \
  --module models.sft_tool_qwen --config sft_tool_qwen_debug \
  --hf_model Qwen/Qwen3-4B --hf_assets_path <dir with Qwen3 tokenizer> \
  --training.steps=3 --fault_tolerance.no_enable
```

On PanoFabric. First fill in `islands[].resources.infra` in `spec.yaml` — the
one thing only you can know: which machines to run on. Either a cloud
(`nebius`, `aws`, `aws/us-east-1`) or an enrolled BYO backend in slash form
(`ssh/<pool>`, `slurm/<cluster>[/<partition>]`, `k8s/<ctx>`); omit the field
entirely to let PanoFabric pick. `panofabric backend list` prints the exact
strings — its "BYO backends" section is the `resources.infra` column verbatim.
Keep the slash: everything before it is parsed as the cloud name, so
`slurm-mycluster` would pass validation and die at island launch, after the
SyncHub and sibling islands are already provisioned and billing. Then:

```bash
panofabric run submit spec.yaml --code models/sft_tool_qwen
```

`--code` packs that dir into a deterministic `models/<pkg>/` tar.gz, runs the
same trust-boundary checks and uploads it to the compute islands.
Identical bytes dedup by digest, so re-submitting unchanged code re-uses the
stored archive.

The launcher then fetches `Qwen/Qwen3-4B` (tokenizer **and** safetensors) into
`assets/hf/Qwen--Qwen3-4B` on each island, installs this overlay, and runs
`torchtitan.train --module models.sft_tool_qwen --config sft_tool_qwen`.

## Bringing your own data

Three sources, all through the same loader:

| source | how |
|---|---|
| the packaged `data.json` | the default; travels inside the overlay |
| a hub dataset, streamed | `sft_tool_qwen_hermes` — `load_dataset_kwargs={"streaming": True}` |
| a path on shared storage | `data.dataset_path` in the spec, or in the preset |

`data.json` ships inside the overlay only because it is tiny — the upload cap is
10 MiB compressed and the allowed extensions are `.py .json .toml .yaml .txt
.md`, and the rule those extensions encode is "datasets ride their own
channel". Treat it as the smoke-test fixture, not the mechanism.

**Streaming** is the normal answer for real data. `sft_tool_qwen_hermes` streams
`NousResearch/hermes-function-calling-v1`: nothing is downloaded up front, and
`ChatDataset` shards the stream per node with `split_dataset_by_node`,
buffer-shuffles it, and re-loops via `set_epoch`. The one thing streaming gives
up is exact resume — `ChatDataset` can only `.skip()` a map-style dataset, so a
restart replays the shard from its start instead of the exact sample. For a
short SFT run that is nothing; for a long one, prefer a map-style local copy.

Per-dataset shape lives in the `sample_processor`, which is why it is a Python
callable and can never come from the spec: `conversation_processor` reads the
packaged file, `sharegpt_processor` maps Hermes' `from`/`value` turns. Hermes
also shows a trap worth naming — it embeds the `<tools>` block in its own system
prompt, so its processor returns `tools: []`; forwarding the dataset's `tools`
column too would render every schema twice.

Local files are read with plain `json` and re-encoded as one string column
on purpose: Arrow's schema inference over nested tool schemas unifies structs
across rows, which drops fields missing from the early rows (a tool's `required`
list) and null-fills argument dicts. That corruption is silent. Hub datasets
carry their own declared schema and are unaffected (Hermes ships `tools` as a
JSON string for the same reason).

What determines quality is the mix, not the loss curve. This set deliberately
includes no-tool-needed turns, a tool error and recovery, parallel calls in one
turn, a chain where call 2 depends on result 1, a missing-required-argument
clarification, an empty result, a decline when no tool fits, and one row with
eight tools in context. Twelve rows is a demo; real work needs thousands, and
should perturb tool order and names per row so the model reads the schema
instead of memorizing positions.

Evaluate on behavior, not loss: JSON parse rate, exact tool-name match, per-arg
F1, over-call rate on no-tool prompts, end-to-end task success.

## Notes from building this

- `add_generation_prompt=False` on the full render is load-bearing.
  `HFBackendTokenizer` defaults it to **True**, which appends a dangling
  `<|im_start|>assistant` the mask then trains the model to emit after every
  answer. torchtitan's `ChatDataset` has this bug today.
- Qwen3-4B has tied word embeddings; FSDP shards `tok_embeddings` and `lm_head`
  into separate groups and rejects a tied weight spanning both. The preset
  reuses the engine's `_untied_flavor`, whose state-dict adapter aliases the
  tied checkpoint into the untied `lm_head` at load.
- `parallelism.spmd_backend = "partial_dtensor"` is required — the HF
  transformers backend raises `NotImplementedError` under `spmd_types`. The
  engine's own `hf_full` / `hf_finetune` presets do not set it.
- Do **not** set `data.dataset` in the spec. The launcher emits
  `--dataloader.dataset`, which `ChatDataLoader.Config` has no field for, and
  tyro aborts the run before the first step.
- Enabling this required one control-plane change (already applied in
  `panofabric`): a code overlay naming `model.hf_model` now gets that repo's
  weights fetched, and `hf_model` is legal alongside `model.code`. Before it,
  `_hf_repo` returned `None` for a custom module, the spec was rejected outright,
  and an overlay could only ever train from random init.
