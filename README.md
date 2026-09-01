<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/418f6c0d-9293-48d0-9fb1-bfac8c73102a" alt="PanocularAI - PanoFabric" width="312">
</p>

---

<h3 align="center">
panofabric-engine: PyTorch-native engine for heterogeneous & decentralized AI training & serving
</h3>

---


## 🧭 Overview

As AI models, especially Large Language Models (LLMs) and Vision-Language Models (VLMs), continue to grow in scale and complexity, the need for heterogeneous and decentralized training strategies is becoming increasingly critical. Training such massive models demands enormous computational resources, which are often inaccessible to most researchers and organizations.

HPC centers around the world host a wide variety of GPUs, ranging across different vendors, architectures, and hardware configurations. However, these variations introduce compatibility and utilization challenges, often preventing AI researchers from seamlessly leveraging multiple HPC systems at once.

This engine is a practical approach to overcoming those challenges: it connects heterogeneous compute in a decentralized manner using DiLoCo and its relatives, enabling collaborative, cross-platform training without homogeneous hardware or centralized orchestration. Two levels of heterogeneity are supported:

* **Cross-hardware heterogeneity:** train across multiple hardware platforms, regardless of vendor or GPU generation.
* **Non-uniform GPU distribution:** clusters vary widely in node configuration (commonly 4 or 8 GPUs per node). Islands of different sizes train together.

## 📊 What it covers

Every recipe in this repo is a `FaultTolerantTrainer.Config`, so a workload does not opt in to decentralized training — it gets it by construction. The strategy is chosen at launch (`--fault_tolerance.semi_sync_method=...`), not baked into the recipe.

|                                          | solo | DiLoCo | HeLoCo | async |
|------------------------------------------|:----:|:------:|:------:|:-----:|
| **pretraining**                          |  ✅  |   ✅   |   ✅   |  ✅   |
| **LoRA fine-tuning**                     |  ✅  |   ✅   |   ✅   |  ✅   |
| **SFT (instruction tuning)**             |  —   |   —    |   —    |  —    |
| **RL (GRPO)** <sup>†</sup>               |  ✅  |   ✅   |   ✅   |  ✅   |
| **serving (spliced pipeline)**           |  ✅  |  n/a   |  n/a   |  n/a  |

<sup>†</sup> `panoengine/train/rl/` holds the presets and the decentralized coordinators —
the Monarch trainer actors, the four replica strategies and their launch entry points — and
`panoengine/decentralized/` holds the parameter server, relay and rollout queue. The RL
implementation they all build on stays in the torchtitan fork and is imported from there, not
copied. See [FORK-DELTA.md](FORK-DELTA.md) for exactly where that line falls.

**On the SFT row:** the em-dashes are honest. What exists is LoRA fine-tuning
(`models/lora`) and a full-parameter HF backend
(`models/hf_transformers`) — but both train on raw text, so they are
parameter-efficient and full-parameter *continued pretraining*. Instruction tuning needs a
data path that does not exist yet: chat-template rendering, prompt-token loss masking,
sample packing with correct attention boundaries, and an eval split. Until that lands,
this repo does not claim SFT. (An `sft/` directory aliasing LoRA would be discovered in
ten minutes and would cost you the rest of the table.)

## 🗂️ Layout

```
panoengine/            the machinery the recipes compose
├── decentralized/    async DiLoCo + HeLoCo, the parameter server, relay, rollout queue
├── train/
│   ├── rl/           decentralized RL: presets, the four replica strategies,
│   │                 the Monarch trainer actors and their launch entry points
│   └── strategies.py the fault-tolerance defaults every recipe shares
└── serve/            the spliced-pipeline inference plane
models/               the training recipes, one package per model — llama3, qwen3,
                      gpt_oss, hf_transformers, lora, resnet. The path a run spec
                      names and ConfigManager resolves for --module
examples/
└── decentralized/    runnable drivers for async DiLoCo and HeLoCo
```

`panoengine.decentralized` needs only torch and torchft — not torchtitan — so it installs
on its own with the `[decentralized]` extra, and the forks' RL adapter can import it without
a dependency cycle.

Each recipe's `config_registry.py` assembles a whole training stack — trainer, optimizer,
loss, LR schedule, dataloader, activation-checkpoint policy, fault-tolerance manager and
checkpoint manager. torchtitan's `ConfigManager` selects one by module and function name,
which *is* the plugin system: any installed module exposing a `config_registry` works.

## 🧪 Tested platforms

- [x] Nvidia GPUs (L40S, A100, H100, H200)
- [x] AMD GPUs (MI300X)

Training on a CPU backend is not supported as of now.

## 📦 Installation

The engine owns its decentralized algorithms (`panoengine.decentralized`) and composes two
forks, [torchtitan](https://github.com/PanocularAI/torchtitan) and
[torchft](https://github.com/PanocularAI/torchft), for the pieces that must subclass upstream
internals — see [FORK-DELTA.md](FORK-DELTA.md) for exactly where that line falls.

**One thing to know first:** torchft builds its Rust extension via maturin, so installing
it from source needs a Rust toolchain, `protoc` 32.0, and CPython ≤ 3.13 (pyo3 0.24's
ceiling). That is the single biggest install-friction item here, and it is why the paths
below are ordered the way they are.

```bash
# 1. Development on the engine itself: sibling fork clones, installed editable.
git clone https://github.com/PanocularAI/panofabric-engine.git
cd panofabric-engine
make all          # toolchain, project, backend-matched torch, the forks
make dev-forks    # optional: editable forks for hacking on them

# 2. Ordinary install (needs the Rust toolchain above).
pip install 'panofabric-engine[train]'

# 3. Just the decentralized algorithms (async DiLoCo / HeLoCo), no torchtitan.
pip install 'panofabric-engine[decentralized]'
```

There are no submodules: `git clone --recursive` plus a Rust build is where a newcomer
bounces. `make all` clones the forks as siblings at their pinned SHAs.

> **Install-order trap.** torchft declares `torch>=2.7`, so resolving `[train]`
> unconstrained can pull a *stable* torch over a backend-matched *nightly* and clobber the
> stack. torch must be installed **last** — which is what `make install-torch` does.

If a Makefile step fails on your setup, see the [Installation Guideline](docs/installation.md).

### Setting up Tailscale VPN

To establish communication between compute islands, each node needs a routable address. If
public IPs are not available, use Tailscale — see the
[instructions](docs/installation.md#tailscale-setup).

## 🚀 Quickstart: train across two machines

This is the thing the stack is for. Two islands, separate networks, one model — run these
in three shells. For fully heterogeneous setups (different GPU counts and vendors) see
[Launching Training](docs/distributed.md).

1. Start the lighthouse (the rendezvous service the islands find each other through):
```bash
RUST_BACKTRACE=1 torchft_lighthouse --bind=<public_ip>:29510 \
  --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. Island 0:
```bash
TORCHFT_LIGHTHOUSE=http://<public_ip>:29510 \
NGPU=1 \
LOCAL_ADDR=<local_ip> \
MASTER_ADDR=<master_c10d_ip> \
MASTER_PORT=29500 \
NNODES=<num_nodes> \
ISHOST=true \
GLOO_SOCKET_IFNAME=<network_card> \
NCCL_SOCKET_IFNAME=<network_card> \
MODULE="models.llama3" \
CONFIG_NAME="llama3_debugmodel" \
uv run ./run_train.sh --fault_tolerance.enable \
  --fault_tolerance.replica_id=0 --fault_tolerance.group_size=2
```

3. Island 1 — identical, with `ISHOST=false` and `--fault_tolerance.replica_id=1`.

Swap the strategy without touching the recipe:

```bash
uv run ./run_train.sh ... --fault_tolerance.semi_sync_method=heloco
```

## 🤖 Models and recipes

| Recipe | Module |
|---|---|
| Llama3 | `models.llama3` |
| Qwen3 (incl. MoE) | `models.qwen3` |
| GPT-OSS | `models.gpt_oss` |
| Any HF architecture | `models.hf_transformers` |
| LoRA (Qwen3, Llama3) | `models.lora` |
| ResNet / CIFAR-10 | `models.resnet` |

`models.<pkg>` is the canonical path: it is what the control plane stores in a run
spec and what torchtitan's ConfigManager resolves for `--module`.

Before training, download the tokenizer:

```bash
uv run python ../torchtitan/scripts/download_hf_assets.py \
  --repo_id <hf_repo_name> --assets tokenizer --hf_token=$HF_TOKEN
```

Replace `<hf_repo_name>` with the HF model path, e.g. `meta-llama/Llama-3.1-8B` or
`Qwen/Qwen3-0.6B`. Llama3 is a gated repo, so it needs `$HF_TOKEN` and an access request.

Many more architectures exist in
[TorchTitan models](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models) and
its [experiments](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments),
and new ones follow the [Adding a new model tutorial](docs/model.md).

## 🔌 Serving: one model spliced across GPU islands

Decentralized *inference* is the other half. `panoengine.serve` splits one model into
pipeline stages, runs each stage as its own vLLM engine — possibly in a different
datacenter — and stitches them back together over the stage transport:

```bash
# Shard a checkpoint into per-stage weight sets.
python -m panoengine.serve.sharder --checkpoint <hf-model> --out staged/ --stage-memory <GiB>

# Run a stage (once per island).
python -m panoengine.serve.engine_stage --stage-dir staged/stage1

# Front the fleet with the gateway (OpenAI-compatible).
python -m panoengine.serve.gateway --port 8800
```

Install it with the `serve` extra. As with training, torch and vLLM are deliberately
**not** declared: every runtime that launches these modules carries backend-matched
nightlies installed last, and a plain resolve would clobber them.

## 📑 Documentation

- [Detailed installation tutorial](docs/installation.md)
- [Decentralized and heterogeneous training](docs/distributed.md)
- [What DiLoCo / HeLoCo / async actually do](docs/decentralized-training.md)
- [Adding a new model](docs/model.md)
- [Deployment on cloud using SkyPilot](docs/skypilot.md)
- [What we changed in the forks, and why](FORK-DELTA.md)

## 🙏 Acknowledgement

This work builds upon the following open-source frameworks:

* [TorchTitan](https://github.com/meta-pytorch/torchtitan) — a PyTorch-native platform for large-scale generative AI model training (Liang et al., ICLR 2025).
* [TorchFT](https://github.com/meta-pytorch/torchft) — a library providing fault-tolerance primitives for distributed PyTorch training (HSDP, LocalSGD, DiLoCo, Streaming DiLoCo).

We gratefully acknowledge the PyTorch, TorchTitan, and TorchFT teams for their foundational contributions to distributed and fault-tolerant ML training infrastructures.
