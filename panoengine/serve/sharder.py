# Copyright (c) Panocular AI.
#
# Weighted layer sharder for the cross-site pipeline.
#
# Splits an HF transformer checkpoint's decoder layers across S pipeline
# stages proportional to each stage's GPU memory, then writes one pruned
# checkpoint dir per stage plus a pipeline manifest. Stage 0 additionally
# carries the embeddings and the LAST stage the lm_head + final norm, so
# their layer shares are discounted by those tensors' actual bytes before
# apportioning. Every stage dir keeps config.json (patched with its layer
# range) and the tokenizer files; non-owned layers/embeddings/head are
# STRIPPED — a stage dir loads on a box that could never hold the full model.
#
# Offline CLI (pure torch, CPU-only — safetensors shuffling, no CUDA):
#   python -m panoengine.serve.sharder --checkpoint /path/to/hf-model \
#       --stage-memory 80,40,80 --out /path/to/staged
#
# The manifest (pipeline.json) records stage order, layer ranges, per-file
# sha256s, and the source model id — the engine stage and the
# integrity-checked fetch (serve/weights.py) both key off it.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .weights import sha256_file

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
# tensors owned by the edges regardless of layer math
_EMBED_KEYS = ("model.embed_tokens.",)
_HEAD_KEYS = ("lm_head.", "model.norm.")
_SIDECAR_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "special_tokens_map.json",
                  "generation_config.json", "tokenizer.model")


@dataclass
class StageAssignment:
    stage: int
    layer_start: int          # inclusive
    layer_end: int            # exclusive
    has_embeddings: bool
    has_head: bool
    checkpoint_dir: str
    files_sha256: dict[str, str]


def _tensor_owners(key: str, ranges: list[tuple[int, int]], *,
                   tied_embeddings: bool = False) -> set[int]:
    """The set of stages that carry a tensor key (empty = drop)."""
    m = _LAYER_RE.match(key)
    if m:
        layer = int(m.group(1))
        return {s for s, (lo, hi) in enumerate(ranges) if lo <= layer < hi}
    if any(key.startswith(p) for p in _EMBED_KEYS):
        # tie_word_embeddings models store NO lm_head.weight — the embedding
        # matrix IS the head, so the last stage needs a copy too or its tied
        # head resolves to random init.
        return {0, len(ranges) - 1} if tied_embeddings else {0}
    if any(key.startswith(p) for p in _HEAD_KEYS):
        return {len(ranges) - 1}
    # anything else (rotary inv_freq buffers etc.) is tiny: replicate to all
    return set(range(len(ranges)))


def plan_ranges(num_layers: int, stage_memory_gb: list[float],
                embed_bytes: int, head_bytes: int) -> list[tuple[int, int]]:
    """Apportion layers to stages proportional to memory, after discounting
    stage 0's embedding bytes and the last stage's head bytes. Layer counts
    are integers; remainders go to the largest fractional parts."""
    if len(stage_memory_gb) < 2:
        raise ValueError("a pipeline needs at least 2 stages")
    if num_layers < len(stage_memory_gb):
        raise ValueError(f"{num_layers} layers cannot fill "
                         f"{len(stage_memory_gb)} stages")
    budgets = [m * (1 << 30) for m in stage_memory_gb]
    budgets[0] = max(budgets[0] - embed_bytes, 0.0)
    budgets[-1] = max(budgets[-1] - head_bytes, 0.0)
    if sum(budgets) <= 0:
        raise ValueError("no stage memory left after embeddings/head")
    exact = [num_layers * b / sum(budgets) for b in budgets]
    counts = [int(x) for x in exact]
    # distribute the remainder to the largest fractional parts
    for i in sorted(range(len(exact)), key=lambda i: exact[i] - counts[i],
                    reverse=True)[: num_layers - sum(counts)]:
        counts[i] += 1
    # every stage must own >= 1 layer: steal from the fattest stage
    for i, c in enumerate(counts):
        while counts[i] == 0:
            donor = max(range(len(counts)), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[i] += 1
    ranges, lo = [], 0
    for c in counts:
        ranges.append((lo, lo + c))
        lo += c
    assert lo == num_layers
    return ranges


def shard_checkpoint(checkpoint: Path, out: Path,
                     stage_memory_gb: list[float], *,
                     engine_mode: bool = False,
                     stage_layers: list[int] | None = None) -> dict:
    """Split `checkpoint` (an HF dir with safetensors) into per-stage dirs
    under `out` and write out/pipeline.json. Returns the manifest dict.

    engine_mode=True prepares stages for the vLLM-engine runtime (the
    prime-vllm technique): layer keys are RENUMBERED to 0-based so vLLM
    loads each stage dir as a plain k-layer model, and embed_tokens +
    final norm + lm_head ride EVERY stage — each stage runs a full local
    engine whose scheduler needs to embed and sample (the ring overrides
    its sampler output with the real tokens). Costs one edge-tensor copy
    per stage; the default keeps the stripped form."""
    import torch                      # noqa: F401  (safetensors needs it)
    from safetensors.torch import load_file, save_file

    checkpoint = Path(checkpoint)
    out = Path(out)
    config = json.loads((checkpoint / "config.json").read_text())
    num_layers = config["num_hidden_layers"]

    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no .safetensors under {checkpoint}")
    tensors = {}
    for wf in weight_files:
        tensors.update(load_file(str(wf)))

    def _nbytes(pred) -> int:
        return sum(t.numel() * t.element_size()
                   for k, t in tensors.items() if pred(k))

    # Fail loudly on an unrecognized layer naming scheme: without this a
    # checkpoint using e.g. "transformer.h.N." matches no layer key, so every
    # tensor replicates to every stage while each stage's config still claims
    # its slice — sha-valid manifests that only misbehave at engine boot.
    if not any(_LAYER_RE.match(k) for k in tensors):
        raise ValueError(
            f"no tensors match {_LAYER_RE.pattern!r} in {checkpoint} — the "
            "sharder only understands HF 'model.layers.N.' naming")

    embed_bytes = _nbytes(lambda k: any(k.startswith(p) for p in _EMBED_KEYS))
    head_bytes = _nbytes(lambda k: any(k.startswith(p) for p in _HEAD_KEYS))
    if config.get("tie_word_embeddings"):
        head_bytes += embed_bytes   # the last stage carries the tied matrix

    if stage_layers is not None:
        # explicit per-stage layer counts (heterogeneous topologies where
        # the operator sizes stages by hand, e.g. TP2 stage gets fewer
        # layers than the single-GPU stages downstream)
        if sum(stage_layers) != num_layers:
            raise ValueError(f"stage_layers {stage_layers} must sum to "
                             f"{num_layers}")
        ranges, lo = [], 0
        for c in stage_layers:
            ranges.append((lo, lo + c))
            lo += c
    else:
        ranges = plan_ranges(num_layers, stage_memory_gb,
                             embed_bytes, head_bytes)

    stages: list[StageAssignment] = []
    for s, (lo, hi) in enumerate(ranges):
        sdir = out / f"stage{s}"
        sdir.mkdir(parents=True, exist_ok=True)
        tied = bool(config.get("tie_word_embeddings"))
        if engine_mode:
            def _keep(k: str) -> bool:
                m = _LAYER_RE.match(k)
                if m:
                    return lo <= int(m.group(1)) < hi
                return True     # all edge + replicated tensors on every stage

            def _rekey(k: str) -> str:
                m = _LAYER_RE.match(k)
                if not m:
                    return k
                rest = k[m.end():]
                return f"model.layers.{int(m.group(1)) - lo}.{rest}"

            owned = {_rekey(k): t for k, t in tensors.items() if _keep(k)}
        else:
            owned = {k: t for k, t in tensors.items()
                     if s in _tensor_owners(k, ranges, tied_embeddings=tied)}
        save_file(owned, str(sdir / "model.safetensors"),
                  metadata={"format": "pt"})
        # a stage's config claims only its own layers so a plain HF load of
        # the stage dir materializes the right depth
        stage_cfg = dict(config)
        stage_cfg["num_hidden_layers"] = hi - lo
        stage_cfg["panofabric_stage"] = {
            "stage": s, "layer_start": lo, "layer_end": hi,
            "num_source_layers": num_layers, "engine_mode": engine_mode,
        }
        (sdir / "config.json").write_text(json.dumps(stage_cfg, indent=2))
        for name in _SIDECAR_FILES:
            src = checkpoint / name
            if src.exists():
                shutil.copy2(src, sdir / name)
        stages.append(StageAssignment(
            stage=s, layer_start=lo, layer_end=hi,
            has_embeddings=(s == 0), has_head=(s == len(ranges) - 1),
            checkpoint_dir=str(sdir),
            files_sha256={f.name: sha256_file(f) for f in sorted(sdir.iterdir())
                          if f.is_file()},
        ))

    manifest = {
        "source_checkpoint": str(checkpoint),
        "model_type": config.get("model_type"),
        "num_layers": num_layers,
        "stage_memory_gb": stage_memory_gb,
        "stages": [asdict(st) for st in stages],
    }
    (out / "pipeline.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="weighted pipeline layer sharder")
    p.add_argument("--checkpoint", required=True, help="HF model dir")
    p.add_argument("--out", required=True, help="output dir for stage dirs")
    p.add_argument("--stage-memory", required=True,
                   help="comma-separated per-stage GPU memory in GB, e.g. 80,40")
    p.add_argument("--stage-layers", default=None,
                   help="explicit comma-separated layer count per stage (e.g. "
                        "8,10,10; must sum to the model's depth) — overrides "
                        "the memory-proportional split")
    p.add_argument("--engine-mode", action="store_true",
                   help="renumber layers 0-based and replicate embed/norm/head "
                        "to every stage (what serve/engine_stage.py loads)")
    args = p.parse_args()
    manifest = shard_checkpoint(
        Path(args.checkpoint), Path(args.out),
        [float(x) for x in args.stage_memory.split(",")],
        engine_mode=args.engine_mode,
        stage_layers=([int(x) for x in args.stage_layers.split(",")]
                      if args.stage_layers else None))
    for st in manifest["stages"]:
        print(f"stage {st['stage']}: layers [{st['layer_start']}, "
              f"{st['layer_end']}) embeddings={st['has_embeddings']} "
              f"head={st['has_head']} -> {st['checkpoint_dir']}")


if __name__ == "__main__":
    main()
