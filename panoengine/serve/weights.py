# Copyright (c) Panocular AI.
#
# Streamed, integrity-verified weight fetch.
#
# Downloads a manifest of files by ranged HTTP chunks with bounded concurrency
# (the ZML-style technique), sha256-verifying every file before it is moved
# into place — a partial or corrupt write can never masquerade as a complete
# checkpoint (the failure mode of interrupted hub downloads: a disk-full or
# dropped connection surfaces as an integrity error, not a poisoned cache).
# Resume: complete verified files are skipped on retry; incomplete temp files
# are restarted from zero (chunk-level resume isn't worth the bookkeeping at
# our file sizes).
#
# The manifest format is the sharder's pipeline.json stage entry
# ({filename: sha256}) so a stage island fetches exactly its own shard —
# private artifacts ride the same relay URLs the RL weight path uses.

from __future__ import annotations

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CHUNK = 8 << 20          # 8 MiB ranged reads
_DEFAULT_WORKERS = 4


class IntegrityError(Exception):
    """Downloaded bytes do not match the manifest's sha256."""


def sha256_file(path: Path) -> str:
    """Chunked file digest — weight shards are multi-GB; never read_bytes()."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_length(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        accept_ranges = resp.headers.get("Accept-Ranges", "")
    return int(cl) if cl and "bytes" in accept_ranges else None


def _fetch_range(url: str, start: int, end: int, timeout: float) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_file(url: str, dest: Path, sha256: str, *,
               workers: int = _DEFAULT_WORKERS, timeout: float = 60.0) -> Path:
    """Download url -> dest, verifying sha256. Already-verified files are
    skipped (resume). Writes to dest.part and renames only after the digest
    matches, so dest existing implies dest is good."""
    dest = Path(dest)
    if dest.exists():
        if sha256_file(dest) == sha256:
            return dest                       # verified resume hit
        dest.unlink()                         # stale/corrupt: refetch

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = _content_length(url, timeout)
    h = hashlib.sha256()
    with open(tmp, "wb") as f:
        if total is None:
            # server doesn't do ranges: single stream
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                for chunk in iter(lambda: resp.read(_CHUNK), b""):
                    h.update(chunk)
                    f.write(chunk)
        else:
            spans = [(s, min(s + _CHUNK, total))
                     for s in range(0, total, _CHUNK)]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # ordered iteration keeps the hash sequential while the
                # ranged reads themselves overlap
                for data in pool.map(
                        lambda sp: _fetch_range(url, sp[0], sp[1], timeout),
                        spans):
                    h.update(data)
                    f.write(data)
    if h.hexdigest() != sha256:
        tmp.unlink()
        raise IntegrityError(
            f"{url}: sha256 mismatch (got {h.hexdigest()[:12]}…, "
            f"manifest says {sha256[:12]}…)")
    tmp.rename(dest)
    return dest


def _get_json(url: str, timeout: float = 30.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_stage(manifest_url: str, stage: int, dest_dir: Path) -> Path:
    """Verified fetch of one pipeline stage's checkpoint dir from a
    published manifest tree: <manifest_url>/pipeline.json (the sharder's
    manifest) names per-file sha256s for each stage."""
    manifest = _get_json(f"{manifest_url.rstrip('/')}/pipeline.json")
    entry = manifest["stages"][stage]
    fetch_manifest(f"{manifest_url.rstrip('/')}/stage{stage}",
                   entry["files_sha256"], dest_dir)
    return Path(dest_dir)


def fetch_model_dir(manifest_url: str, dest: Path) -> Path:
    """Verified fetch of a full model dir: <manifest_url>/manifest.json maps
    filename -> sha256 (same shape as a sharder stage's files_sha256)."""
    fetch_manifest(manifest_url, _get_json(
        f"{manifest_url.rstrip('/')}/manifest.json"), dest)
    return dest


def fetch_manifest(base_url: str, files_sha256: dict[str, str],
                   dest_dir: Path, *, workers: int = _DEFAULT_WORKERS) -> list[Path]:
    """Fetch every manifest file under base_url into dest_dir, verified."""
    out = []
    for name, digest in files_sha256.items():
        out.append(fetch_file(f"{base_url.rstrip('/')}/{name}",
                              Path(dest_dir) / name, digest,
                              workers=workers))
    return out


def main() -> None:
    """Fetch-then-exec wrapper for PLAIN inference islands: verified manifest
    download into a local dir, then exec vLLM's OpenAI server on it, replacing
    this process (signals/teardown flow to vLLM directly; the control plane
    sees ONE process either way). Splice stages have their own path
    (engine_stage --manifest-url / --ensure-from); this covers ordinary
    serving runs whose weights come from a relay/private artifact store
    instead of the HF hub:

        python -m panoengine.serve.weights \\
            --manifest-url http://relay/models/qwen3-0.6b --dest /data/w \\
            -- --port 8800 --max-model-len 8192 ...     (vllm serve args)
    """
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(
        description="verified-fetch wrapper around the vLLM OpenAI server")
    p.add_argument("--manifest-url", required=True)
    p.add_argument("--dest", required=True,
                   help="local dir the verified weights land in")
    p.add_argument("vllm_args", nargs=argparse.REMAINDER,
                   help="args after `--` go to vllm.entrypoints.openai."
                        "api_server verbatim (do NOT pass --model)")
    args = p.parse_args()

    model_dir = fetch_model_dir(args.manifest_url, Path(args.dest))
    vllm_args = [a for a in args.vllm_args if a != "--"]
    if "--model" in vllm_args:
        raise SystemExit("--model is set by the wrapper (the fetched dir)")
    os.execv(sys.executable, [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir), *vllm_args,
    ])


if __name__ == "__main__":
    main()
