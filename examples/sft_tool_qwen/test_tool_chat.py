#!/usr/bin/env python
"""Runnable check for the tool-calling loss mask. No framework, just asserts.

    <engine-venv>/bin/python test_tool_chat.py

Verifies on the REAL Qwen3 chat template that every assistant span is trained,
that nothing else is, and that the packaged data all renders.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "models"))

from sft_tool_qwen.tool_chat import (  # noqa: E402
    IGNORE_INDEX,
    ToolChatDataset,
    assistant_header_ids,
    conversation_processor,
    load_conversations,
    mask_labels,
)

DATA = HERE / "models" / "sft_tool_qwen" / "data.json"
REPO = "Qwen/Qwen3-4B"


def _tokenizer():
    """The real HFBackendTokenizer, loaded by path.

    Its package __init__ imports transformers, which a bare dev box often has
    pinned incompatibly; the module itself only needs torchtitan.
    """
    import torchtitan
    src = (Path(torchtitan.__file__).parent
           / "experiments/transformers_modeling_backend/tokenizer.py")
    spec = importlib.util.spec_from_file_location("_hf_backend_tokenizer", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from huggingface_hub import snapshot_download
    path = snapshot_download(
        REPO, allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json"]
    )
    return mod.HFBackendTokenizer(tokenizer_path=path)


def _spans(labels):
    """Token ids of each contiguous trained run."""
    out, cur = [], []
    for label in labels:
        if label != IGNORE_INDEX:
            cur.append(label)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def main() -> None:
    tk = _tokenizer()
    header = assistant_header_ids(tk)
    assert header, "empty assistant header"
    print(f"assistant header ids: {header}  eot: {tk.eos_id}")

    # --- unit: the mask itself ------------------------------------------------
    # header=[9,9], eot=0.  tokens: A A H H x y 0 B H H z 0
    toks = [7, 7, 9, 9, 1, 2, 0, 8, 9, 9, 3, 0]
    labels = mask_labels(toks, [9, 9], 0)
    trained = {i + 1 for i, v in enumerate(labels) if v != IGNORE_INDEX}
    assert trained == {4, 5, 6, 10, 11}, trained          # spans incl. terminators
    assert all(labels[i - 1] == toks[i] for i in trained)  # labels are shifted by one
    try:
        mask_labels([1, 2, 3], [9, 9], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("a sample with no assistant turn must raise")

    # --- end to end over the packaged data -----------------------------------
    ds = ToolChatDataset(
        dataset=load_conversations(str(DATA)),
        tokenizer=tk,
        sample_processor=conversation_processor,
        seq_len=4096,
    )
    rows = list(load_conversations(str(DATA)))
    total_spans = total_trained = total_tokens = 0
    for i, row in enumerate(rows):
        result = ds._tokenize_sample(row)
        assert result is not None, f"row {i} was dropped -- raise seq_len"
        inputs, labels = result
        assert len(inputs) == len(labels)

        payload = conversation_processor(row)
        expected = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        spans = _spans(labels)
        assert len(spans) == expected, (
            f"row {i}: {len(spans)} trained spans, {expected} assistant turns"
        )

        for span in spans:
            assert span[-1] == tk.eos_id, f"row {i}: span does not end at end-of-turn"
            text = tk.decode(span, skip_special_tokens=False)
            # The environment's turns must never be a target.
            assert "tool_response" not in text, f"row {i}: tool result is trained"
            assert "<|im_start|>" not in text, f"row {i}: span crosses a turn header"

        calls = sum(len(m["tool_calls"]) for m in payload["messages"] if m.get("tool_calls"))
        trained_text = "".join(tk.decode(s, skip_special_tokens=False) for s in spans)
        assert trained_text.count("<tool_call>") == calls, (
            f"row {i}: expected {calls} <tool_call> blocks in the trained spans"
        )

        total_spans += len(spans)
        total_trained += sum(1 for v in labels if v != IGNORE_INDEX)
        total_tokens += len(labels)

    # The bug this pipeline exists to avoid: a dangling generation prompt at the
    # end of the rendered text, which the mask would train the model to emit.
    payload = conversation_processor(rows[0])
    text = tk.apply_chat_template(
        payload["messages"], tools=payload["tools"], add_generation_prompt=False
    )
    assert not text.rstrip("\n").endswith("assistant"), (
        "render ends with a dangling assistant header"
    )

    pct = 100 * total_trained / total_tokens
    print(f"{len(rows)} rows, {total_spans} assistant spans, "
          f"{total_trained}/{total_tokens} tokens trained ({pct:.1f}%)")
    print("OK")


if __name__ == "__main__":
    main()
