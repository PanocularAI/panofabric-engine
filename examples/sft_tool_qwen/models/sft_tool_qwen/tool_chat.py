"""Tool-calling chat data: multi-span assistant loss masking.

torchtitan's ChatProcessor is single-turn ([user, assistant]) and masks ONE
prompt prefix (its `renderers` path is multi-turn but never passes `tools`), so
it cannot express a tool-calling trajectory

    system, user, assistant+tool_call, tool, assistant

which has TWO assistant spans with an environment turn between them.

ToolChatProcessor renders the whole conversation once through the model's own
chat template (tool schemas included), tokenizes once, then unmasks every span
that follows the template's assistant header. Loss lands on the assistant's tokens
only -- the tool-call JSON, the final answer, and each turn's terminator --
never on tool results: those are the environment's output, and training on them
teaches the model to hallucinate results instead of waiting for them.

Why scan for the header instead of re-rendering message prefixes (the obvious
extension of ChatProcessor's trick): Qwen3's template inserts an empty <think>
block into an assistant message only when it is the LAST one, so
render(msgs[:k]) is NOT a token prefix of render(msgs) and the per-turn length
deltas are wrong. Header scanning is immune to that, needs one tokenization
pass, and has no BPE-boundary hazard because role headers are special tokens.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, Features, Value, load_dataset

from torchtitan.components.data import TextSequence
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import ChatProcessor

logger = logging.getLogger(__name__)


def assistant_header_ids(tokenizer: BaseTokenizer) -> list[int]:
    """Token ids of the template's assistant header, derived from the template.

    The generation prompt IS the assistant header, so rendering one probe
    message with and without it and diffing yields the header for whatever
    model this is -- no hardcoded per-family token ids.
    """
    probe = [{"role": "user", "content": "x"}]
    with_gen = tokenizer.apply_chat_template(probe, add_generation_prompt=True)
    without = tokenizer.apply_chat_template(probe, add_generation_prompt=False)
    if not with_gen.startswith(without):
        raise ValueError(
            "chat template does not append the generation prompt; cannot locate "
            "assistant spans by header scan"
        )
    header = tokenizer.encode(with_gen[len(without):], add_bos=False, add_eos=False)
    if not header:
        raise ValueError("chat template has an empty generation prompt")
    return header


def mask_labels(tokens: list[int], header: list[int], eot: int) -> list[int]:
    """Labels for ``tokens[1:]``, IGNORE_INDEX outside assistant spans.

    A span runs from just after an assistant header to the next end-of-turn
    token, inclusive -- the terminator must be trained or the model never
    learns to stop a tool call.
    """
    labels = [IGNORE_INDEX] * (len(tokens) - 1)
    i = spans = 0
    while i <= len(tokens) - len(header):
        if tokens[i:i + len(header)] != header:
            i += 1
            continue
        start = i + len(header)
        end = start
        while end < len(tokens) and tokens[end] != eot:
            end += 1
        end = min(end, len(tokens) - 1)
        for j in range(start, end + 1):
            labels[j - 1] = tokens[j]   # labels[j-1] is the target for tokens[j]
        spans += 1
        i = end + 1
    if not spans:
        raise ValueError(
            "no assistant span found -- the sample has no assistant turn, or the "
            "chat template's header does not match the rendered text"
        )
    return labels


def _split_payload(payload: Any) -> tuple[list[dict], list[dict]]:
    """Accept either a bare message list or {"messages": ..., "tools": ...}."""
    if isinstance(payload, dict):
        return payload["messages"], payload.get("tools") or []
    return payload, []


class ToolChatProcessor(ChatProcessor):
    """ChatProcessor with tool schemas rendered and every assistant span trained.

    ``messages_fn`` returns a bare message list or {"messages", "tools"}.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChatProcessor.Config):
        pass

    def __init__(self, config: Config, *, context) -> None:
        super().__init__(config, context=context)
        self._header = assistant_header_ids(self._tokenizer)

    def __call__(self, sample: dict, rng: np.random.Generator) -> TextSequence | None:
        del rng
        messages, tools = _split_payload(self._messages_fn(sample))
        # add_generation_prompt=False is load-bearing: the tokenizer wrapper
        # defaults it to True, which appends a dangling assistant header the
        # mask would then train the model to emit after every answer.
        text = self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False
        ).rstrip("\n")
        tokens = self._tokenizer.encode(text, add_bos=True, add_eos=False)
        if tokens[-1] != self._eos_id:
            tokens.append(self._eos_id)

        # Drop rather than truncate: a trajectory cut mid tool-call is poison.
        if len(tokens) - 1 > self._max_context_length:
            logger.warning(
                f"dropping sample: {len(tokens)} tokens > max_context_length "
                f"{self._max_context_length} (tool schemas are expensive -- raise "
                f"training.max_context_length)"
            )
            return None

        sequence = self._create_sequence(tokens, full_text=text)
        sequence.labels[:] = mask_labels(tokens, self._header, self._eos_id)
        return sequence


def load_conversations(dataset_path: str, **load_kwargs):
    """Load conversations from a local JSON/JSONL file or any HF dataset id.

    A hub id (or anything else `datasets` understands) is passed straight to
    load_dataset as a map-style Dataset; the Grain pipeline shuffles, shards
    it per DP rank and repeats it. To stream a hub dataset instead, use
    torchtitan's HuggingFaceStreamingSource (see sft_tool_qwen_hermes).

    A local file is read with plain json and re-encoded as ONE string column.
    That is deliberate: Arrow's schema inference over nested tool schemas
    unifies structs across rows, which silently DROPS fields absent from the
    first rows (a tool's `required` list, say) and null-fills argument dicts.
    Encoding each row as a JSON string keeps the data exactly as authored.
    Hub datasets are unaffected -- they carry their own declared schema, and
    the ones that ship tool schemas usually ship them as a JSON string too.
    """
    path = Path(dataset_path)
    if path.suffix in (".json", ".jsonl"):
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text())
        return Dataset.from_list(
            [{"conversation": json.dumps(row)} for row in rows],
            features=Features({"conversation": Value("string")}),
        )
    return load_dataset(dataset_path, **{"split": "train", **load_kwargs})


def conversation_processor(sample: dict) -> dict:
    """Rows carrying a JSON-encoded conversation (see load_conversations)."""
    return json.loads(sample["conversation"])


# ShareGPT role tags -> chat roles. Hermes and most ShareGPT-derived tool sets
# use these; a dataset with different tags needs its own mapping, which is
# exactly the kind of per-dataset glue a messages_fn exists to hold.
_SHAREGPT_ROLES = {"system": "system", "human": "user", "gpt": "assistant",
                   "tool": "tool", "observation": "tool"}


def sharegpt_processor(sample: dict) -> dict:
    """NousResearch/hermes-function-calling-v1 and friends.

    `tools` is NOT forwarded on purpose: this dataset already embeds the
    <tools> block in its system prompt, so passing them again would render the
    schemas twice. The assistant turns carry <tool_call> markup as raw text,
    which the tokenizer still resolves to the real special tokens.
    """
    return {
        "messages": [
            {"role": _SHAREGPT_ROLES[turn["from"]], "content": turn["value"]}
            for turn in sample["conversations"]
        ],
        "tools": [],
    }


@dataclass(frozen=True, kw_only=True, slots=True)
class ConversationSource:
    """Grain source over load_conversations: a local file or a map-style hub id.

    `path` and `load_dataset_kwargs` mirror torchtitan's HF sources, so the
    engine's --dataset_path override swaps the path the same way.
    """

    path: str
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, *, dataset_iteration_policy) -> Dataset:
        del dataset_iteration_policy  # SingleDatasetConfig shuffles/shards/repeats
        return load_conversations(self.path, **self.load_dataset_kwargs)
