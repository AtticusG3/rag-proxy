"""Export captured fine-tuning turns as dataset JSONL.

Examples:
  python scripts/export_finetune_dataset.py \
    --input /var/lib/rag_proxy/capture/finetune.jsonl \
    --output finetune_messages.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") == "finetune_turn":
                records.append(record)
    return records


def build_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("conversation_id") or record.get("trace_id") or len(grouped))
        grouped.setdefault(key, []).append(record)

    examples: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(key=lambda record: str(record.get("ts") or ""))
        latest = group[-1]
        messages = _dedupe_consecutive_messages(
            list(latest.get("messages") or []) + [latest.get("assistant") or {}]
        )
        messages = [msg for msg in messages if msg.get("role") and msg.get("content")]
        if len(messages) >= 2:
            examples.append({"messages": messages})
    return examples


def write_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")


def _dedupe_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if out and out[-1].get("role") == message.get("role"):
            if out[-1].get("content") == message.get("content"):
                continue
        out.append(message)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fine-tuning JSONL from capture logs.")
    parser.add_argument("--input", required=True, type=Path, help="Path to finetune.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    examples = build_examples(records)
    write_examples(args.output, examples)
    print(f"Wrote {len(examples)} example(s) to {args.output}")


if __name__ == "__main__":
    main()
