#!/usr/bin/env python3
"""T-057 gate inputs: exactly N token IDs of a text (or of the concatenated files of a source tree) as the JSON array
tools/analysis/t056_capture.py reads, tokenized with the checkpoint's own tokenizer. Fails if the text is too short.
  t057_tokens.py <checkpoint> <tokens> <out.json> <text file | directory> [glob ...]   (directory: sorted globs)"""
import json
import pathlib
import sys

from transformers import AutoTokenizer


def text_of(source: pathlib.Path, globs: list[str]) -> str:
    if source.is_file():
        return source.read_text(errors="ignore")
    files = sorted({f for g in globs for f in source.glob(g) if f.is_file()})
    return "\n\n".join(f"=== {f.relative_to(source)} ===\n" + f.read_text(errors="ignore") for f in files)


def main() -> None:
    ckpt, n, out, source, *globs = sys.argv[1:]
    n = int(n)
    ids = AutoTokenizer.from_pretrained(ckpt).encode(text_of(pathlib.Path(source), globs), add_special_tokens=False)
    if len(ids) < n:
        sys.exit(f"t057_tokens: {source} has {len(ids)} tokens, fewer than {n}")
    pathlib.Path(out).write_text(json.dumps(ids[:n]))
    print(f"{out}: {n} of {len(ids)} tokens")


if __name__ == "__main__":
    main()
