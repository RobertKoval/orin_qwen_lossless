"""Benchmark inputs built from a frozen public corpus.

Private planning documents are not included. Long prefill/haystack inputs therefore differ from historical local
experiments and require a new baseline; short decode inputs are retained. Editing current docs does not change
these fixtures. `python -m bench.prompts --sha` prints a sha256 per input.

  PROMPTS / N_PREDICT / TOP_K   the four greedy prompts of bench/xengine/run_prompts.sh (three short, one ~1500-token
                                document): llama.cpp reference (ref_llama.py), vLLM comparison (compare_llama.py)
  serving_prompts()             the four decode prompts of bench/serving/bench_openai.py (document cut to 6000 chars)
  serving_long_doc()            bench_openai.py's ~12k-token prefill document
  prefill_doc()                 bench/serving/bench_prefill.py's document (cut to N tokens there)
  prefix_cache_doc()            tests/serving/prefix_cache_check.py's first-turn document
  haystack_docs()               the docs part of tests/serving/long_ctx_test.py's haystack, as (file name, text)
"""
import pathlib

CORPUS = pathlib.Path(__file__).resolve().parent / "corpus"


def read(rel: str) -> str:
    """Text of a frozen corpus file (path as it was in the repo, e.g. "docs/FORMAT.md")."""
    return (CORPUS / rel).read_text(encoding="utf-8")


def chat(u: str) -> str:
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


_SHORT = [
    "Write a Python function that checks whether a number is prime, then explain its complexity in two sentences.",
    "Поясни коротко, чому небо блакитне.",
    "A train leaves at 14:35 and arrives at 18:10. How long is the trip in minutes? Answer with reasoning.",
]
PROMPTS = [chat(u) for u in _SHORT] + [chat("Summarize this document in three sentences.\n\n" + read("docs/FORMAT.md"))]
N_PREDICT = 96
TOP_K = 5


def serving_long_doc() -> str:
    return read("docs/FORMAT.md") + read("docs/RESULTS.md")


def serving_prompts() -> list:
    return [chat(u) for u in _SHORT] + [chat("Summarize this document in three sentences.\n\n" + serving_long_doc()[:6000])]


def prefill_doc() -> str:
    return "".join(read(f) for f in ["docs/FORMAT.md", "docs/RESULTS.md", "README.md"])


def prefix_cache_doc() -> str:
    return read("docs/FORMAT.md")[:3000]


def haystack_docs() -> list:
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted((CORPUS / "docs").glob("*.md"))]


def inputs() -> dict:
    """Every frozen input by name (what --sha hashes)."""
    d = {f"PROMPTS[{i}]": p for i, p in enumerate(PROMPTS)}
    d.update({f"serving_prompts[{i}]": p for i, p in enumerate(serving_prompts())})
    d.update(serving_long_doc=serving_long_doc(), prefill_doc=prefill_doc(), prefix_cache_doc=prefix_cache_doc())
    d.update({f"haystack_docs[{n}]": t for n, t in haystack_docs()})
    return d
