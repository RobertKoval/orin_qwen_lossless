"""python -m bench.prompts --sha: sha256 of every frozen benchmark input (run from the repo root)."""
import hashlib
import sys

from bench.prompts.prompts import inputs

if sys.argv[1:] != ["--sha"]:
    sys.exit("usage: python -m bench.prompts --sha")
for name, text in inputs().items():
    print(hashlib.sha256(text.encode("utf-8")).hexdigest(), name)
