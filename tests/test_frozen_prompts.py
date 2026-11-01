"""The benchmark / correctness inputs are those of the published numbers: bench/prompts/inputs.sha256 was produced
from the frozen corpus after proving equality with the pre-monorepo clients (which read the live docs). Pure, no GPU."""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.prompts.prompts import inputs  # noqa: E402


def test_inputs_equal_the_published_ones():
    want = dict(reversed(line.split(" ", 1)) for line in (ROOT / "bench/prompts/inputs.sha256").read_text().splitlines())
    got = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in inputs().items()}
    assert got == want
