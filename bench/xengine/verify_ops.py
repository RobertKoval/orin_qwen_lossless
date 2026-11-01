#!/usr/bin/env python3
"""Exact old/new llama CUDA BF16N outputs. Compiles only into --out; never builds/mutates either llama tree.

Use the same source/build for both sides to establish old-old reproducibility first.
Every graph runs in a fresh process (the production converter retains auxiliary buffers).
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    tensor: str
    op: str
    tokens: int
    fmt: int
    routing: str = "uniform"
    variant: str = "inherited"

    @property
    def name(self):
        return f"{self.tensor}-{self.op}-t{self.tokens}-f{self.fmt}-{self.routing}-{self.variant}"


def cases():
    """M1 instantiations + both representative blocks, dispatch boundaries and batched T=4096."""
    result = []
    for block in (0, 20):
        prefix = f"blk.{block}."
        for tensor in ("attn_qkv", "attn_gate", "ssm_out"):
            # NC=1/2/3/4, remainder group, and below/at/above the decode cutoff.
            for t in (1, 2, 3, 4, 5, 7, 8, 9):
                result.append(Case(prefix + tensor + ".weight", "mul_mat", t, 2))
        for tensor in ("ssm_alpha", "ffn_down_shexp", "ffn_gate_shexp"):
            result.append(Case(prefix + tensor + ".weight", "mul_mat", 1, 1))
        result.append(Case(prefix + "ffn_gate_up_shexp.weight", "glu", 1, 1))
        for tensor, op in (("ffn_gate_exps", "mul_mat_id"), ("ffn_down_exps", "mul_mat_id"),
                           ("ffn_gate_up_exps", "glu")):
            for t in (1, 2, 7, 8, 9, 512, 2048, 4096):
                for routing in (("uniform", "concentrated") if t > 8 else ("uniform",)):
                    result.append(Case(prefix + tensor + ".weight", op, t, 2, routing))
    # Metadata audited against the actual GGUF: regular attention is absent from blocks 0/20.
    for tensor, fmt in (("attn_q", 2), ("attn_output", 2), ("attn_k", 1), ("attn_v", 1)):
        for t in (1, 2, 3, 4, 5, 7, 8, 9):
            result.append(Case(f"blk.3.{tensor}.weight", "mul_mat", t, fmt))
    for block in (0, 20):
        for tensor in ("ffn_gate_inp", "ffn_gate_inp_shexp"):
            result.append(Case(f"blk.{block}.{tensor}.weight", "mul_mat", 1, 1))
        # Real down experts: E=256, used=8; interleave boundaries T=768 and 1536.
        for t in (767, 768, 769, 1535, 1536, 1537):
            result.append(Case(f"blk.{block}.ffn_down_exps.weight", "mul_mat_id", t, 2))
    for t in (1, 2, 3, 4, 5, 7, 8):
        result.append(Case("output.weight", "mul_mat", t, 2))
    for fmt in (1, 2):
        result.append(Case("token_embd.weight", "get_rows", 9, fmt))
        result.append(Case("output.weight", "get_rows", 9, fmt))
        for tensor in ("token_embd.weight", "output.weight"):
            result.append(Case(tensor, "get_rows_view_first", 1, fmt))
            result.append(Case(tensor, "get_rows_view_last", 1, fmt))
    # Opt-in paths are reported separately from inherited production flags. No inherited flags are cleared.
    for tensor in ("attn_qkv", "ssm_out"):
        for t in (2, 3, 4, 5):
            result.append(Case(f"blk.0.{tensor}.weight", "mul_mat", t, 2, variant="no_multi_col"))
    for t in (1, 9, 512):
        result.append(Case("blk.0.ffn_gate_up_exps.weight", "glu", t, 2, variant="no_glu"))
    result.append(Case("blk.0.ffn_gate_up_shexp.weight", "glu", 1, 2, variant="hp_glu"))
    for tensor in ("ffn_gate_exps", "ffn_down_exps"):
        result.append(Case(f"blk.0.{tensor}.weight", "mul_mat_id", 9, 2, variant="no_mma"))
        result.append(Case(f"blk.0.{tensor}.weight", "mul_mat_id", 9, 2, variant="reference"))
    result.append(Case("blk.0.attn_qkv.weight", "mul_mat", 1, 2, variant="reference"))
    return result


VARIANTS = {"inherited": {}, "no_multi_col": {"GGML_BF16N_NO_MULTI_COL": "1"},
            "no_glu": {"GGML_BF16N_NO_GLU": "1"}, "hp_glu": {"GGML_BF16N_HP_GLU": "1"},
            "no_mma": {"GGML_BF16N_NO_MMA": "1"}, "reference": {"GGML_BF16N_REFERENCE": "1"}}


def compare_dumps(a: Path, b: Path):
    """Return SHA256 only after exact comparison; report the first different F32 word."""
    size = a.stat().st_size
    if size == 0 or size % 4 or b.stat().st_size != size:
        raise ValueError(f"invalid/unequal dump sizes: {size} vs {b.stat().st_size}")
    sha = hashlib.sha256()
    with a.open("rb") as fa, b.open("rb") as fb:
        offset = 0
        while x := fa.read(1024 * 1024):
            y = fb.read(len(x))
            if x != y:
                byte = next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)
                word = byte // 4 * 4
                u, v = x[word:word + 4], y[word:word + 4]
                raise ValueError(f"F32[{(offset + word) // 4}]: 0x{u[::-1].hex()} vs 0x{v[::-1].hex()} "
                                 f"({struct.unpack('<f', u)[0]!r} vs {struct.unpack('<f', v)[0]!r})")
            sha.update(x)
            offset += len(x)
    return sha.hexdigest()


def compile_runner(source, build, output, cxx):
    include = source / "ggml/include"
    lib = build / "bin"
    if not (include / "ggml.h").is_file() or not lib.is_dir():
        raise ValueError(f"expected source/ggml/include and build/bin: {source}, {build}")
    command = shlex.split(cxx) + ["-O2", "-std=c++17", "-I", str(include), str(ROOT / "bench/xengine/dump_ops.cpp"),
                                 "-o", str(output), "-L", str(lib), "-lggml", "-lggml-base", "-lggml-cuda",
                                 f"-Wl,-rpath,{lib}"]
    subprocess.run(command, check=True)
    return {"source": str(source), "build": str(build), "compile": command}


def run(args):
    selected = [c for c in cases() if not args.case or re.search(args.case, c.name)]
    if args.list:
        print("\n".join(c.name for c in selected))
        return
    if not selected:
        raise ValueError("no cases selected")
    for key in ("old_source", "old_build", "new_source", "new_build", "model", "out"):
        if getattr(args, key) is None:
            raise ValueError(f"--{key.replace('_', '-')} is required")
    model = args.model.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)  # stale results must never pass a rerun
    manifest = {"schema": 1, "comparison": "bitwise-f32", "partial": bool(args.case),
                "model": {"path": str(model), "bytes": model.stat().st_size, "mtime_ns": model.stat().st_mtime_ns},
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("GGML_", "LLAMA_", "CUDA_", "BF16N_"))},
                "sides": {}, "cases": [], "passed": False}
    try:
        for side in ("old", "new"):
            source = getattr(args, side + "_source").resolve(strict=True)
            build = getattr(args, side + "_build").resolve(strict=True)
            manifest["sides"][side] = compile_runner(source, build, out / (side + "-dump"), args.cxx)
        for case in selected:
            metadata = []
            dumps = []
            for side in ("old", "new"):
                dump = out / (side + "-" + case.name + ".f32")
                log = out / (side + "-" + case.name + ".log")
                env = {**os.environ, **VARIANTS[case.variant]}
                lib = str(Path(manifest["sides"][side]["build"]) / "bin")
                env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                command = [str(out / (side + "-dump")), str(model), case.tensor, case.op, str(case.tokens),
                           str(case.fmt), case.routing, str(dump), str(args.device + 1)]
                with log.open("w") as stderr:
                    completed = subprocess.run(command, check=True, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
                meta = json.loads(completed.stdout)
                if meta["bytes"] != dump.stat().st_size or meta["bytes"] != 4 * math.prod(meta["shape"]):
                    raise ValueError(f"{case.name}: invalid output metadata")
                metadata.append(meta)
                dumps.append(dump)
            if metadata[0] != metadata[1]:
                raise ValueError(f"{case.name}: unequal output metadata: {metadata}")
            digest = compare_dumps(*dumps)
            manifest["cases"].append({**asdict(case), **metadata[0], "environment_overlay": VARIANTS[case.variant], "sha256": digest})
            if not args.keep_dumps:
                for dump in dumps:
                    dump.unlink()
            print(f"EXACT {case.name} {digest}", flush=True)
        if model.stat().st_size != manifest["model"]["bytes"] or model.stat().st_mtime_ns != manifest["model"]["mtime_ns"]:
            raise ValueError("model changed during verification")
        manifest["passed"] = True
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{'PARTIAL' if args.case else 'FULL'} PASS: {len(selected)} exact cases; {out / 'manifest.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("old-source", "old-build", "new-source", "new-build", "model", "out"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    p.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default 0)")
    p.add_argument("--case", help="regex subset; result explicitly marked PARTIAL")
    p.add_argument("--list", action="store_true", help="list cases without building or accessing GPU/model")
    p.add_argument("--keep-dumps", action="store_true", help="retain matching dumps too (default keeps only mismatches)")
    args = p.parse_args()
    if args.device < 0:
        p.error("--device must be nonnegative")
    try:
        run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"verify-xengine: FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
