#!/usr/bin/env python3
"""Print '<file> <byte offset> <nbytes> <dtype> <shape...>' for tensors of a safetensors HF snapshot.

The GEMV microbenchmarks read raw BF16 rows straight from the checkpoint:
    python3 locate_tensor.py $SNAP model.language_model.layers.0.mlp.experts.gate_up_proj
    ./hufpair_gemv <file> <offset> 2048 0 8192 1 16      # 8 experts of gate_up (K=2048), R=16
"""
import json
import struct
import sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
snap = sys.argv[1]
weight_map = json.load(open(snap + "/model.safetensors.index.json"))["weight_map"]
for name in sys.argv[2:]:
    path = snap + "/" + weight_map[name]
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    t = header[name]
    a, b = t["data_offsets"]
    print(path, 8 + n + a, b - a, t["dtype"], *t["shape"])
