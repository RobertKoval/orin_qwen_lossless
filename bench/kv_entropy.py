#!/usr/bin/env python3
"""Lossless-compressibility of a real F16 KV cache.

Input: a llama-server slot file (POST /slots/0?action=save) of a single-sequence, flash-attention (non-transposed V)
context. Every K and V row is split into sign (1 bit), exponent (5 bit) and mantissa (10 bit); the script reports the
empirical entropy of each field under several coding contexts, the stability of tables built on a prefix, and lzma as a
generic baseline.

    python3 bench/kv_entropy.py kv32768.bin [--head-dim 256] [--rope-dims 64]
"""
import argparse
import heapq
import lzma
import struct

import numpy as np

F16 = 1  # ggml type id


def H(counts):
    """Plug-in entropy in bits of a histogram (any shape, flattened)."""
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def H_rows(counts):
    """Conditional entropy H(X | row) for a 2-D histogram [row, symbol], weighted by row mass."""
    c = counts.astype(np.float64)
    n = c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(c > 0, np.log2(c / np.maximum(n, 1)), 0.0)
    return float(-(c * lp).sum() / c.sum())


def huffman_len(counts):
    """Average code length (bits/symbol) of an optimal Huffman code for one histogram."""
    c = [int(x) for x in counts if x > 0]
    if len(c) == 1:
        return 1.0
    heap = [(x, i, 0) for i, x in enumerate(c)]
    heapq.heapify(heap)
    total, uid = 0, len(c)
    while len(heap) > 1:
        a, _, _ = heapq.heappop(heap)
        b, _, _ = heapq.heappop(heap)
        total += a + b  # every merge adds one bit to all symbols below it
        heapq.heappush(heap, (a + b, uid, 0))
        uid += 1
    return total / sum(c)


def parse(path):
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    magic, version, n_tok = struct.unpack_from("<III", buf, 0)
    off = 12 + 4 * n_tok
    n_stream, cell_count = struct.unpack_from("<II", buf, off)
    assert n_stream == 1, n_stream
    meta = off + 8
    # data section: v_trans, n_layer, then per layer (k_type, k_size_row, cells x row)
    hdr = bytes(buf[meta:meta + 64 * cell_count + 4096])
    for v_trans in (0, 1):
        for n_layer in range(1, 129):
            for row in (512, 1024, 2048, 4096):
                pat = struct.pack("<IIiQ", v_trans, n_layer, F16, row)
                i = hdr.find(pat)
                if i >= 0 and i % cell_count == 0:
                    data = meta + i
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        raise SystemExit("data header not found")
    assert v_trans == 0, "transposed V (run with flash attention)"
    meta_per_cell = (data - meta) // cell_count
    off = data + 8
    layers = []
    for kind in ("K", "V"):
        for il in range(n_layer):
            t, row = struct.unpack_from("<iQ", buf, off)
            assert t == F16, (kind, il, t)
            off += 12
            a = np.frombuffer(buf, dtype=np.uint16, count=cell_count * row // 2, offset=off).reshape(cell_count, row // 2)
            off += cell_count * row
            layers.append((kind, il, a))
    return dict(version=version, n_tok=n_tok, cells=cell_count, n_layer=n_layer, meta_per_cell=meta_per_cell,
                kv_bytes=off - data, rest_bytes=len(buf) - off), layers


def analyse(a, head_dim, rope_dims, prefix):
    T, C = a.shape
    x = np.ascontiguousarray(a)
    s = (x >> 15).astype(np.int64)
    e = ((x >> 10) & 31).astype(np.int64)
    m = (x & 1023).astype(np.int64)
    ch = np.broadcast_to(np.arange(C), (T, C))
    r = {}
    r["zero_sub"] = float((e == 0).mean())
    r["infnan"] = float((e == 31).mean())
    r["H_e"] = H(np.bincount(e.ravel(), minlength=32))
    he_ch = np.bincount((ch * 32 + e).ravel(), minlength=C * 32).reshape(C, 32)
    r["H_e|ch"] = H_rows(he_ch)
    r["huf_e|ch"] = float(np.mean([huffman_len(row) for row in he_ch]))
    # previous token, same channel
    jp = np.bincount(((ch[1:] * 32 + e[:-1]) * 32 + e[1:]).ravel(), minlength=C * 1024).reshape(C * 32, 32)
    r["H_e|ch,prev"] = H_rows(jp)
    # neighbouring channel pairs (2i, 2i+1) within a token row, as BF16N codes weights
    pr = e[:, 0::2] * 32 + e[:, 1::2]
    r["H_pair/2"] = H(np.bincount(pr.ravel(), minlength=1024)) / 2
    cp = np.broadcast_to(np.arange(C // 2), (T, C // 2))
    r["H_pair|ch/2"] = H_rows(np.bincount((cp * 1024 + pr).ravel(), minlength=C // 2 * 1024).reshape(C // 2, 1024)) / 2
    r["H_s"] = H(np.bincount(s.ravel(), minlength=2))
    r["H_s|ch"] = H_rows(np.bincount((ch * 2 + s).ravel(), minlength=C * 2).reshape(C, 2))
    r["H_m"] = H(np.bincount(m.ravel(), minlength=1024))
    r["H_m|ch"] = H_rows(np.bincount((ch * 1024 + m).ravel(), minlength=C * 1024).reshape(C, 1024))
    # top 3 mantissa bits given exponent and channel
    m3 = m >> 7
    r["H_m3|e,ch"] = H_rows(np.bincount(((ch * 32 + e) * 8 + m3).ravel(), minlength=C * 256).reshape(C * 32, 8))
    # rotated vs. pass-through dims (partial RoPE rotates the first rope_dims of each head)
    d = np.arange(C) % head_dim
    rot = d < rope_dims
    r["H_e|ch rope"] = H_rows(he_ch[rot])
    r["H_e|ch pass"] = H_rows(he_ch[~rot])
    # static per-channel tables built on the first `prefix` tokens, applied to the rest (add-1/2 smoothing)
    q = np.bincount((ch[:prefix] * 32 + e[:prefix]).ravel(), minlength=C * 32).reshape(C, 32) + 0.5
    q = q / q.sum(1, keepdims=True)
    rest = (ch[prefix:] * 32 + e[prefix:]).ravel()
    r["xH_e|ch prefix"] = float(-np.log2(q.ravel()[rest]).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot_file")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--rope-dims", type=int, default=64)
    ap.add_argument("--prefix", type=int, default=4096)
    ap.add_argument("--lzma-layer", type=int, default=4)
    args = ap.parse_args()
    info, layers = parse(args.slot_file)
    print(info)
    keys = ["zero_sub", "H_e", "H_e|ch", "huf_e|ch", "H_e|ch,prev", "H_pair/2", "H_pair|ch/2", "H_s", "H_s|ch", "H_m",
            "H_m|ch", "H_m3|e,ch", "H_e|ch rope", "H_e|ch pass", "xH_e|ch prefix"]
    print("| tensor | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    agg = {"K": [], "V": []}
    for kind, il, a in layers:
        r = analyse(a, args.head_dim, args.rope_dims, args.prefix)
        agg[kind].append(r)
        print(f"| {kind}{il} | " + " | ".join(f"{r[k]:.4f}" if k == "zero_sub" else f"{r[k]:.3f}" for k in keys) + " |",
              flush=True)
    print()
    for kind, rs in agg.items():
        mean = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        # bits per value for coding schemes (sign + mantissa raw unless stated)
        schemes = {
            "exponent, one table per tensor": 11 + mean["H_e"],
            "exponent pairs, one table per tensor": 11 + mean["H_pair/2"],
            "exponent, one table per channel": 11 + mean["H_e|ch"],
            "  same, Huffman per channel (real code)": 11 + mean["huf_e|ch"],
            "  same, tables from the first prefix tokens": 11 + mean["xH_e|ch prefix"],
            "exponent | channel, previous token": 11 + mean["H_e|ch,prev"],
            "exponent + sign | channel": 10 + mean["H_e|ch"] + mean["H_s|ch"],
            "entropy bound: s|ch + e|ch,prev + m|ch": mean["H_s|ch"] + mean["H_e|ch,prev"] + mean["H_m|ch"],
        }
        print(f"{kind}: mean over layers: " + ", ".join(f"{k} {v:.3f}" for k, v in mean.items()))
        for name, bits in schemes.items():
            print(f"  {kind} {name:45s} {bits:6.3f} bit/value  ratio {bits / 16:.3f}")
    for kind, il, a in layers:
        if il != args.lzma_layer:
            continue
        raw = np.ascontiguousarray(a).tobytes()
        planes = np.ascontiguousarray(a).view(np.uint8).reshape(-1, 2).T.copy().tobytes()
        print(f"lzma {kind}{il}: raw ratio {len(lzma.compress(raw, preset=6)) / len(raw):.3f}, "
              f"byte planes ratio {len(lzma.compress(planes, preset=6)) / len(planes):.3f}", flush=True)


if __name__ == "__main__":
    main()
