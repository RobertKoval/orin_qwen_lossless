#!/usr/bin/env python3
"""Render the README figures as static SVG (light + dark) from the measured numbers in docs/RESULTS.md.

    python3 tools/make_charts.py   ->  docs/img/*.svg
"""
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
DATA = json.loads((Path(__file__).resolve().parent / "chart_data.json").read_text())

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", border="rgba(11,11,11,0.10)", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
                  pale="#86b6ef"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", border="rgba(255,255,255,0.10)", s1="#3987e5", s2="#d95926", s3="#199e70",
                 pale="#184f95"),
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"'


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_w(s, size):  # rough width estimate for layout (system sans averages ~0.55 em)
    return len(str(s)) * size * 0.56


def card(w, h, t, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT} role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="{t["surface"]}" stroke="{t["border"]}"/>',
        f'<text x="24" y="36" font-size="17" font-weight="600" fill="{t["ink"]}">{esc(title)}</text>',
        f'<text x="24" y="58" font-size="13" fill="{t["ink2"]}">{esc(subtitle)}</text>',
    ]


def hbar_path(x0, y, length, h, r=4):
    """Bar square at the baseline (x0), 4px rounded at the data end."""
    if length <= r:
        return f'<rect x="{x0}" y="{y}" width="{max(length, 0.5)}" height="{h}"/>'
    x1 = x0 + length
    return (f'<path d="M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 - r},{y + h} '
            f'H{x0} Z"/>')


def bar_chart(name, spec):
    rows = spec["rows"]
    w = 760
    label_w = max(text_w(r["label"], 13) for r in rows) + 36
    x0, x1 = 24 + label_w, w - 90
    band, bar_h = 40, 22
    top = 92
    h = top + band * len(rows) + 56
    vmax = spec["axis_max"]
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    for mode, t in THEMES.items():
        o = card(w, h, t, spec["title"], spec["subtitle"])
        y_axis = top + band * len(rows) + 4
        for tick in spec["ticks"]:  # hairline grid + tick labels
            x = sx(tick)
            o.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{y_axis}" stroke="{t["grid"]}" stroke-width="1"/>')
            o.append(f'<text x="{x:.1f}" y="{y_axis + 18}" font-size="11" fill="{t["muted"]}" text-anchor="middle">{tick:g}</text>')
        o.append(f'<line x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{y_axis}" stroke="{t["axis"]}" stroke-width="1"/>')
        o.append(f'<text x="{x1}" y="{y_axis + 38}" font-size="11" fill="{t["muted"]}" text-anchor="end">{esc(spec["unit"])}</text>')
        halo = f'stroke="{t["surface"]}" stroke-width="5" stroke-linejoin="round" paint-order="stroke"'
        if "reference" in spec:  # one labelled reference line (solid, recessive), under the labels
            ref = spec["reference"]
            xr = sx(ref["value"])
            o.append(f'<line x1="{xr:.1f}" y1="{top - 10}" x2="{xr:.1f}" y2="{y_axis}" stroke="{t["ink2"]}" stroke-width="1.5"/>')
            o.append(f'<text x="{xr - 6:.1f}" y="{top - 14}" font-size="12" fill="{t["ink2"]}" text-anchor="end">{esc(ref["label"])}</text>')
        for i, r in enumerate(rows):
            y = top + band * i + (band - bar_h) / 2
            fill = t["s1"] if r.get("highlight") else t["pale"]
            o.append(f'<g fill="{fill}">{hbar_path(x0, y, sx(r["value"]) - x0, bar_h)}</g>')
            o.append(f'<text x="{x0 - 12}" y="{y + bar_h / 2 + 4.5}" font-size="13" fill="{t["ink2"]}" text-anchor="end">{esc(r["label"])}</text>')
            # knock the reference line out behind the value label
            o.append(f'<rect x="{sx(r["value"]) + 3:.1f}" y="{y}" width="{text_w(r["value_label"], 13) + 8:.1f}" height="{bar_h}" fill="{t["surface"]}"/>')
            o.append(f'<text x="{sx(r["value"]) + 8:.1f}" y="{y + bar_h / 2 + 4.5}" font-size="13" font-weight="600" fill="{t["ink"]}" {halo}>{esc(r["value_label"])}</text>')
        o.append("</svg>")
        (OUT / f"{name}-{mode}.svg").write_text("\n".join(o) + "\n")


def line_chart(name, spec):
    pts = spec["points"]
    w, h = 760, 360
    x0, x1, top, bottom = 72, w - 40, 92, h - 64
    xs = [p[0] for p in pts]
    vmin, vmax = spec["y_min"], spec["y_max"]
    # log2 x scale over context length (0 is drawn at the first tick)
    import math
    lx = lambda c: math.log2(max(c, spec["x_floor"]))
    xa, xb = lx(spec["x_ticks"][0]), lx(spec["x_ticks"][-1])
    sx = lambda c: x0 + (x1 - x0) * (lx(c) - xa) / (xb - xa)
    sy = lambda v: bottom - (bottom - top) * (v - vmin) / (vmax - vmin)
    for mode, t in THEMES.items():
        o = card(w, h, t, spec["title"], spec["subtitle"])
        for v in spec["y_ticks"]:
            y = sy(v)
            o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
            o.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" font-size="11" fill="{t["muted"]}" text-anchor="end">{v:g}</text>')
        o.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
        for c, lab in zip(spec["x_ticks"], spec["x_tick_labels"]):
            o.append(f'<text x="{sx(c):.1f}" y="{bottom + 20}" font-size="11" fill="{t["muted"]}" text-anchor="middle">{esc(lab)}</text>')
        o.append(f'<text x="{x1}" y="{bottom + 42}" font-size="11" fill="{t["muted"]}" text-anchor="end">{esc(spec["x_unit"])}</text>')
        o.append(f'<text x="{x0 - 10}" y="{top - 14}" font-size="11" fill="{t["muted"]}" text-anchor="end">{esc(spec["y_unit"])}</text>')
        d = " ".join(("M" if i == 0 else "L") + f"{sx(c):.1f},{sy(v):.1f}" for i, (c, v) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{t["s1"]}" stroke-width="2" stroke-linejoin="round"/>')
        for c, v in pts:  # >= 8px markers with a surface ring
            o.append(f'<circle cx="{sx(c):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{t["s1"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        for c, v in (pts[0], pts[-1]):  # label the endpoints only
            anchor = "start" if c == pts[0][0] else "end"
            dx = 10 if anchor == "start" else -10
            o.append(f'<text x="{sx(c) + dx:.1f}" y="{sy(v) - 12:.1f}" font-size="13" font-weight="600" fill="{t["ink"]}" text-anchor="{anchor}" '
                     f'stroke="{t["surface"]}" stroke-width="5" stroke-linejoin="round" paint-order="stroke">{v:.1f}</text>')
        o.append("</svg>")
        (OUT / f"{name}-{mode}.svg").write_text("\n".join(o) + "\n")


def format_diagram(name):
    """BF16 bits -> sign|mantissa plane + exponent-pair Huffman stream -> decode in registers -> FMA."""
    w, h = 760, 292
    for mode, t in THEMES.items():
        o = card(w, h, t, "How a weight is stored and decoded",
                 "bit-exact: every BF16 pattern comes back; only the exponent is entropy-coded")

        def box(x, y, bw, bh, label, sub, stroke):
            o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{t["surface"]}" stroke="{stroke}" stroke-width="2"/>')
            o.append(f'<text x="{x + bw / 2}" y="{y + bh / 2 - 4}" font-size="13" font-weight="600" fill="{t["ink"]}" text-anchor="middle">{esc(label)}</text>')
            o.append(f'<text x="{x + bw / 2}" y="{y + bh / 2 + 14}" font-size="11.5" fill="{t["ink2"]}" text-anchor="middle">{esc(sub)}</text>')

        def arrow(xa, ya, xb, yb):
            o.append(f'<line x1="{xa}" y1="{ya}" x2="{xb - 7}" y2="{yb}" stroke="{t["muted"]}" stroke-width="1.5"/>')
            o.append(f'<path d="M{xb},{yb} l-8,-4 v8 Z" fill="{t["muted"]}"/>')

        y = 104
        o.append(f'<text x="24" y="{y - 10}" font-size="12" fill="{t["ink2"]}">BF16 weight, 16 bits</text>')
        x = 24
        for lab, bits, col in (("sign", 1, t["s2"]), ("exponent · 8", 8, t["s1"]), ("mantissa · 7", 7, t["s3"])):
            bw = bits * 12.5
            o.append(f'<rect x="{x:.1f}" y="{y}" width="{bw - 2:.1f}" height="26" rx="3" fill="{col}"/>')
            o.append(f'<text x="{x + (bw - 2) / 2:.1f}" y="{y + 44}" font-size="11.5" fill="{t["ink2"]}" text-anchor="middle">{"s" if bits == 1 else esc(lab)}</text>')
            x += bw
        box(296, 80, 214, 56, "sign | mantissa byte", "8 bit, stored raw", t["s3"])
        box(296, 156, 214, 56, "exponent pairs → Huffman", "≈ 2.7 bit/weight, per-lane streams", t["s1"])
        arrow(226, 112, 296, 108)
        arrow(226, 124, 296, 184)
        box(556, 112, 180, 68, "fused GEMV kernel", "LUT → bf16×2 → FP32 FMA", t["ink2"])
        arrow(510, 108, 556, 136)
        arrow(510, 184, 556, 158)
        o.append(f'<text x="24" y="{h - 44}" font-size="12" fill="{t["ink2"]}">10.7 bit/weight on average (ratio 0.671). Decoded in registers, never materialised as BF16 during decode.</text>')
        o.append(f'<text x="24" y="{h - 24}" font-size="12" fill="{t["ink2"]}">Every tensor is decoded back on the GPU and compared bit-for-bit when the model loads.</text>')
        o.append("</svg>")
        (OUT / f"{name}-{mode}.svg").write_text("\n".join(o) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bar_chart("decode", DATA["decode"])
    bar_chart("memory", DATA["memory"])
    bar_chart("prefill", DATA["prefill"])
    if "context" in DATA:
        line_chart("context", DATA["context"])
    format_diagram("format")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
