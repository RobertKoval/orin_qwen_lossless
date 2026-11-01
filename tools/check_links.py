"""Offline link check: relative file links and #anchors (GitHub slugs) in tracked *.md files.
  tools/check_links.py <repo root>   (the frozen corpus in bench/prompts/corpus is skipped)"""
import pathlib, re, subprocess, sys, unicodedata
root = pathlib.Path(sys.argv[1]).resolve()
files = [root / f for f in subprocess.check_output(["git", "-C", str(root), "ls-files", "*.md"], text=True).split()
         if not f.startswith("bench/prompts/corpus/")]
def slugs(p):
    out, seen = set(), {}
    text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
    for m in re.finditer(r"^#{1,6}\s+(.*?)\s*$", text, re.M):
        h = re.sub(r"`|\*\*|\[(.*?)\]\(.*?\)", r"\1", m.group(1)).strip().lower()
        s = "".join(c for c in h if c.isalnum() or c in " -_" or unicodedata.category(c).startswith("L")).replace(" ", "-")
        n = seen.get(s, 0); seen[s] = n + 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out
bad = 0
for f in files:
    text = re.sub(r"```.*?```", "", f.read_text(), flags=re.S)
    for m in re.finditer(r"\]\(([^)\s]+)\)|srcset=\"([^\"]+)\"|src=\"([^\"]+)\"", text):
        link = next(g for g in m.groups() if g)
        if re.match(r"[a-z]+:", link): continue
        path, _, anchor = link.partition("#")
        tgt = (f.parent / path).resolve() if path else f
        if not tgt.exists():
            print(f"{f.relative_to(root)}: missing {link}"); bad += 1; continue
        if anchor and tgt.suffix == ".md" and anchor not in slugs(tgt):
            print(f"{f.relative_to(root)}: bad anchor {link}"); bad += 1
print(f"{len(files)} files, {bad} broken"); sys.exit(1 if bad else 0)
