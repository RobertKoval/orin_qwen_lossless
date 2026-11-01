import json,struct,sys,numpy as np
# usage: python3 exponent_coverage.py <HF snapshot dir of Qwen/Qwen3.6-35B-A3B>
# Per-row escape rates of exponent coding variants (16-wide window, ALT+15 window, best-16 LUT) on sampled tensors.
if len(sys.argv) < 2:
    sys.exit("usage: exponent_coverage.py <HF snapshot dir>")
S=sys.argv[1]
m=json.load(open(S+"/model.safetensors.index.json"))["weight_map"]
def load(name, max_rows=4096):
    f=S+"/"+m[name]
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
    t=h[name]; a,b=t["data_offsets"]; shp=t["shape"]; K=shp[-1]; rows=int(np.prod(shp[:-1]))
    rs=np.linspace(0,rows-1,min(rows,max_rows)).astype(np.int64)
    mm=np.memmap(f,dtype=np.uint16,mode="r",offset=8+n+a,shape=(rows,K))
    return np.asarray(mm[rs]),shp
def cov(e):
    # e: rows x K exponents; per-row coverage for window16, alt+window15, best16 set
    w16=a15=b16=0; tot=e.size
    for r in e:
        h=np.bincount(r,minlength=256).astype(np.int64)
        c=np.cumsum(np.concatenate([[0],h]))
        win16=(c[16:]-c[:-16]).max(); w16+=win16
        win15=c[15:]-c[:-15]  # window start s covers s..s+14
        best=0
        alt=np.argsort(h)[::-1][:3]
        for A in alt:
            hh=h.copy(); hh[A]=0; cc=np.cumsum(np.concatenate([[0],hh])); best=max(best,h[A]+(cc[15:]-cc[:-15]).max())
        a15+=max(best,win16)
        b16+=np.sort(h)[::-1][:16].sum()
    return 1-w16/tot,1-a15/tot,1-b16/tot
names=[]
for L in (0,3,10,20,39):
    p=f"model.language_model.layers.{L}."
    names+= [p+"mlp.experts.gate_up_proj",p+"mlp.experts.down_proj",p+"mlp.shared_expert.down_proj.weight",p+"mlp.shared_expert.gate_proj.weight",p+"mlp.gate.weight"]
    names+= [p+"self_attn.q_proj.weight",p+"self_attn.o_proj.weight",p+"self_attn.k_proj.weight"] if (L+1)%4==0 else [p+"linear_attn.in_proj_qkv.weight",p+"linear_attn.in_proj_z.weight",p+"linear_attn.out_proj.weight"]
names+=["lm_head.weight","model.language_model.embed_tokens.weight"]
for nm in names:
    if nm not in m: print("missing",nm); continue
    a,shp=load(nm); e=((a>>7)&0xFF).astype(np.int64)
    w16,a15,b16=cov(e)
    print(f"{nm[21:]:55s} {str(shp):18s} esc: win16={w16:.2e} alt+win15={a15:.2e} best16={b16:.2e}  exp3frac={(e==3).mean():.3f}",flush=True)
