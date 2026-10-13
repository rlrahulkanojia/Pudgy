# G0.1 eye-bar proxy, run from the session scratchpad on 2026-10-12 and checked in as-is
# afterwards (docs/REPORT_G01.md §3.2 / §6). Paths are the box paths of that run.
import json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as nd
from skimage.morphology import skeletonize
SRC=Path('/workspace/Data/processed/expv2_h3/vae_gate/png768'); DEC=Path('/workspace/expv2/eval/phase0/vae_gate/decoded')
lum=lambda a: a[...,:3].astype(np.float32)@np.array([.299,.587,.114],np.float32)
import importlib.util,sys
spec=importlib.util.spec_from_file_location('a','/tmp/claude-0/-workspace/2a1eb145-41c8-4647-9b23-cb1a0357fa81/scratchpad/g01_audit.py')
src_hl=None
def highlights(L):
    w=L>230; lab,n=nd.label(w); out=np.zeros_like(w)
    for i,s in enumerate(nd.sum(w,lab,range(1,n+1)),1):
        if 4<=s<=600:
            comp=lab==i; ring=nd.binary_dilation(comp,iterations=2)&~comp
            if (L[ring]<80).mean()>0.6: out|=comp
    return out
out={}
for clip in sorted(p.name for p in SRC.iterdir()):
    if clip.startswith('flatfill'): continue
    src=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in sorted((SRC/clip).glob('*.png'))])
    row={}
    for v in ('h3','wan21'):
        dec=np.load(DEC/v/'768'/f'{clip}.npz')['dec']; T=min(len(src),len(dec))
        cen,brk,hc,worst=[],[],[],[]
        for i in range(T):
            Ls,Ld=lum(src[i]),lum(dec[i]); o=Ls<64
            sk=skeletonize(o); dark=Ld<128
            cen.append(dark[sk].mean())
            # broken stroke: a run of >=6 consecutive skeleton px lost -> count gaps as components
            lost=sk&~dark; lab,n=nd.label(lost,structure=np.ones((3,3)))
            brk.append(int((nd.sum(lost,lab,range(1,n+1))>=6).sum()) if n else 0)
            h=highlights(Ls); hcore=nd.binary_erosion(h,iterations=1)
            if hcore.sum(): hc.append((Ld[hcore]>200).mean())
        row[v]={'centerline_min':round(float(min(cen)),4),'centerline_mean':round(float(np.mean(cen)),4),
                'broken_strokes_max_per_frame':max(brk),'worst_frame':int(np.argmin(cen)),
                'highlight_core_min':round(float(min(hc)),4) if hc else None}
    out[clip]=row; print(clip,row,flush=True)
json.dump(out,open('/workspace/expv2/eval/phase0/vae_gate/eye_proxy_centerline.json','w'),indent=1)
