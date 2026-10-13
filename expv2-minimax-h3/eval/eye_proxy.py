# G0.1 eye-bar proxy, run from the session scratchpad on 2026-10-12 and checked in as-is
# afterwards (docs/REPORT_G01.md §3.2 / §6). Paths are the box paths of that run.
import json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as nd
SRC=Path('/workspace/Data/processed/expv2_h3/vae_gate/png768'); DEC=Path('/workspace/expv2/eval/phase0/vae_gate/decoded')
lum=lambda a: a[...,:3].astype(np.float32)@np.array([.299,.587,.114],np.float32)
def psnr(a,b): m=np.mean((a.astype(np.float32)-b.astype(np.float32))**2); return 10*np.log10(255**2/max(m,1e-9))
def highlights(L):
    w=L>230; lab,n=nd.label(w); out=np.zeros_like(w)
    if n==0: return out
    sizes=nd.sum(w,lab,range(1,n+1))
    for i,s in enumerate(sizes,1):
        if 4<=s<=600:
            comp=lab==i; ring=nd.binary_dilation(comp,iterations=2)&~comp
            if (L[ring]<80).mean()>0.6: out|=comp
    return out
def grad(L): return np.hypot(*np.gradient(L))
res={}
for clip in sorted(p.name for p in SRC.iterdir()):
    if clip.startswith('flatfill'): continue
    src=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in sorted((SRC/clip).glob('*.png'))])
    row={}
    for v in ('h3','wan21','wan22_5b'):
        dec=np.load(DEC/v/'768'/f'{clip}.npz')['dec']; T=min(len(src),len(dec)); s,d=src[:T],dec[:T]
        al={k:np.mean([psnr(s[i],d[i+k]) for i in range(1,T-1)]) for k in (-1,0,1)}
        rec,hrec,hn,sh=[],[],0,[]
        for i in range(T):
            Ls,Ld=lum(s[i]),lum(d[i]); o=Ls<64
            rec.append((Ld[o]<128).mean())
            h=highlights(Ls); hn+=h.sum()
            if h.sum(): hrec.append((Ld[h]>200).mean())
            e=grad(Ls)>20; sh.append(grad(Ld)[e].mean()/grad(Ls)[e].mean())
        row[v]={'aligned':al[0]>max(al[-1],al[1]),'psnr_shift':{k:round(x,2) for k,x in al.items()},
                'outline_recall':round(float(np.min(rec)),4),'outline_recall_mean':round(float(np.mean(rec)),4),
                'highlight_recall':round(float(np.min(hrec)),4) if hrec else None,'highlight_px_per_frame':int(hn/T),
                'sharpness_retention':round(float(np.mean(sh)),3)}
    res[clip]=row
    print(clip, {v:(r['aligned'],r['outline_recall'],r['highlight_recall'],r['sharpness_retention']) for v,r in row.items()},flush=True)
h3o=[r['h3']['outline_recall'] for r in res.values()]; wo=[r['wan21']['outline_recall'] for r in res.values()]
h3h=[r['h3']['highlight_recall'] for r in res.values() if r['h3']['highlight_recall'] is not None]
wh=[r['wan21']['highlight_recall'] for r in res.values() if r['wan21']['highlight_recall'] is not None]
rule={'aligned_all':all(r[v]['aligned'] for r in res.values() for v in r),
      'outline_min_h3':min(h3o),'outline_min_wan21':min(wo),'highlight_min_h3':min(h3h),'highlight_min_wan21':min(wh),
      'outline_pass':min(h3o)>=0.95 and all(a>=b-0.03 for a,b in zip(h3o,wo)),
      'highlight_pass':min(h3h)>=0.95 and min(h3h)>=min(wh)-0.03}
rule['eye_proxy_pass']=rule['aligned_all'] and rule['outline_pass'] and rule['highlight_pass']
print(json.dumps(rule,indent=1))
json.dump({'rule':'outline & highlight recall >=0.95 every clip and within 0.03 of wan21; alignment shift 0 best','summary':rule,'clips':res},
          open('/workspace/expv2/eval/phase0/vae_gate/eye_proxy.json','w'),indent=1,default=float)
