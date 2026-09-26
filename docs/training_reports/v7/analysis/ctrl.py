import sys, itertools, json, numpy as np, re, glob
sys.path.insert(0,"/workspace/Pudgy/finetune/wan")
import gates_v6 as g
from concurrent.futures import ProcessPoolExecutor
LABELS=["happy","surprised","angry","confused","crying","neutral","laughing"]; SEEDS=[42,43,44]
fb=g.face_box(1.00)
def clip(step,ch,l,s): return f"clips/refine__step{step:08d}__{ch}_{l}_s{s}.mp4"
_cache={}
def vid(p):
    if p not in _cache: _cache[p]=g.read_video(p)
    return _cache[p]
def pair(args):
    a,b=args; return 1-g.ssim_pair(vid(a),vid(b),region=fb)
res={}
for step in (500,2500):
  for ch in ("pax","polly"):
    items=[(l,s) for l in LABELS for s in SEEDS]
    prs=list(itertools.combinations(range(len(items)),2))
    with ProcessPoolExecutor(32) as ex:
        ds=list(ex.map(pair,[(clip(step,ch,*items[i]),clip(step,ch,*items[j])) for i,j in prs],chunksize=4))
    D=np.zeros((21,21))
    for (i,j),d in zip(prs,ds): D[i,j]=D[j,i]=d
    res[(step,ch)]=(items,D)
np.save("ctrl_D.npy",{str(k):v[1] for k,v in res.items()},allow_pickle=True)
rng=np.random.default_rng(0)
for (step,ch),(items,D) in res.items():
    L=np.array([i[0] for i in items]); Sd=np.array([i[1] for i in items])
    iu=np.triu_indices(21,1)
    sameL=(L[:,None]==L[None,:])[iu]; sameS=(Sd[:,None]==Sd[None,:])[iu]; d=D[iu]
    cat=lambda a,b: d[(sameL==a)&(sameS==b)]
    bl_ss, wl_ds, bl_ds = cat(False,True), cat(True,False), cat(False,False)
    # 1-NN leave-one-out
    Dn=D+np.eye(21)*9; nn=Dn.argmin(1)
    accL=(L[nn]==L).mean(); accS=(Sd[nn]==Sd).mean()
    # label-effect vs seed-effect (additive model on distances)
    eff_label=bl_ds.mean()-wl_ds.mean(); eff_seed=bl_ds.mean()-bl_ss.mean()
    # permutation test on eff_label: shuffle label assignment within each seed
    null=[]
    for _ in range(2000):
        Lp=L.copy()
        for s in SEEDS:
            m=Sd==s; Lp[m]=rng.permutation(Lp[m])
        sl=(Lp[:,None]==Lp[None,:])[iu]
        null.append(d[(~sl)&(~sameS)].mean()-d[sl&(~sameS)].mean())
    p=(np.sum(np.array(null)>=eff_label)+1)/2001
    print(f"step{step} {ch:5s}| 1-SSIM: diffLabel/sameSeed {bl_ss.mean():.4f} | sameLabel/diffSeed {wl_ds.mean():.4f} | diffLabel/diffSeed {bl_ds.mean():.4f}"
          f" || label effect {eff_label:+.4f} (perm p={p:.4f})  seed effect {eff_seed:+.4f} || 1-NN acc label {accL:.2f} (chance .14) seed {accS:.2f} (chance .33)")
