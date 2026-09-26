import math, re, collections, json
import numpy as np
from lora_geom import *
E="v7__weights__expr-lownoise__pudgy-v7-expr-lownoise-step%08d.safetensors"
M="v7__weights__motion-lownoise__pudgy-v7-motion-lownoise-step%08d.safetensors"
G=load(f("v2__weights__curated__lora_lownoise_GOLDEN_ep40.safetensors"))
GH=load(f("v2__weights__curated__lora_highnoise_GOLDEN_ep40.safetensors"))
es=[250,500,1000,1500,2000,2500,3000,3500]; ms=[250,500,1000,1500,2000]
Ek={s:load(f(E%s)) for s in es}; Ek[3600]=load(f("v7__weights__expr-lownoise__pudgy-v7-expr-lownoise.safetensors"))
Mk={s:load(f(M%s)) for s in ms}; Mk[2144]=load(f("v7__weights__motion-lownoise__pudgy-v7-motion-lownoise.safetensors"))
MH=load(f("v7__weights__motion-highnoise__pudgy-v7-motion-highnoise.safetensors"))
MH500=load(f("v7__weights__motion-highnoise__pudgy-v7-motion-highnoise-step00000500.safetensors"))
names=sorted(G)
def mtype(n): return re.sub(r"lora_unet_blocks_\d+_","",n)
def blk(n): return int(re.search(r"blocks_(\d+)_",n).group(1))
def gl(fn):  # sum over modules of a per-module quantity
    return sum(fn(n) for n in names)
single=lambda L,n:[L[n]]
out={}
# ---------- A. norms ----------
nG=math.sqrt(gl(lambda n: ip(single(G,n),single(G,n))))
nGH=math.sqrt(gl(lambda n: ip(single(GH,n),single(GH,n))))
print(f"||G_low||={nG:.4f}  ||G_high||={nGH:.4f}")
def tnorm(L,G0): return math.sqrt(gl(lambda n: ip(tau(L,G0,n),tau(L,G0,n))))
en={s:tnorm(Ek[s],G) for s in Ek}; mn={s:tnorm(Mk[s],G) for s in Mk}
print("expr  ||tau(t)|| / ||G||:", {s:round(v/nG,4) for s,v in en.items()})
print("motL  ||tau(t)|| / ||G||:", {s:round(v/nG,4) for s,v in mn.items()})
mh=tnorm(MH,GH); mh5=tnorm(MH500,GH)
print(f"motH  ||tau||/||G_high||: step500 {mh5/nGH:.4f}  final {mh/nGH:.4f}")
# growth exponent  ||tau(t)|| ~ t^p
for lab,d in (("expr",en),("motL",mn)):
    t=np.array(sorted(d)); y=np.array([d[s] for s in t]); p=np.polyfit(np.log(t),np.log(y),1)[0]
    print(f"{lab}: log-log slope p={p:.3f}  (p=1 drift, p=0.5 random walk)")
# ---------- B. direction consistency ----------
def gcos(X,Y):  # X,Y: functions n-> lowrank list
    num=gl(lambda n: ip(X(n),Y(n))); a=gl(lambda n: ip(X(n),X(n))); b=gl(lambda n: ip(Y(n),Y(n)))
    return num/math.sqrt(a*b)
tE=lambda s:(lambda n: tau(Ek[s],G,n))
print("cos(tau_e(t), tau_e(3600)):", {s:round(gcos(tE(s),tE(3600)),4) for s in es})
# increments  d_k = L_k - L_{k-1}
def inc(L1,L0): return lambda n: [L1[n], (-L0[n][0],L0[n][1],L0[n][2])]
seq=[500,1000,1500,2000,2500,3000,3500]
incs=[inc(Ek[b],Ek[a]) for a,b in zip(seq,seq[1:])]
print("cos(consecutive 500-step expr increments):",[round(gcos(incs[i],incs[i+1]),4) for i in range(len(incs)-1)])
print("cos(inc 500->1000, inc 3000->3500):", round(gcos(incs[0],incs[-1]),4))
# ---------- C. expression vs motion task vectors ----------
te=tE(2500); tm=lambda n: tau(Mk[2144],G,n)
print(f"GLOBAL cos(tau_expr2500, tau_motion_final) = {gcos(te,tm):.4f}")
bytype=collections.defaultdict(lambda:[0,0,0])
for n in names:
    a,b=te(n),tm(n); t=bytype[mtype(n)]; t[0]+=ip(a,b); t[1]+=ip(a,a); t[2]+=ip(b,b)
print("cos by module type:", {k:round(v[0]/math.sqrt(v[1]*v[2]),3) for k,v in sorted(bytype.items())})
# per-module cos distribution + subspace overlap (top-8 left & right singular spaces)
cs=[];ovL=[];ovR=[]
for n in names:
    a,b=te(n),tm(n); cs.append(cos(a,b))
    Ua,Ub=left_basis(a,8),left_basis(b,8); ovL.append((torch.linalg.svdvals(Ua.T@Ub)**2).mean().item())
    Va,Vb=right_basis(a,8),right_basis(b,8); ovR.append((torch.linalg.svdvals(Va.T@Vb)**2).mean().item())
cs=np.array(cs); print(f"per-module cos: mean {cs.mean():.3f} median {np.median(cs):.3f} p10 {np.percentile(cs,10):.3f} p90 {np.percentile(cs,90):.3f}")
dout=5120; print(f"top-8 subspace overlap  left(out) {np.mean(ovL):.3f}  right(in) {np.mean(ovR):.3f}   (random 8-dim in 5120-dim ~ {8/5120:.4f})")
# ---------- D. double-counting when both files are stacked ----------
# stacked: W + L_e + L_m = W + 2G + te + tm ; intended: W + G + te + tm ; error term = G
num=gl(lambda n: ip(single(G,n),single(G,n)))
intended=gl(lambda n: ip([G[n]]+te(n)+tm(n),[G[n]]+te(n)+tm(n)))
nte=math.sqrt(gl(lambda n: ip(te(n),te(n)))); ntm=math.sqrt(gl(lambda n: ip(tm(n),tm(n))))
print(f"stacking error ||G||={nG:.4f} vs ||tau_e||={nte:.4f} ||tau_m||={ntm:.4f};  ||G||/||intended delta|| = {nG/math.sqrt(intended):.3f}")
# ---------- E. spectrum of task vectors & relation to golden subspace ----------
def eff_rank(sv):
    p=(sv**2)/(sv**2).sum(); return math.exp(-(p*torch.log(p+1e-30)).sum().item())
er_e=[];er_m=[];inG=[];e16=[]
for n in names:
    s=svals(te(n)); er_e.append(eff_rank(s)); e16.append(((s[:16]**2).sum()/(s**2).sum()).item())
    er_m.append(eff_rank(svals(tm(n))))
    Ug=left_basis(single(G,n),16); X=te(n); U=torch.cat([c*B for c,B,_ in X],1); V=torch.cat([A for _,_,A in X],0)
    P=Ug@(Ug.T@U)  # project column space onto golden's 16-dim output subspace
    inG.append(ip([(1.0,P,V)],[(1.0,P,V)])/ip(X,X))
print(f"eff. rank tau_e mean {np.mean(er_e):.2f}  tau_m {np.mean(er_m):.2f} (max 32); energy in top-16 {np.mean(e16):.3f}")
print(f"fraction of tau_e energy inside golden's output subspace: {np.mean(inG):.3f}  (random 16/5120 = {16/5120:.4f})")
# ---------- F. where did it move (by block depth / type), relative to golden ----------
rel=collections.defaultdict(list)
for n in names: rel[mtype(n)].append(nrm(te(n))/(nrm(single(G,n))+1e-12))
print("||tau_e||/||G|| by type:", {k:round(float(np.mean(v)),3) for k,v in sorted(rel.items())})
depth=np.zeros(40); 
for n in names: depth[blk(n)]+=ip(te(n),te(n))
depth/=depth.sum(); print("tau_e energy by block (quintiles):", [round(depth[i*8:(i+1)*8].sum(),3) for i in range(5)])
depthm=np.zeros(40)
for n in names: depthm[blk(n)]+=ip(tm(n),tm(n))
depthm/=depthm.sum(); print("tau_m energy by block (quintiles):", [round(depthm[i*8:(i+1)*8].sum(),3) for i in range(5)])
