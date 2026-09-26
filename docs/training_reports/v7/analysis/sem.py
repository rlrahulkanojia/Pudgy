import sys, numpy as np, torch
sys.path.insert(0,"/workspace/Pudgy/finetune/wan"); import gates_v6 as g
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
LABELS=["happy","surprised","angry","confused","crying","neutral","laughing"]; SEEDS=[42,43,44]
PH={"happy":"smiling happily","surprised":"surprised with wide eyes","angry":"angry and scowling","confused":"confused and puzzled",
    "crying":"crying with tears","neutral":"calm with a neutral face","laughing":"laughing with eyes squeezed shut"}
dev="cuda"
def T_(o): return o if torch.is_tensor(o) else o.pooler_output
clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval(); cp=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
dino=AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval(); dp=AutoImageProcessor.from_pretrained("facebook/dinov2-base")
y0,y1,x0,x1=g.face_box(1.0)
def frames(p):
    v=g.read_video(p); T=len(v); idx=np.linspace(T//2,T-1,6).astype(int)   # expression is at its peak in the back half
    return [v[i] for i in idx], [v[i][y0:y1,x0:x1] for i in idx]
@torch.no_grad()
def emb(imgs):
    c=T_(clip.get_image_features(**cp(images=imgs,return_tensors="pt").to(dev))); c=torch.nn.functional.normalize(c,dim=-1).mean(0)
    d=dino(**dp(images=imgs,return_tensors="pt").to(dev)).last_hidden_state[:,0]; d=torch.nn.functional.normalize(d,dim=-1).mean(0)
    return torch.nn.functional.normalize(c,dim=0), torch.nn.functional.normalize(d,dim=0)
for step in (500,2500):
  for ch in ("pax","polly"):
    items=[(l,s) for l in LABELS for s in SEEDS]; C=[];Dn=[]
    for l,s in items:
        full,face=frames(f"clips/refine__step{step:08d}__{ch}_{l}_s{s}.mp4"); c,d=emb(full); C.append(c); Dn.append(d)
    C=torch.stack(C); Dn=torch.stack(Dn); L=np.array([i[0] for i in items]); Sd=np.array([i[1] for i in items])
    name="Pax, a blue" if ch=="pax" else "Polly, a pink"
    txt=[f"a 2D cartoon of {name} penguin {PH[l]}" for l in LABELS]
    with torch.no_grad():
        t=T_(clip.get_text_features(**cp(text=txt,return_tensors="pt",padding=True).to(dev))); t=torch.nn.functional.normalize(t,dim=-1)
    zs=(C@t.T).argmax(1).cpu().numpy(); zacc=(np.array(LABELS)[zs]==L).mean()
    out=[]
    for nm,E in (("clip",C),("dino",Dn)):
        S_=(E@E.T).cpu().numpy(); np.fill_diagonal(S_,-9); nn=S_.argmax(1)
        out.append(f"{nm} 1-NN label {(L[nn]==L).mean():.2f} seed {(Sd[nn]==Sd).mean():.2f}")
    print(f"step{step} {ch:5s} | CLIP zero-shot label acc {zacc:.2f} (chance .14) | "+" | ".join(out))
