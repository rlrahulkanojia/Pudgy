# G0.1 eye-bar proxy, run from the session scratchpad on 2026-10-12 and checked in as-is
# afterwards (docs/REPORT_G01.md §3.2 / §6). Paths are the box paths of that run.
import sys, numpy as np, torch
sys.path[:0]=['/workspace/Pudgy/expv2-minimax-h3','/workspace/Pudgy/expv2-minimax-h3/eval']
import h3lib as L, vae_gate as G
from pathlib import Path
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage as nd
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
f=torch.float32; vc=dict(offload_dtype=f,offload_device="cpu",onload_dtype=f,onload_device="cpu",preparing_dtype=f,preparing_device="cuda",computation_dtype=f,computation_device="cuda")
pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=f,device="cuda",model_configs=[L.model_config(L.PARTS["vae"],**vc)],processor_config=L.model_config(L.C.PROCESSOR),vram_limit=70)
lum=lambda a:a.astype(np.float32)@np.array([.299,.587,.114],np.float32)
S=Path('/workspace/Data/processed/expv2_h3/vae_gate/png768')
for clip in ('polly_angry_FRONT__white_f39','pax_angry_FRONT__white_f39','polly_walking_FRONT__white_f22'):
    fr=[Image.open(f).convert('RGB') for f in sorted((S/clip).glob('*.png'))]; src=np.stack([np.asarray(f) for f in fr])
    dec,_=G.h3_roundtrip(pipe,fr)
    cen,brk=[],[]
    for i in range(min(len(src),len(dec))):
        sk=skeletonize(lum(src[i])<64); dark=lum(dec[i])<128; cen.append(dark[sk].mean())
        lost=sk&~dark; lab,n=nd.label(lost,structure=np.ones((3,3))); brk.append(int((nd.sum(lost,lab,range(1,n+1))>=6).sum()) if n else 0)
    mse=np.mean((src[:len(dec)].astype(float)-dec[:len(src)].astype(float))**2)
    print(clip,'fp32: psnr %.2f centerline_min %.4f broken_max %d'%(10*np.log10(255**2/mse),min(cen),max(brk)),flush=True)
