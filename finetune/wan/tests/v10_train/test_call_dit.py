"""asset_embed_v9.py's patched musubi call_dit path (CPU, no GPU).

Uses the REAL musubi v0.3.4 WanNetworkTrainer.call_dit/_call_dit, LoRANetwork (lora_wan) and the
--base_weights merge path (create_arch_network_from_weights + merge_to, as trainer_base._build_network
does), with a tiny stand-in transformer that has WanModel's call signature and uses the text context.
The legacy comparison loads asset_embed_v9.py as committed in LEGACY_COMMIT (v9c2) from git history.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import common
from common import check
import torch
import musubi_tuner.wan_train_network as W  # noqa: E402
from musubi_tuner.networks import lora as L, lora_wan  # noqa: E402
from musubi_tuner.wan.configs import WAN_CONFIGS  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

LEGACY_COMMIT = "6450000d45b274bfe01400abb85d8b439fd5646c"   # last commit of asset_embed_v9.py before the v10 port
W.main = lambda: None                          # cmd_train patches musubi, then calls W.main(): make it a no-op
PATCHED = [(L.LoRANetwork, "prepare_optimizer_params"), (L.LoRANetwork, "get_trainable_params"),
           (L.LoRANetwork, "save_weights"), (W.WanNetworkTrainer, "call_dit"),
           (W.WanNetworkTrainer, "on_post_optimizer_step"), (W.WanNetworkTrainer, "extra_step_logs"),
           (L, "create_network_from_weights"), (W, "load_wan_model")]
ORIG = {(id(o), n): getattr(o, n) for o, n in PATCHED}
TMP = Path(tempfile.mkdtemp(prefix="calldit_", dir=common.TMP))
D, DC, K = 8, 6, 3                              # latent channels, "T5" width, E rows (phrase tokens)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def legacy_module():
    """The committed v9c2 original, with its one CUDA-only line pointed at the CPU."""
    src = subprocess.run(["git", "-C", str(common.REPO), "show", f"{LEGACY_COMMIT}:finetune/wan/asset_embed_v9.py"],
                         capture_output=True, text=True, check=True).stdout
    assert src.count('dev = torch.device("cuda")') == 1, "unexpected legacy source"
    p = TMP / "asset_embed_v9_legacy_cpu.py"
    p.write_text(src.replace('dev = torch.device("cuda")', 'dev = torch.device("cpu")'))
    return load(p, "ae_old")


NEW = load(common.FW / "asset_embed_v9.py", "ae_new")
OLD = legacy_module()


def install(mod, flags, rest):
    for o, n in PATCHED:
        setattr(o, n, ORIG[(id(o), n)])
    sys.argv = ["asset_embed_v9.py", "train", *flags, "--", *rest]
    mod.main()


class WanAttentionBlock(torch.nn.Module):         # the class name lora_wan targets
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(D, D)
        self.ctx = torch.nn.Linear(DC, D)


class Tiny(torch.nn.Module):
    """Stand-in WanModel: same call signature, uses the context so the E splice matters."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([WanAttentionBlock(), WanAttentionBlock()])
        self.blocks_to_swap = 1
        self.calls = []

    def switch_block_swap_for_inference(self):
        self.calls.append("inference")

    def switch_block_swap_for_training(self):
        self.calls.append("training")

    def forward(self, x, t, context, clip_fea, seq_len, y):
        h = x.permute(0, 2, 3, 4, 1) + y.mean(1, keepdim=True).permute(0, 2, 3, 4, 1)
        c = torch.stack([ctx.mean(0) for ctx in context])
        for b in self.blocks:
            h = h + torch.tanh(b.q(h)) + b.ctx(c)[:, None, None, None, :]
        return [hh.permute(3, 0, 1, 2) for hh in h]


class Acc:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, m):
        return m


def rand_lora(net, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in net.unet_loras:
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def build(golden_extra=False):
    """Tiny DiT, golden merged the way trainer_base._build_network does (--base_weights), then the
    trained LoRA applied with non-zero weights. Returns (tiny, unmerged copy, network)."""
    torch.manual_seed(0)
    tiny = Tiny()
    raw = copy.deepcopy(tiny)
    gnet = lora_wan.create_arch_network(1.0, 2, 4, None, None, tiny)
    rand_lora(gnet, 1)
    golden = {}
    for m in gnet.unet_loras:                          # musubi's LoRA file layout
        golden[f"{m.lora_name}.lora_down.weight"] = m.lora_down.weight.detach().clone()
        golden[f"{m.lora_name}.lora_up.weight"] = m.lora_up.weight.detach().clone()
        golden[f"{m.lora_name}.alpha"] = torch.tensor(float(m.alpha))
    if golden_extra:
        golden["lora_unet_blocks_9_q.lora_down.weight"] = torch.zeros(2, D)
        golden["lora_unet_blocks_9_q.lora_up.weight"] = torch.zeros(D, 2)
        golden["lora_unet_blocks_9_q.alpha"] = torch.tensor(4.0)
    module = lora_wan.create_arch_network_from_weights(1.0, golden, unet=tiny, for_inference=True)
    module.merge_to(None, tiny, golden, None, "cpu")
    net = lora_wan.create_arch_network(1.0, 4, 4, None, None, tiny)
    net.apply_to(None, tiny, apply_text_encoder=False, apply_unet=True)
    rand_lora(net, 2)
    net.prepare_optimizer_params(unet_lr=1e-4)         # the patched prep stores the network
    return tiny, raw, net


def trainer():
    tr = W.WanNetworkTrainer()
    tr.config = WAN_CONFIGS["i2v-A14B"]
    tr._i2v_training, tr._control_training, tr.high_low_training = True, False, False
    return tr


def inputs(seed=3):
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, D, 2, 4, 4, generator=g)
    noise = torch.randn(lat.shape, generator=g)
    noisy = 0.05 * lat + 0.95 * noise
    ctx = torch.randn(7, DC, generator=g)
    img = torch.randn(1, 4, 2, 4, 4, generator=g)
    return lat, noise, noisy, ctx, img


def batch_of(ctx, img, span=(0, 0), preserve=None):
    b = {"latents_image": img, "t5": [ctx.clone()], "assetspan": [torch.tensor(list(span))], "timesteps": None}
    if preserve is not None:
        b["preserve"] = [torch.tensor([preserve])]
    return b


ARGS = argparse.Namespace(gradient_checkpointing=True, max_train_steps=10)
T = torch.tensor([950.0])


def run(tr, tiny, batch, lat, noise, noisy):
    return tr.call_dit(ARGS, Acc(), tiny, lat, batch, noise, noisy, T, torch.float32)


def ref(tiny, net, mult, noisy, ctx, img):
    net.set_multiplier(mult)
    with torch.no_grad():
        out = torch.stack(tiny(noisy, t=T, context=[ctx], clip_fea=None, seq_len=0, y=img))
    net.set_multiplier(1.0)
    return out


E5 = torch.randn(K, DC, generator=torch.Generator().manual_seed(7))
E5_PATH = TMP / "e5.asset.safetensors"
save_file({"E": E5}, str(E5_PATH))
REST = ["--task", "i2v-A14B", "--dit", "high.safetensors", "--base_weights", "golden.safetensors",
        "--output_dir", str(TMP), "--output_name", "run"]
V10 = ["--init", str(E5_PATH), "--offset", "--freeze-asset", "--preserve", "--preserve-weight", "1.0",
       "--preserve-marker", "key", "--log-every", "5", "--strict-base-weights"]

print("== v10 key mode, single expert, golden via --base_weights")
install(NEW, V10, REST)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tiny, raw, net = build()
check("base weights: 4/4 LoRA modules match the DiT" in buf.getvalue(),
      "--strict-base-weights reports the merge: " + (buf.getvalue().strip().splitlines() or ["(nothing)"])[-1][:100])
tr = trainer()
lat, noise, noisy, ctx, img = inputs()
base_ref = ref(tiny, net, 0.0, noisy, ctx, img)
lora_ref = ref(tiny, net, 1.0, noisy, ctx, img)
with torch.no_grad():
    unmerged = torch.stack(raw(noisy, t=T, context=[ctx], clip_fea=None, seq_len=0, y=img))
check(not torch.allclose(base_ref, unmerged), "golden is merged into the DiT (multiplier-0 base != raw DiT)")
check(not torch.allclose(base_ref, lora_ref), "the trained LoRA changes the prediction")

# 1. replay-marked sample: target = base (LoRA x0, golden kept), pred = LoRA on, swap switched around it
out = run(tr, tiny, batch_of(ctx, img, preserve=1), lat, noise, noisy)
check(torch.allclose(out.target, base_ref, atol=1e-6, rtol=0), "replay: target == golden-merged base pred (LoRA x0)")
check(torch.allclose(out.pred, lora_ref, atol=1e-6, rtol=0), "replay: pred == LoRA-on pred")
check(not out.target.requires_grad and out.pred.requires_grad, "replay: target detached, pred has grad")
check(tiny.calls == ["inference", "training"], f"replay: block swap forward-only around the base pass {tiny.calls}")
check(net.multiplier == 1.0 and all(m.multiplier == 1.0 for m in net.unet_loras), "replay: multiplier restored to 1")
loss = torch.nn.functional.mse_loss(out.pred, out.target)
loss.backward()
gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
check(gn > 0, f"replay: preservation loss {float(loss):.4f} back-propagates into the LoRA (grad norm {gn:.3f})")
net.zero_grad()

# 2. scene window without the phone: plain data loss
tiny.calls.clear()
out = run(tr, tiny, batch_of(ctx, img, preserve=0), lat, noise, noisy)
check(torch.allclose(out.target, noise - lat), "plain scene: target == noise - latents (data MSE)")
check(torch.allclose(out.pred, lora_ref, atol=1e-6, rtol=0), "plain scene: pred == LoRA-on pred, no splice")
check(tiny.calls == [], "plain scene: no base pass")

# 3. scene window naming the phone: E5 added to rows [s, e), data loss
b = batch_of(ctx, img, span=(2, 2 + K), preserve=0)
out = run(tr, tiny, b, lat, noise, noisy)
spliced = ctx.clone()
spliced[2:2 + K] += E5
check(torch.allclose(b["t5"][0], spliced), "phone scene: batch t5 rows [2, 5) == cached rows + E5 (offset)")
check(torch.equal(b["t5"][0][:2], ctx[:2]) and torch.equal(b["t5"][0][2 + K:], ctx[2 + K:]), "phone scene: other rows untouched")
phone_ref = ref(tiny, net, 1.0, noisy, spliced, img)
check(torch.allclose(out.pred, phone_ref, atol=1e-6, rtol=0), "phone scene: the DiT saw the spliced context")
check(torch.allclose(out.target, noise - lat), "phone scene: target == data")

# 4. inconsistencies stop the run
for name, bb in [("replay marker + phone span", batch_of(ctx, img, span=(2, 2 + K), preserve=1)),
                 ("no varlen_preserve (unmarked cache)", batch_of(ctx, img)),
                 ("span length != E rows", batch_of(ctx, img, span=(1, 3), preserve=0))]:
    try:
        run(tr, tiny, bb, lat, noise, noisy)
        check(False, f"raises on {name}")
    except RuntimeError as e:
        check(True, f"raises on {name}: {str(e)[:70]}")

# 5. counters: log line and sidecar
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tr.on_post_optimizer_step(ARGS, Acc(), net, tiny, True, 4)       # musubi passes the pre-increment step
    tr.on_post_optimizer_step(ARGS, Acc(), net, tiny, True, 5)       # step 6: not a multiple of 5
    tr.on_post_optimizer_step(ARGS, Acc(), net, tiny, False, 8)      # not a sync step
line = [x for x in buf.getvalue().splitlines() if x.startswith("PRESERVE_STATS")]
check(len(line) == 1 and "step=5 " in line[0] and "preserve=1 " in line[0] and "data=2 " in line[0]
      and "asset=1 " in line[0], f"PRESERVE_STATS every 5 steps: {line}")
logs = tr.extra_step_logs(ARGS, {})
check(logs.get("preserve/samples_preserve") == 1 and logs.get("preserve/samples_data") == 2, f"tensorboard extras {logs}")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    net.save_weights(str(TMP / "run-step00000025.safetensors"), None, {"ss_steps": "25"})
with safe_open(str(TMP / "run-step00000025.asset.safetensors"), "pt") as f:
    md = f.metadata()
    E_saved = f.get_tensor("E")
check(md["samples_preserve"] == "1" and md["samples_data"] == "2" and md["samples_asset"] == "1"
      and md["samples_total"] == "3" and md["mode"] == "single-expert" and md["preserve_marker"] == "key"
      and md["samples_high"] == "0" and md["frozen"] == "True" and md["delta_norm"] == "0.0000",
      f"sidecar metadata {md}")
check(torch.equal(E_saved, E5), "sidecar E == E5 (frozen)")
check("PRESERVE_STATS step=25" in buf.getvalue(), "checkpoint save prints PRESERVE_STATS")

# 6. preservation weight scales both sides
install(NEW, [*V10[:6], "4.0", *V10[7:]], REST)
tiny, raw, net = build()
tr = trainer()
out = run(tr, tiny, batch_of(ctx, img, preserve=1), lat, noise, noisy)
check(torch.allclose(out.target, 2 * ref(tiny, net, 0.0, noisy, ctx, img), atol=1e-6)
      and torch.allclose(out.pred, 2 * ref(tiny, net, 1.0, noisy, ctx, img), atol=1e-6), "--preserve-weight 4: pred and target x2")

# 7. 100 samples without one replay marker -> stop
install(NEW, V10, REST)
tiny, raw, net = build()
tr = trainer()
i = -1
try:
    for i in range(100):
        run(tr, tiny, batch_of(ctx, img, preserve=0), lat, noise, noisy)
    check(False, "stops after 100 samples with no replay marker")
except RuntimeError as e:
    check(i == 99, f"stops at sample {i + 1} with no replay marker: {str(e)[:60]}")

# 8. resume continues the counters from the step's sidecar
install(NEW, V10, [*REST, "--resume", str(TMP / "run-step00000025-state")])
tiny, raw, net = build()
tr = trainer()
run(tr, tiny, batch_of(ctx, img, preserve=1), lat, noise, noisy)
with contextlib.redirect_stdout(io.StringIO()):
    net.save_weights(str(TMP / "run-step00000050.safetensors"), None, {"ss_steps": "50"})
with safe_open(str(TMP / "run-step00000050.asset.safetensors"), "pt") as f:
    md = f.metadata()
check(md["samples_preserve"] == "2" and md["samples_total"] == "4",
      f"resume: counters continue (preserve 1+1, total 3+1): {md['samples_preserve']}, {md['samples_total']}")

# 9. strict base weights catches a golden module the DiT does not have
install(NEW, V10, REST)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        build(golden_extra=True)
    check(False, "--strict-base-weights raises on an unmatched golden module")
except RuntimeError as e:
    check(True, f"--strict-base-weights raises on an unmatched golden module: {str(e)[:60]}")

# 10. legacy (v9c2) flags: NEW == OLD exactly, and no marker needed
print(f"== legacy nospan mode: new code vs asset_embed_v9.py at {LEGACY_COMMIT[:7]}")
LEG = ["--init", str(E5_PATH), "--offset", "--freeze-asset", "--preserve", "--preserve-weight", "1.0"]
REST2 = ["--task", "i2v-A14B", "--dit", "low.safetensors", "--dit_high_noise", "high.safetensors"]
results = {}
for tag, mod in (("old", OLD), ("new", NEW)):
    install(mod, LEG, REST2)
    tiny, raw, net = build()
    tr = trainer()
    rs = []
    for span in ((0, 0), (2, 2 + K)):
        tiny.calls.clear()
        o = run(tr, tiny, batch_of(ctx, img, span=span), lat, noise, noisy)
        rs.append((o.pred.detach().clone(), o.target.detach().clone(), list(tiny.calls)))
    results[tag] = rs
for i, what in enumerate(("empty span -> preserve", "phone span -> data")):
    po, to, co = results["old"][i]
    pn, tn, cn = results["new"][i]
    check(torch.equal(po, pn) and torch.equal(to, tn) and co == cn, f"legacy {what}: identical pred/target/swap calls ({co})")
check(torch.allclose(results["new"][0][1], ref(tiny, net, 0.0, noisy, ctx, img), atol=1e-6),
      "legacy: empty span is still distilled to the base")

for o, n in PATCHED:
    setattr(o, n, ORIG[(id(o), n)])
common.finish()
