#!/usr/bin/env python3
"""G1: blind human A/B of the v10 candidate vs v8merged (Training_Approach_v10.md section 5.1; Q1, Q12, Q19).

build   After suite_v10 generated both configs: one pair per (G1 job, seed) where both I2V clips exist
        (34 jobs x 3 seeds = 102 pairs), same first frame, caption and seed. Pair order is shuffled and
        each pair's sides are randomised with --seed. Clips are re-encoded (H.264, metadata stripped,
        faststart) to clips/pNNN_L.mp4 / pNNN_R.mp4, so nothing in the folder names a model. Writes a
        self-contained static page <out>/index.html (open it locally: no server, no network) and the
        KEY next to the folder, <out>.key.json - never send the key to a judge.
        The page shows the caption and the two clips side by side (looping, replay together, 0.5x),
        records one choice per pair - left better / right better / can't tell - keeps progress in the
        browser's localStorage (per page id), jumps to the first unanswered pair, and exports the
        answers as JSON (an exported file can be imported back to continue elsewhere).
tally   Reads the key and one or more answer files. A judgement is a win for v10 when the judge
        chose v10's side. "Can't tell" counts HALF a win to each side; the v10 score is rounded DOWN
        to an integer k, and n = all judged pairs (ties are not dropped, so n stays fixed and the test
        stays conservative). Pass: exact one-sided binomial P(X >= k | n, 1/2) <= 0.05 (n=80 needs 48).
        Several judges' files are pooled (each judgement counted; they share the same pairs, so the
        pooled p is optimistic - per-judge results are printed too). The sign test without ties is
        printed for information.
threshold  prints the smallest passing k for n.

Box (after `suite_v10.py gen` for both configs; CPU only):
    cd /workspace/Pudgy/finetune/wan/eval_v10; PY=/workspace/Pudgy/.venv-wan/bin/python
    $PY g1_pairs_v10.py build --cand v10@150      # -> runs/$RUN_NAME/g1/pages/v10_150_vs_v8merged/ (+ .key.json)
Mac (judges open index.html in Safari / Chrome; export; then):
    python3 g1_pairs_v10.py tally --key v10_150_vs_v8merged.key.json g1_answers_<judge>_<page>.json [...]
    python3 g1_pairs_v10.py threshold --n 102
Exit codes: build 0 / 2 (fewer than --min-pairs pairs); tally 0 PASS / 10 FAIL / 2 incomplete or wrong key.
The tally and threshold need only the Python standard library.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ALPHA = 0.05
SEEDS = (42, 43, 44)
EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 10, 2


# ---------------------------------------------------------------------------------------------
# statistics (stdlib only)
# ---------------------------------------------------------------------------------------------
def binom_sf(k, n):
    """P(X >= k), X ~ Binomial(n, 1/2), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def threshold(n, alpha=ALPHA):
    return next((k for k in range(n + 1) if binom_sf(k, n) <= alpha), None)


def verdict(wins, losses, ties):
    n = wins + losses + ties
    score = wins + ties / 2
    k = int(math.floor(score))
    p = binom_sf(k, n)
    m = wins + losses
    return {"n": n, "v10_wins": wins, "v8merged_wins": losses, "ties": ties, "score": score, "k": k,
            "p_one_sided": p, "threshold_k": threshold(n), "pass": n > 0 and p <= ALPHA,
            "sign_test_without_ties": {"n": m, "k": wins, "p": binom_sf(wins, m) if m else None}}


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def encode(src, dst):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1", "-an",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)])
    if r.returncode:
        sys.exit(f"ffmpeg failed on {src}")


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which shot is better? (blind A/B)</title>
<style>
:root { --bg:#f6f5f2; --fg:#1d1d1f; --muted:#6b6b70; --line:#d9d7d2; --accent:#2f5d8a; --pick:#2f7d4f; --tie:#8a6d2f; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background:var(--bg); color:var(--fg); }
header { display:flex; gap:16px; align-items:center; flex-wrap:wrap; padding:10px 18px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:2; }
header h1 { font-size:16px; margin:0 8px 0 0; }
#progress { font-variant-numeric:tabular-nums; color:var(--muted); }
#bar { flex:1; min-width:120px; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
#bar > div { height:100%; background:var(--accent); width:0; }
main { max-width:1100px; margin:0 auto; padding:8px 18px 24px; }
.help { color:var(--muted); font-size:12.5px; margin:0 0 6px; }
#caption { background:#fff; border:1px solid var(--line); border-radius:8px; padding:7px 12px; margin:0 0 8px; font-size:14px; }
#caption b { display:block; font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin-bottom:3px; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.side { background:#fff; border:2px solid var(--line); border-radius:10px; padding:6px; text-align:center; }
.side.chosen { border-color:var(--pick); box-shadow:0 0 0 3px rgba(47,125,79,.15); }
.side.tie { border-color:var(--tie); }
.side h2 { margin:0 0 4px; font-size:13px; color:var(--muted); font-weight:600; }
video { display:block; margin:0 auto; height:calc(100vh - 360px); min-height:260px; width:auto; max-width:100%; background:#000; border-radius:6px; }
.controls { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:10px 0 4px; }
button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:9px 16px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
button.choice { min-width:170px; font-weight:600; }
button.choice.on { background:var(--pick); border-color:var(--pick); color:#fff; }
button.choice.tieon { background:var(--tie); border-color:var(--tie); color:#fff; }
.nav { display:flex; gap:10px; justify-content:center; align-items:center; margin-top:6px; color:var(--muted); }
kbd { font:12px ui-monospace, Menlo, monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; background:#fff; }
.tools { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
input[type=text] { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; width:150px; }
#done { display:none; background:#eaf4ee; border:1px solid #b9dcc6; border-radius:8px; padding:10px 14px; margin:10px 0; }
@media (max-width:760px) { .pair { grid-template-columns:1fr; } }
</style></head>
<body>
<header>
  <h1>Which shot is better?</h1>
  <span id="progress">0 / 0</span><div id="bar"><div></div></div>
  <div class="tools">
    <label>Your name <input type="text" id="judge" placeholder="required to export"></label>
    <button id="export" class="primary">Download answers</button>
    <label><button id="importbtn" type="button">Import answers</button><input type="file" id="import" accept=".json,application/json" hidden></label>
  </div>
</header>
<main>
  <p class="help">Both clips start from the same first frame, with the same caption. Pick the one that is the better
  <b>production shot</b>: on-model characters, clean animation, and how well it follows the caption. Sides are random.
  Use <i>Can't tell</i> only when you really cannot choose. Keys: <kbd>1</kbd>/<kbd>&larr;</kbd> left,
  <kbd>3</kbd>/<kbd>&rarr;</kbd> right, <kbd>2</kbd>/<kbd>&darr;</kbd> can't tell, <kbd>R</kbd> replay both,
  <kbd>S</kbd> slow motion, <kbd>N</kbd>/<kbd>P</kbd> next / previous. Progress is saved in this browser.</p>
  <div id="done">All pairs answered. Please <b>Download answers</b> and send the file back. You can still revisit and change any pair.</div>
  <div id="caption"><b>Caption <span id="pid"></span></b><span id="captext"></span></div>
  <div class="pair">
    <div class="side" id="sideL"><h2>Left</h2><video id="vL" muted loop playsinline preload="auto"></video></div>
    <div class="side" id="sideR"><h2>Right</h2><video id="vR" muted loop playsinline preload="auto"></video></div>
  </div>
  <div class="controls">
    <button class="choice" id="cL">&larr; Left is better</button>
    <button class="choice" id="cT">Can't tell</button>
    <button class="choice" id="cR">Right is better &rarr;</button>
  </div>
  <div class="nav">
    <button id="prev">Previous</button>
    <button id="replay">Replay both</button>
    <button id="slow">Slow motion: off</button>
    <button id="next">Next</button>
    <button id="first">First unanswered</button>
  </div>
</main>
<pre id="autotest-out" hidden></pre>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("data").textContent);
  const KEY = "g1_answers_" + D.page_id;
  const $ = (id) => document.getElementById(id);
  let st = load();
  let pos = Math.min(st.pos || 0, D.pairs.length - 1);
  let slow = false;

  function load() {
    try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.answers) return s; } catch (e) {}
    return { judge: "", answers: {}, pos: 0 };
  }
  function save() {
    st.pos = pos;
    try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) { console.warn("localStorage unavailable", e); }
  }
  function nAnswered() { return D.pairs.filter((p) => st.answers[p.id]).length; }
  function firstOpen() { const i = D.pairs.findIndex((p) => !st.answers[p.id]); return i < 0 ? pos : i; }

  function show(i) {
    pos = Math.max(0, Math.min(D.pairs.length - 1, i));
    const p = D.pairs[pos];
    $("pid").textContent = "(" + p.id + ", pair " + (pos + 1) + " of " + D.pairs.length + ")";
    $("captext").textContent = p.caption;
    for (const s of ["L", "R"]) {
      const v = $("v" + s);
      if (v.getAttribute("src") !== p[s]) { v.setAttribute("src", p[s]); v.load(); }
      v.playbackRate = slow ? 0.5 : 1;
    }
    replay();
    paint(); save();
  }
  function paint() {
    const a = (st.answers[D.pairs[pos].id] || {}).c;
    $("cL").className = "choice" + (a === "L" ? " on" : "");
    $("cR").className = "choice" + (a === "R" ? " on" : "");
    $("cT").className = "choice" + (a === "T" ? " tieon" : "");
    $("sideL").className = "side" + (a === "L" ? " chosen" : a === "T" ? " tie" : "");
    $("sideR").className = "side" + (a === "R" ? " chosen" : a === "T" ? " tie" : "");
    const n = nAnswered();
    $("progress").textContent = n + " / " + D.pairs.length + " answered";
    $("bar").firstElementChild.style.width = (100 * n / D.pairs.length) + "%";
    $("done").style.display = n === D.pairs.length ? "block" : "none";
    $("judge").value = st.judge || "";
  }
  function replay() {
    for (const s of ["L", "R"]) { const v = $("v" + s); try { v.currentTime = 0; } catch (e) {} const pr = v.play(); if (pr) pr.catch(() => {}); }
  }
  function choose(c) {
    st.answers[D.pairs[pos].id] = { c: c, t: new Date().toISOString() };
    save(); paint();
    setTimeout(() => show(nAnswered() === D.pairs.length ? pos + 1 : firstOpenAfter(pos)), 180);
  }
  function firstOpenAfter(i) {
    for (let k = 1; k <= D.pairs.length; k++) { const j = (i + k) % D.pairs.length; if (!st.answers[D.pairs[j].id]) return j; }
    return i;
  }
  function exportObj() {
    return { kind: "pudgy-g1-answers", page_id: D.page_id, judge: st.judge || "", exported: new Date().toISOString(),
             n_pairs: D.pairs.length, n_answered: nAnswered(), answers: st.answers };
  }
  function download() {
    if (!st.judge) { alert("Please type your name first."); $("judge").focus(); return; }
    const blob = new Blob([JSON.stringify(exportObj(), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "g1_answers_" + st.judge.replace(/[^A-Za-z0-9_-]+/g, "_") + "_" + D.page_id + ".json";
    document.body.appendChild(a); a.click(); a.remove();
  }
  function importFile(f) {
    const r = new FileReader();
    r.onload = () => {
      try {
        const o = JSON.parse(r.result);
        if (o.page_id !== D.page_id) { alert("That file belongs to a different page (" + o.page_id + ")."); return; }
        st.answers = Object.assign({}, o.answers, st.answers); if (!st.judge) st.judge = o.judge || "";
        save(); show(firstOpen());
      } catch (e) { alert("Not an answers file: " + e); }
    };
    r.readAsText(f);
  }

  $("cL").onclick = () => choose("L"); $("cR").onclick = () => choose("R"); $("cT").onclick = () => choose("T");
  $("prev").onclick = () => show(pos - 1); $("next").onclick = () => show(pos + 1);
  $("first").onclick = () => show(firstOpen()); $("replay").onclick = replay;
  $("slow").onclick = () => { slow = !slow; $("slow").textContent = "Slow motion: " + (slow ? "0.5x" : "off");
    for (const s of ["L", "R"]) $("v" + s).playbackRate = slow ? 0.5 : 1; };
  $("judge").oninput = (e) => { st.judge = e.target.value.trim(); save(); };
  $("export").onclick = download;
  $("importbtn").onclick = () => $("import").click();
  $("import").onchange = (e) => { if (e.target.files[0]) importFile(e.target.files[0]); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    const k = e.key;
    if (k === "1" || k === "ArrowLeft") choose("L");
    else if (k === "3" || k === "ArrowRight") choose("R");
    else if (k === "2" || k === "ArrowDown") choose("T");
    else if (k === "r" || k === "R") replay();
    else if (k === "s" || k === "S") $("slow").click();
    else if (k === "n" || k === "N") show(pos + 1);
    else if (k === "p" || k === "P") show(pos - 1);
    else return;
    e.preventDefault();
  });

  // Self-test for the builder's checks (opens with #autotest): answers every pair with a fixed
  // pattern through the same code path, then writes the export object into #autotest-out.
  if (location.hash === "#autotest") {
    st = { judge: "autotest", answers: {}, pos: 0 };
    const pat = ["L", "R", "T", "L"];
    D.pairs.forEach((p, i) => { pos = i; st.answers[p.id] = { c: pat[i % 4], t: "autotest" }; });
    save();
    const back = load();
    const out = exportObj();
    out.reloaded_from_localStorage = Object.keys(back.answers).length;
    $("autotest-out").textContent = JSON.stringify(out);
    $("autotest-out").hidden = false;
    st = { judge: "", answers: {}, pos: 0 }; pos = 0;
    localStorage.removeItem(KEY);
    paint();
    return;
  }
  show(st.pos ? pos : firstOpen());
})();
</script>
</body></html>
"""


def build(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common_v10 as C
    jobs = json.loads((Path(a.jobs) if a.jobs else C.G1DIR / "jobs.json").read_text())
    if a.limit:
        jobs = jobs[:a.limit]
    cdir = Path(a.cand_dir) if a.cand_dir else C.RESULTS / "suite" / C.cfg_dir(a.cand)
    bdir = Path(a.base_dir) if a.base_dir else C.RESULTS / "suite" / C.cfg_dir(a.base)
    pairs, missing = [], []
    for j in jobs:
        for s in a.seeds:
            pc, pb = cdir / f"{j['tag']}_s{s}.mp4", bdir / f"{j['tag']}_s{s}.mp4"
            if pc.exists() and pb.exists():
                pairs.append((j, s, pc, pb))
            else:
                missing.append(f"{j['tag']}_s{s}")
    if len(pairs) < a.min_pairs:
        print(f"only {len(pairs)} complete pairs (< {a.min_pairs}); missing e.g. {missing[:4]}")
        return EXIT_MISSING
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    name = f"{C.cfg_dir(a.cand)}_vs_{C.cfg_dir(a.base)}"
    out = Path(a.out) if a.out else C.RESULTS / "g1" / "pages" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "clips").mkdir(parents=True)
    page, key = [], {}
    for i, (j, s, pc, pb) in enumerate(pairs):
        pid = f"p{i + 1:03d}"
        cand_left = rng.random() < 0.5
        left, right = (pc, pb) if cand_left else (pb, pc)
        encode(left, out / "clips" / f"{pid}_L.mp4")
        encode(right, out / "clips" / f"{pid}_R.mp4")
        page.append({"id": pid, "caption": j["prompt"], "L": f"clips/{pid}_L.mp4", "R": f"clips/{pid}_R.mp4"})
        key[pid] = {"job": j["tag"], "seed": s, "v10_side": "L" if cand_left else "R",
                    "cand_clip": str(pc), "base_clip": str(pb)}
    page_id = hashlib.sha256(json.dumps([page, a.seed, a.cand, a.base]).encode()).hexdigest()[:12]
    data = json.dumps({"page_id": page_id, "pairs": page}, ensure_ascii=False).replace("</", "<\\/")
    (out / "index.html").write_text(PAGE.replace("__DATA__", data))
    (out / "README.txt").write_text(
        "Open index.html in Safari or Chrome (double-click; no internet needed). Type your name, judge every pair,\n"
        "then press 'Download answers' and send the downloaded JSON file back. Progress is kept in this browser;\n"
        "'Import answers' restores an exported file (e.g. on another computer).\n")
    keyf = out.parent / f"{out.name}.key.json"
    keyf.write_text(json.dumps({"page_id": page_id, "cand": a.cand, "base": a.base, "seed": a.seed,
                                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_pairs": len(page),
                                "missing": missing, "pairs": key}, indent=1))
    n = len(page)
    print(f"{n} pairs ({sum(v['v10_side'] == 'L' for v in key.values())} with v10 on the left) -> {out}/index.html")
    print(f"KEY (keep away from judges): {keyf}")
    print(f"pass needs >= {threshold(n)} of {n} (exact one-sided binomial, alpha {ALPHA}; can't tell = half)")
    if missing:
        print(f"  {len(missing)} job/seed pairs skipped (a clip missing): {missing[:6]}")
    return 0


def tally(a):
    key = json.loads(Path(a.key).read_text())
    files = [json.loads(Path(f).read_text()) for f in a.answers]
    ok_files = True
    per, pooled = [], [0, 0, 0]
    for f, o in zip(a.answers, files):
        if o.get("page_id") != key["page_id"]:
            print(f"{f}: page {o.get('page_id')} != key {key['page_id']}"); ok_files = False; continue
        w = l_ = t = 0
        for pid, ans in o["answers"].items():
            c = ans["c"] if isinstance(ans, dict) else ans
            side = key["pairs"][pid]["v10_side"]
            if c == "T":
                t += 1
            elif c == side:
                w += 1
            else:
                l_ += 1
        v = verdict(w, l_, t)
        v.update(judge=o.get("judge"), file=f, answered=len(o["answers"]), n_pairs=key["n_pairs"])
        per.append(v)
        pooled[0] += w; pooled[1] += l_; pooled[2] += t
    if not per:
        return EXIT_MISSING
    for v in per:
        print(f"judge {v['judge']!r:<14} answered {v['answered']}/{v['n_pairs']}  v10 {v['v10_wins']}  v8merged "
              f"{v['v8merged_wins']}  can't tell {v['ties']}  -> k {v['k']} of {v['n']} (needs {v['threshold_k']})  "
              f"p {v['p_one_sided']:.4f}  {'PASS' if v['pass'] else 'fail'}")
    pv = verdict(*pooled)
    incomplete = any(v["answered"] < v["n_pairs"] for v in per)
    print(f"POOLED over {len(per)} judge(s): v10 {pv['v10_wins']}  v8merged {pv['v8merged_wins']}  can't tell {pv['ties']}  "
          f"k {pv['k']} of n {pv['n']} (threshold {pv['threshold_k']}), one-sided exact p = {pv['p_one_sided']:.5f}")
    st = pv["sign_test_without_ties"]
    if st["n"]:
        print(f"  (sign test without ties: {st['k']} of {st['n']}, p = {st['p']:.5f}; information only)")
    if pv["n"] < 80:
        print(f"  WARNING: n = {pv['n']} < 80, the size G1 was powered for")
    res = {"key": a.key, "per_judge": per, "pooled": pv, "incomplete": incomplete,
           "G1": "PASS" if pv["pass"] and not incomplete and ok_files else "FAIL" if not incomplete and ok_files else "INCOMPLETE",
           "rule": "exact one-sided binomial over all judged pairs, alpha 0.05; can't tell = half a win, score rounded down"}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"G1: {res['G1']}")
    if not ok_files or incomplete:
        return EXIT_MISSING
    return EXIT_PASS if pv["pass"] else EXIT_FAIL


def main():
    ap = argparse.ArgumentParser(description="v10 G1 blind pairs: build the page, tally the answers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cand", required=True); b.add_argument("--base", default="v8merged")
    b.add_argument("--seed", type=int, default=2026, help="pair order and sides")
    b.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    b.add_argument("--out"); b.add_argument("--jobs", help="default /workspace/eval_v10/g1/jobs.json")
    b.add_argument("--cand-dir"); b.add_argument("--base-dir")
    b.add_argument("--limit", type=int, default=0); b.add_argument("--min-pairs", type=int, default=80)
    t = sub.add_parser("tally"); t.add_argument("--key", required=True); t.add_argument("answers", nargs="+")
    t.add_argument("--json", help="write the result here")
    h = sub.add_parser("threshold"); h.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a)
    if a.cmd == "tally":
        return tally(a)
    k = threshold(a.n)
    print(f"n = {a.n}: pass needs k >= {k} (P(X >= {k}) = {binom_sf(k, a.n):.5f}; "
          f"P(X >= {k - 1}) = {binom_sf(k - 1, a.n):.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
