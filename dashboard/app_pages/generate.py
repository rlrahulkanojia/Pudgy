"""Generate videos: make new clips with the trained model on the GPU inference server
(finetune/wan/infer_api). The server is switched on only when needed; when it is off this page
says so and still shows every video saved so far."""
import io
import json
import time

import streamlit as st
from PIL import Image

import azure_utils
import inference_client as ic

LENGTHS = {   # key -> (frames per take, takes, label)
    "test": (33, 1, "1.4 s · quick test"),
    "one": (81, 1, "3.4 s"),
    "two": (81, 2, "6.7 s"),
    "three": (81, 3, "10 s"),
}
MODE_LABELS = {"chain": "Continuous", "loop": "Seamless loop", "first_last": "First + last frame"}
CHARACTERS = {"pax": "Pax", "polly": "Polly", "both": "Pax and Polly"}
STATE_BADGE = {"queued": ("Waiting", "gray", ":material/schedule:"),
               "running": ("Generating", "orange", ":material/autorenew:"),
               "done": ("Done", "green", ":material/check_circle:"),
               "failed": ("Failed", "red", ":material/error:"),
               "cancelled": ("Cancelled", "gray", ":material/cancel:"),
               "interrupted": ("Interrupted", "red", ":material/power_off:")}


# ---- cached server and storage calls ------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def server_status():
    return ic.discover()


@st.cache_data(ttl=120, show_spinner=False)
def server_info(url):
    return ic.info(url)


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def keyframe_png(url, name):
    return ic.keyframe_png(url, name)


@st.cache_data(ttl=600, show_spinner=False, max_entries=256)
def prompt_preview(url, guided_json, text):
    try:
        return ic.preview_prompt(url, guided=json.loads(guided_json) if guided_json else None, prompt=text)
    except ic.ApiError as e:
        return {"error": e.message}


@st.cache_data(ttl=60, show_spinner=False)
def saved_videos():
    return azure_utils.list_inference_jobs(24)


@st.cache_data(ttl=600, show_spinner=False, max_entries=128)
def signed(blob):
    return azure_utils.signed_url(blob)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def server_video(url, job_id):
    return ic.video_bytes(url, job_id)


def minutes(seconds):
    if seconds is None:
        return "—"
    m = max(1, round(seconds / 60))
    return f"{m} min" if m < 90 else f"{m / 60:.1f} h"


def job_title(j):
    g = (j.get("request") or {}).get("guided") or {}
    if g.get("action"):
        return f"{g['action'].replace('_', ' ').capitalize()} · {CHARACTERS.get(g.get('characters'), '')}"
    return "Custom prompt"


def play(url, j):
    """The saved copy in Azure when there is one (it outlives the server), else the server's file."""
    blob = (j.get("result") or {}).get("blob")
    try:
        if blob:
            src = signed(blob)
            st.video(src)
            st.link_button("Download", src, icon=":material/download:", width="stretch")
        elif url:
            data = server_video(url, j["id"])
            st.video(data)
            st.download_button("Download", data, file_name=f"pudgy_{j['id']}.mp4", mime="video/mp4",
                               icon=":material/download:", width="stretch", key=f"dl_{j['id']}")
    except Exception as e:
        st.error(f"Could not load the video: {e}")


def job_card(url, j):
    r = j.get("request") or {}
    label, colour, icon = STATE_BADGE.get(j["state"], (j["state"], "gray", None))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{job_title(j)}**")
            st.badge(label, color=colour, icon=icon)
        st.caption(f"{r.get('width')}×{r.get('height')} · {r.get('length_seconds')} s · "
                   f"{MODE_LABELS.get(r.get('mode'), r.get('mode'))} · seed {r.get('seed')} · {j['created_utc'][:16].replace('T', ' ')} UTC")
        if j["state"] == "queued":
            pos = j.get("queue_position") or 1
            st.progress(0.0, text=f"Number {pos} in line · ready in about {minutes(j.get('eta_seconds'))}")
        elif j["state"] == "running":
            st.progress(min(float(j.get("progress") or 0), 1.0),
                        text=f"{(j.get('stage') or 'working').capitalize()} · about {minutes(j.get('eta_seconds'))} left")
        elif j["state"] == "done":
            play(url, j)
        elif j.get("error"):
            st.error(j["error"], icon=":material/error:")
        for w in j.get("warnings") or []:
            st.caption(f":material/warning: {w}")
        if j["state"] in ("queued", "running") and url:
            if st.button("Cancel", key=f"cancel_{j['id']}", icon=":material/close:", type="tertiary"):
                try:
                    ic.cancel(url, j["id"])
                    st.toast("Cancelled.")
                except ic.ApiError as e:
                    st.toast(e.message, icon=":material/error:")
                st.rerun(scope="fragment")
        with st.popover("Prompt", icon=":material/notes:", width="stretch"):
            st.code(r.get("prompt", ""), language=None, wrap_lines=True)


@st.fragment(run_every=20)
def status_panel():
    s = server_status()
    if st.session_state.server_online is not None and s["online"] != st.session_state.server_online:
        st.session_state.server_online = s["online"]
        st.rerun()                         # the form appears / disappears with the server
    st.session_state.server_online = s["online"]
    with st.container(border=True):
        if not s["online"]:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("#### :material/cloud_off: No inference server available")
                st.badge("Offline", color="red")
            st.markdown(s["reason"])
            st.caption("The GPU machine is switched on only when videos are needed. As soon as it is "
                       "started, this page finds it by itself (it checks every 20 seconds). Saved videos "
                       "are below.")
            if st.button("Check now", icon=":material/refresh:"):
                server_status.clear()
                st.rerun()
            return
        h = s["health"]
        state = h.get("state")
        badge = {"idle": ("Ready", "green"), "busy": ("Generating", "orange"),
                 "unavailable": ("Unavailable", "red")}.get(state, (state, "gray"))
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("#### :material/cloud_done: Inference server online")
            st.badge(badge[0], color=badge[1])
        if state == "unavailable":
            st.warning(h.get("reason") or "The server cannot take jobs right now.", icon=":material/block:")
        gpu = h.get("gpu") or {}
        bits = [f"GPU {gpu['name']}" if gpu.get("name") else None,
                f"model loaded: {h['model_loaded']}" if h.get("model_loaded") else "model loads on the first job",
                f"{h.get('queue', 0)} waiting" if h.get("queue") else None]
        st.caption(" · ".join(b for b in bits if b))


@st.fragment(run_every=8)
def jobs_panel(url):
    try:
        jobs = ic.jobs(url, 12)
    except ic.ApiError as e:
        st.warning(e.message, icon=":material/cloud_off:")
        return
    if not jobs:
        st.caption("No videos have been made on this server yet.")
        return
    mine = set(st.session_state.my_jobs)          # this visitor's own jobs first, each group newest first
    jobs = [j for j in jobs if j["id"] in mine] + [j for j in jobs if j["id"] not in mine]
    for i in range(0, len(jobs), 3):
        for col, j in zip(st.columns(3), jobs[i:i + 3]):
            with col:
                job_card(url, j)


def on_preset_change(keyframes):
    """A preset start frame shows a known action, character and set: fill the guided prompt with them."""
    k = next((k for k in keyframes if k["name"] == st.session_state.preset), None)
    if not k:
        return
    st.session_state.g_who = k["characters"]
    if k["action"] in st.session_state.get("_actions", {}):
        st.session_state.g_action = k["action"]
    if k.get("setting"):
        st.session_state.g_setting = k["setting"]


def on_prompt_mode_change():
    if st.session_state.prompt_mode == "Write my own" and not st.session_state.get("custom_prompt"):
        st.session_state.custom_prompt = st.session_state.get("last_guided_prompt", "")


# ---- page --------------------------------------------------------------------------------------------
st.session_state.setdefault("server_online", None)
st.session_state.setdefault("my_jobs", [])
st.session_state.setdefault("last_submit", (None, 0.0))
st.session_state.setdefault("g_who", "pax")        # also set by on_preset_change, so no widget default

st.title("Generate videos")
st.caption("Make new Pudgy Penguins clips with the trained model: pick a start frame, say what happens, "
           "and the GPU server renders it. Finished videos are saved and stay here.")

status_panel()
server = server_status()
configured = bool(ic.api_key())
if server["online"] and not configured:
    st.info("Generation is not set up on this dashboard yet: the API key (INFERENCE_API_KEY) is missing "
            "from its settings.", icon=":material/key_off:")

info = None
if server["online"] and configured:
    try:
        info = server_info(server["url"])
    except ic.ApiError as e:
        st.error(e.message, icon=":material/error:")

if info:
    url, health = server["url"], server["health"]
    st.session_state["_actions"] = info["presets"]["actions"]
    keyframes = info["keyframes"]
    problems = []

    st.subheader("New video")
    left, right = st.columns([3, 2], gap="large")
    with left:
        # 1. start frame ----------------------------------------------------------------------------
        st.markdown("**1 · Start frame**")
        src = st.segmented_control("Start frame", ["Preset", "Upload"], default="Preset", required=True,
                                   key="start_src", label_visibility="collapsed")
        start_png, start_name, preset = None, None, None
        if src == "Preset" and keyframes:
            names = [k["name"] for k in keyframes]
            if st.session_state.get("preset") not in names:
                st.session_state.preset = names[0]
                on_preset_change(keyframes)
            meta = {k["name"]: k for k in keyframes}
            preset = st.selectbox(
                "Preset start frame", names, key="preset", on_change=on_preset_change, args=(keyframes,),
                format_func=lambda n: f"{meta[n]['action'].replace('_', ' ').capitalize()} · "
                                      f"{CHARACTERS[meta[n]['characters']]} · {meta[n].get('setting') or 'own'} set",
                help="The client's characters on their living-room sets, at the size and spot the model was "
                     "tested with. Picking one also fills in the prompt below.")
            start_png = keyframe_png(url, preset)
            start_name = f"{preset}.png"
        elif src == "Preset":
            st.caption("This server has no preset start frames; upload one instead.")
        else:
            up = st.file_uploader("Start frame image", type=["png", "jpg", "jpeg", "webp"],
                                  help="The first frame of the video: Pax and/or Polly in their scene, ideally "
                                       "720×1280 for portrait or 1280×720 for landscape.")
            if up:
                start_png, start_name = up.getvalue(), up.name

        # 2. prompt ---------------------------------------------------------------------------------------
        st.markdown("**2 · What happens**")
        pmode = st.segmented_control("Prompt", ["Guided", "Write my own"], default="Guided", required=True,
                                     key="prompt_mode", on_change=on_prompt_mode_change,
                                     label_visibility="collapsed")
        actions = info["presets"]["actions"]
        guided, custom = None, None
        if pmode == "Guided":
            who = st.segmented_control("Characters", list(CHARACTERS), format_func=CHARACTERS.get,
                                       required=True, key="g_who")
            opts = [a for a, v in actions.items() if (v["who"] == "both") == (who == "both")]
            if st.session_state.get("g_action") not in opts:
                st.session_state.g_action = opts[0]
            action = st.selectbox("Action", opts, key="g_action",
                                  format_func=lambda a: f"{actions[a]['label'].capitalize()} · {actions[a]['kind']}",
                                  help="Every action here was in the training data, in these words.")
            sets = info["presets"]["sets"]
            setting = st.selectbox("Setting", list(sets) + ["other"], key="g_setting",
                                   format_func=lambda s: "Describe another setting…" if s == "other"
                                   else f"{s.capitalize()}: {sets[s]}",
                                   help="Match the start frame's background: the model keeps the scene it starts from.")
            setting_text = st.text_input("Describe the setting", max_chars=300, key="g_setting_text",
                                         placeholder="inside a snowy forest clearing, …") if setting == "other" else ""
            details = st.text_input("Extra details (optional)", max_chars=300, key="g_details",
                                    placeholder="e.g. smiling at the end",
                                    help="Added after the action. Keep it short; the trained wording does the work.")
            guided = {"characters": who, "action": action, "details": details}
            if setting == "other":
                guided["setting_text"] = setting_text
            else:
                guided["setting"] = setting
            if setting == "other" and not setting_text.strip():
                problems.append("Describe the setting, or pick one of the sets.")
        else:
            custom = st.text_area("Prompt", key="custom_prompt", height=180, max_chars=1500,
                                  placeholder="A 2D cartoon animation in the Pudgy Penguins style, … showing Pax, a "
                                              "short round blue penguin, …",
                                  help="Name Pax and Polly with their full wording and keep the style prefix: the "
                                       "checks on the right say what to fix.")
            if len((custom or "").strip()) < 10:
                problems.append("Write a prompt.")

        # 3. format -----------------------------------------------------------------------------------------
        st.markdown("**3 · Format**")
        sizes = info["sizes"]
        size = st.segmented_control("Size", list(sizes), format_func=lambda k: sizes[k]["label"],
                                    default=info["defaults"]["size"], required=True, key="size")
        mode = st.segmented_control("Motion", list(MODE_LABELS), format_func=MODE_LABELS.get, default="chain",
                                    required=True, key="mode", help=" · ".join(
                                        f"{MODE_LABELS[m]}: {d}" for m, d in info["modes"].items()))
        lengths = {k: v for k, v in LENGTHS.items() if mode != "first_last" or v[1] == 1}
        if st.session_state.get("length") not in lengths:
            st.session_state.length = "one"
        length = st.segmented_control("Length", list(lengths), format_func=lambda k: lengths[k][2],
                                      required=True, key="length",
                                      help="One take is 81 frames (3.4 s at 24 fps). Longer videos chain takes; "
                                           "10 s is the longest tested.")
        end_png = None
        if mode == "first_last":
            up_end = st.file_uploader("End frame image", type=["png", "jpg", "jpeg", "webp"], key="end_up",
                                      help="The video moves from the start frame to this one.")
            if up_end:
                end_png = (up_end.name, up_end.getvalue())
            else:
                problems.append("Upload the end frame for a first + last frame video.")

        with st.expander("Advanced", icon=":material/tune:"):
            models = [m for m in info["models"] if m["available"]]
            model = st.selectbox("Model", [m["id"] for m in models], key="model",
                                 format_func=lambda i: next(m["label"] for m in models if m["id"] == i))
            random_seed = st.toggle("Random seed", value=True, key="random_seed",
                                    help="Turn off to repeat a result: the same seed, start frame and settings "
                                         "give the same video.")
            seed = None if random_seed else int(st.number_input("Seed", 0, 2**31 - 4, 42, key="seed"))
            steps = st.slider("Sampling steps", 10, 50, 25, key="steps",
                              help="25 is the tested setting. Fewer is faster and rougher.")
            negative_extra = st.text_input("Also avoid (optional)", max_chars=300, key="neg_extra",
                                           placeholder="e.g. text, logo",
                                           help="Added to the built-in list of things to avoid.")

    # ---- right: what will be sent --------------------------------------------------------------------
    with right:
        w, h = sizes[size]["width"], sizes[size]["height"]
        if start_png:
            try:
                im = Image.open(io.BytesIO(start_png))
                iw, ih = im.size
                ratio = (iw / ih) / (w / h)
                if max(ratio, 1 / ratio) > info["limits"]["aspect_tolerance"]:
                    problems.append(f"The start frame is {iw}×{ih} but the output is {w}×{h}: pick a "
                                    f"{'portrait' if ih > iw else 'landscape'} size, or crop the image first.")
                st.image(start_png, caption=f"Start frame · {iw}×{ih}", width=220)
            except Exception:
                problems.append("The start frame is not a readable image.")
        else:
            problems.append("Choose a start frame.")
        prev = prompt_preview(url, json.dumps(guided, sort_keys=True) if guided else "", custom if custom else None) \
            if (guided or (custom and len(custom.strip()) >= 10)) else None
        if prev and prev.get("error"):
            problems.append(prev["error"])
        elif prev:
            if guided:
                st.session_state.last_guided_prompt = prev["prompt"]
            with st.container(border=True):
                st.caption("Prompt the model gets")
                st.markdown(prev["prompt"])
            for warn in prev["warnings"]:
                st.warning(warn, icon=":material/lightbulb:")
            if info.get("phone_embedding") and info["presets"]["phone_phrase"] in prev["prompt"]:
                st.caption(":material/smartphone: The learned phone look (E5) applies to this prompt.")

        frames, takes, _ = lengths[length]
        per_take = info["estimates"][size][str(frames)] * steps / 25
        load = 0 if health.get("model_loaded") == model else info["load_seconds"]
        ahead = health.get("queue", 0) + (1 if health.get("running") else 0)
        st.metric("Estimated time", minutes(takes * per_take + load),
                  help="From this server's measured speed. The first job after a start also loads the model.")
        notes = [f"{takes} take{'s' if takes > 1 else ''} of {frames} frames"]
        if load:
            notes.append(f"includes about {minutes(load)} to load the model")
        if ahead:
            notes.append(f"{ahead} job{'s' if ahead > 1 else ''} ahead of yours")
        st.caption(" · ".join(notes))

        for p in problems:
            st.error(p, icon=":material/block:")
        blocked = health.get("state") == "unavailable"
        go = st.button("Generate", type="primary", icon=":material/movie:", width="stretch",
                       disabled=bool(problems) or blocked)
        if go:
            params = {"model": model, "size": size, "frames": frames, "takes": takes, "mode": mode, "seed": seed,
                      "steps": steps, "negative_extra": negative_extra}
            if guided:
                params["guided"] = guided
            else:
                params["prompt"] = custom
            if preset and src == "Preset":
                params["start_preset"] = preset
            sig = json.dumps([params, start_name, len(start_png or b""), end_png[0] if end_png else None],
                             sort_keys=True)
            last_sig, last_t = st.session_state.last_submit
            if sig == last_sig and time.time() - last_t < 15:
                st.toast("That video was just queued.", icon=":material/info:")
            else:
                try:
                    up_start = None if params.get("start_preset") else (start_name, start_png, "image/png")
                    up_end = (end_png[0], end_png[1], "image/png") if end_png else None
                    job = ic.submit(url, params, up_start, up_end)
                    st.session_state.my_jobs.insert(0, job["id"])
                    st.session_state.last_submit = (sig, time.time())
                    server_status.clear()                 # the status badge shows the new job at once
                    st.toast(f"Queued: ready in about {minutes(job.get('eta_seconds'))}.", icon=":material/check:")
                    for warn in job.get("warnings") or []:
                        st.toast(warn, icon=":material/warning:")
                except ic.ApiError as e:
                    st.error(e.message, icon=":material/error:")

    st.subheader("Jobs on the server")
    jobs_panel(url)

# ---- saved videos (Azure): shown whether or not the server is on ----------------------------------
st.subheader("Saved videos")
saved = saved_videos()
if not saved:
    st.caption("No generated videos have been saved yet.")
else:
    st.caption("Every finished video is copied here, so it stays available after the GPU server is switched off.")
    for i in range(0, len(saved), 4):
        for col, j in zip(st.columns(4), saved[i:i + 4]):
            with col, st.container(border=True):
                st.markdown(f"**{job_title(j)}**")
                r = j.get("request") or {}
                st.caption(f"{r.get('width')}×{r.get('height')} · {(j.get('result') or {}).get('seconds')} s · "
                           f"seed {r.get('seed')} · {(j.get('finished_utc') or '')[:10]}")
                play(None, j)
                with st.popover("Prompt", icon=":material/notes:", width="stretch"):
                    st.code(r.get("prompt", ""), language=None, wrap_lines=True)

with st.expander("Use the API from your own code", icon=":material/code:"):
    base = server["url"] or "https://<server address>"
    st.markdown(
        f"The same server answers scripts directly. **Address now:** `{base}`"
        + (" (online)" if server["online"] else " (offline)")
        + ". It changes each time the GPU machine starts, and this page always shows the current one. "
        f"Interactive documentation, where you can try every call with your key: `{base}/docs`. "
        "Ask the team for an API key.")
    st.code(f"""# 1. queue a video (preset start frame + the guided prompt builder)
curl -s -X POST {base}/v1/jobs -H "X-API-Key: $PUDGY_API_KEY" \\
  -F 'params={{"guided": {{"characters": "pax", "action": "waving", "setting": "door"}},
              "start_preset": "waving_pax", "size": "portrait_720", "frames": 81, "takes": 1}}'
#    or your own start frame: add  -F start_image=@my_frame.png  and drop start_preset

# 2. poll until "state" is "done" (also: progress, stage, eta_seconds, error)
curl -s {base}/v1/jobs/<id> -H "X-API-Key: $PUDGY_API_KEY"

# 3. download
curl -s -o pudgy.mp4 {base}/v1/jobs/<id>/video -H "X-API-Key: $PUDGY_API_KEY\"""", language="bash")
