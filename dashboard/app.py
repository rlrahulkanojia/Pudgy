#!/usr/bin/env python3
"""Pudgy Penguins — client-facing training/inference dashboard.

Run with:
    source .venv-dashboard/bin/activate
    streamlit run dashboard/app.py
"""
import os
import sys

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import azure_utils
from manifest import (
    DATASETS,
    TRAINING_APPROACHES,
    approach_name,
    latest_approach,
    total_clip_count,
)

st.set_page_config(
    page_title="Pudgy Penguins — Training Dashboard",
    page_icon="🐧",
    layout="wide",
)


@st.cache_data(ttl=300, show_spinner=False)
def cached_list_videos(prefix):
    return azure_utils.list_videos(prefix)


@st.cache_data(ttl=600, show_spinner=False)
def cached_signed_url(blob_name):
    return azure_utils.signed_url(blob_name)


# 60s, not the 300s used for video listings: this is live progress and a five-minute-old
# step count reads as a stalled run.
@st.cache_data(ttl=60, show_spinner=False)
def cached_live_status():
    # The v8 run is finished and publishes no live status, so this renders nothing
    # (the v7 JSON would show a long-finished run as live).
    return azure_utils.read_json("v8/status.json")


def _pretty_name(display_name):
    return display_name.rsplit(".", 1)[0].replace("_", " ")


def render_video_grid(video_pairs, per_row=4):
    if not video_pairs:
        st.caption("No rendered output videos yet for this section.")
        return
    for i in range(0, len(video_pairs), per_row):
        chunk = video_pairs[i : i + per_row]
        columns = st.columns(per_row)
        for col, (blob_name, display_name) in zip(columns, chunk):
            with col, st.container(border=True):
                try:
                    url = cached_signed_url(blob_name)
                    st.video(url, width="stretch")
                except Exception as e:
                    st.error(f"Could not load {display_name}: {e}")
                st.caption(_pretty_name(display_name))


def render_outputs(approach):
    """Rendered inference output for one experiment, grouped when the run has sub-groups."""
    if approach["video_groups"]:
        for group in approach["video_groups"]:
            prefix = f"{approach['video_prefix']}/{group}"
            videos = cached_list_videos(prefix)
            st.markdown(f"#### {group.replace('_', ' ')} ({len(videos)})")
            render_video_grid(videos)
    else:
        videos = cached_list_videos(approach["video_prefix"])
        render_video_grid(videos)


def render_latest_experiment():
    """Headline the newest run on the main page so it lands without hunting through tabs."""
    latest = latest_approach()
    st.subheader("Latest experiment")
    with st.container(border=True):
        st.markdown(f"#### {latest['name']}")
        st.badge(latest["status"], color=latest["status_color"])
        st.markdown(f"**Base model:** {latest['base_model']}")
        st.markdown(f"**Approach:** {latest['thesis']}")
        st.markdown("**Key findings**")
        for bullet in latest["summary"]:
            st.markdown(f"- {bullet}")
    st.markdown("**Final output**")
    render_outputs(latest)



def render_live_training():
    """Live progress for the in-flight v7 runs, published from the GPU box.

    The box is not reachable from here, so everything comes from v7/status.json, which
    `finetune/wan/publish_status_v7.py` pushes on a timer. Absent or stale JSON renders
    as "no data" rather than an error — the dashboard outlives the box.
    """
    status = cached_live_status()
    if not status or not status.get("runs"):
        return
    st.subheader("Live training")
    st.caption(f"Published {status.get('generated_utc', '?')} UTC by the GPU box.")

    # What is happening RIGHT NOW. Training finishing is not the end of the work, and
    # without this the panel reads "all complete" while hours of evaluation are still
    # running — the single most misleading state this dashboard can show.
    stage = status.get("stage")
    if stage:
        with st.container(border=True):
            head, badge = st.columns([4, 1])
            head.markdown(f"#### Now running — {stage['name']}")
            badge.badge(stage.get("state", "running"), color="orange")
            st.markdown(stage["focus"])
            if stage.get("decides"):
                st.markdown(f"**What it decides:** {stage['decides']}")
            if stage.get("total"):
                st.progress(min(stage.get("done", 0) / stage["total"], 1.0))
                st.caption(f"{stage.get('done', 0)} of {stage['total']} test clips "
                           f"generated in the current batch.")
    else:
        st.info("No evaluation running — the box is idle between stages.", icon="⏸️")

    for run in status["runs"]:
        state = run.get("state", "unknown")
        colour = {"running": "green", "complete": "blue",
                  "stopped": "red", "not started": "grey"}.get(state, "grey")
        with st.container(border=True):
            head, badge = st.columns([4, 1])
            head.markdown(f"**{run['name']}**")
            badge.badge(state, color=colour)
            if state == "not started":
                st.caption("Queued — has not started on the box yet.")
                continue

            done, total = run.get("step", 0), run.get("total_steps", 0)
            a, b, c, d = st.columns(4)
            a.metric("Step", f"{done:,} / {total:,}")
            b.metric("Steps remaining", f"{run.get('steps_remaining', 0):,}")
            c.metric("Elapsed", run.get("elapsed", "—"))
            d.metric("Time remaining", run.get("eta", "—") if state == "running" else "—")
            if total:
                st.progress(min(done / total, 1.0))

            bits = [f"epoch {run['epoch']}" if run.get("epoch") else None,
                    f"{run['sec_per_step']:.1f} s/step" if run.get("sec_per_step") else None,
                    f"loss {run['loss']:.5f}" if run.get("loss") is not None else None,
                    f"{run.get('checkpoints', 0)} checkpoints"]
            st.caption("  ·  ".join(x for x in bits if x))
            if run.get("oom"):
                st.warning("This run hit a CUDA OOM at some point — check the log.", icon="⚠️")


def render_overview():
    render_live_training()
    render_latest_experiment()

    st.subheader("Datasets")

    # Explorable: filter by the run that used a dataset, pick one, drill into it.
    run_names = [a["name"] for a in TRAINING_APPROACHES
                 if any(a["id"] in d["used_by"] for d in DATASETS)]
    fcol, scol = st.columns([1, 2])
    with fcol:
        chosen_runs = st.multiselect(
            "Filter by run", run_names, default=run_names,
            help="Show only datasets used by these training runs.",
        )
    visible = [
        d for d in DATASETS
        if {approach_name(v) for v in d["used_by"]} & set(chosen_runs)
    ] or DATASETS
    with scol:
        selected_name = st.selectbox(
            "Inspect a dataset", [d["name"] for d in visible],
            help="Pick a dataset to see its full spec and provenance notes.",
        )

    st.dataframe(
        [
            {
                "Dataset": d["name"],
                "Clips (from client)": d["clip_count"],
                # str(): the column mixes ints and "~303 windows", which Arrow can't type.
                "Training clips": str(d["training_clips"]),
                "Resolution": d["resolution"],
                "FPS": d["fps"],
                "Frames": d["frames"],
                "Duration": d["duration"],
                "Used by": ", ".join(approach_name(v) for v in d["used_by"]),
            }
            for d in visible
        ],
        hide_index=True,
        width="stretch",
    )

    chosen = next((d for d in DATASETS if d["name"] == selected_name), None)
    if chosen:
        with st.container(border=True):
            st.markdown(f"#### {chosen['name']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clips from client", chosen["clip_count"],
                      help="Clips as delivered. 'Training clips' is what reached the trainer "
                           "after processing.")
            m2.metric("FPS", chosen["fps"])
            m3.metric("Frames", chosen["frames"])
            m4.metric("Duration", chosen["duration"])
            st.markdown(f"**Training clips:** {chosen['training_clips']}"
                        + ("" if chosen["training_clips"] == chosen["clip_count"]
                           else "  ·  derived from the client's clips by processing, see notes"))
            st.markdown(f"**Resolution:** {chosen['resolution']}")
            st.markdown(f"**Used by:** "
                        f"{', '.join(approach_name(v) for v in chosen['used_by'])}")
            st.markdown(f"**Notes:** {chosen['notes']}")

            share = chosen["clip_count"] / max(total_clip_count(), 1)
            st.caption(f"{share:.0%} of all client-delivered clips "
                       f"({chosen['clip_count']} of {total_clip_count()})")
            st.progress(min(share, 1.0))

    st.subheader("Training approach summary")
    st.dataframe(
        [
            {
                "Experiment": a["name"],
                "Base model": a["base_model"],
                "Status": a["status"],
                "Approach": a["thesis"],
            }
            for a in TRAINING_APPROACHES
        ],
        hide_index=True,
        width="stretch",
    )


def render_approach_tab(approach):
    st.badge(approach["status"], color=approach["status_color"])
    st.markdown(f"**Base model:** {approach['base_model']}")
    st.markdown(f"**Approach:** {approach['thesis']}")

    st.markdown("**Key findings**")
    for bullet in approach["summary"]:
        st.markdown(f"- {bullet}")

    st.divider()
    st.markdown("### Final output")
    render_outputs(approach)


def main():
    st.title("🐧 Pudgy Penguins — Training Dashboard")
    st.caption(
        "Training-run status and rendered inference output across every base-model track "
        "explored for the Pudgy Penguins 2D-animation video model."
    )

    if not azure_utils.container_exists():
        st.warning(
            f"Azure container `{azure_utils.CONTAINER}` not found or unreachable — "
            "video playback will be unavailable until assets are uploaded "
            "(see `dashboard/upload_assets.py`). Stats below are still shown.",
            icon="⚠️",
        )

    latest = latest_approach()
    # The last column is wider — the latest experiment's name needs the room.
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.metric("Datasets", len(DATASETS))
    c2.metric("Clips from client", total_clip_count(),
              help="Total clips delivered by the client across all datasets.")
    c3.metric("Training approaches", len(TRAINING_APPROACHES))
    c4.metric("Latest experiment", latest["name"], help=latest["status"])

    st.divider()

    tab_labels = ["Overview"] + [a["name"] for a in TRAINING_APPROACHES]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview()

    for tab, approach in zip(tabs[1:], TRAINING_APPROACHES):
        with tab:
            render_approach_tab(approach)


if __name__ == "__main__":
    main()
