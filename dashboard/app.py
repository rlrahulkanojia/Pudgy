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
from manifest import DATASETS, TRAINING_APPROACHES, total_clip_count

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


def render_overview():
    st.subheader("Datasets")
    st.dataframe(
        [
            {
                "Dataset": d["name"],
                "Clips": d["clip_count"],
                "Resolution": d["resolution"],
                "FPS": d["fps"],
                "Frames": d["frames"],
                "Duration": d["duration"],
                "Used by": ", ".join(d["used_by"]),
            }
            for d in DATASETS
        ],
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Dataset notes"):
        for d in DATASETS:
            st.markdown(f"**{d['name']}** — {d['notes']}")

    st.subheader("Training approach summary")
    st.dataframe(
        [
            {
                "Version": a["id"],
                "Base model": a["base_model"],
                "Status": a["status"],
                "Approach": a["thesis"],
            }
            for a in TRAINING_APPROACHES
        ],
        hide_index=True,
        use_container_width=True,
    )


def render_version_tab(approach):
    badge_color = approach["status_color"]
    st.markdown(
        f"<span style='background:{badge_color}22;color:{badge_color};"
        f"padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85em;'>"
        f"{approach['status']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Base model:** {approach['base_model']}")
    st.markdown(f"**Approach:** {approach['thesis']}")

    st.markdown("**Key findings**")
    for bullet in approach["summary"]:
        st.markdown(f"- {bullet}")

    st.divider()
    st.markdown("### Final output")

    if approach["video_groups"]:
        for group in approach["video_groups"]:
            prefix = f"{approach['video_prefix']}/{group}"
            videos = cached_list_videos(prefix)
            st.markdown(f"#### {group} ({len(videos)})")
            render_video_grid(videos)
    else:
        videos = cached_list_videos(approach["video_prefix"])
        render_video_grid(videos)


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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", len(DATASETS))
    c2.metric("Total clips curated", total_clip_count())
    c3.metric("Training approaches", len(TRAINING_APPROACHES))
    current_lead = next(a for a in TRAINING_APPROACHES if a["id"] == "v4")
    c4.metric("Current lead", current_lead["id"].upper(), current_lead["status"])

    st.divider()

    tab_labels = ["Overview"] + [a["title"] for a in TRAINING_APPROACHES]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview()

    for tab, approach in zip(tabs[1:], TRAINING_APPROACHES):
        with tab:
            render_version_tab(approach)


if __name__ == "__main__":
    main()
