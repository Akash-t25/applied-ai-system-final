"""
VibeFinder — Streamlit UI
Agentic music recommender powered by Groq (llama-3.3-70b) + a custom scoring engine.
"""

import os
import sys
import html

# Ensure src/ is on the path regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from recommender import load_songs
from agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder",
    page_icon="🎵",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Sora:wght@500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
        }
        .main > div {
            padding-top: 1rem;
        }
        .vf-hero {
            position: relative;
            border-radius: 18px;
            padding: 1.1rem;
            color: white;
            margin-bottom: 1rem;
            overflow: hidden;
            background:
                radial-gradient(circle at 12% 20%, rgba(167, 139, 250, 0.35) 0, rgba(167, 139, 250, 0.0) 20%),
                radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.35) 0, rgba(56, 189, 248, 0.0) 18%),
                radial-gradient(circle at 70% 80%, rgba(244, 114, 182, 0.28) 0, rgba(244, 114, 182, 0.0) 24%),
                linear-gradient(145deg, #020617 0%, #111827 45%, #312e81 100%);
            animation: vfNebulaPulse 9s ease-in-out infinite;
            box-shadow: 0 14px 30px rgba(2, 6, 23, 0.35);
        }
        .vf-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image:
                radial-gradient(circle at 10% 15%, rgba(255,255,255,0.95) 0 1px, transparent 2px),
                radial-gradient(circle at 25% 70%, rgba(255,255,255,0.75) 0 1px, transparent 2px),
                radial-gradient(circle at 40% 35%, rgba(255,255,255,0.7) 0 1px, transparent 2px),
                radial-gradient(circle at 58% 22%, rgba(255,255,255,0.9) 0 1px, transparent 2px),
                radial-gradient(circle at 74% 48%, rgba(255,255,255,0.8) 0 1px, transparent 2px),
                radial-gradient(circle at 88% 78%, rgba(255,255,255,0.9) 0 1px, transparent 2px),
                radial-gradient(circle at 52% 84%, rgba(255,255,255,0.65) 0 1px, transparent 2px);
            opacity: 0.8;
            animation: vfTwinkle 5s ease-in-out infinite;
        }
        .vf-hero::after {
            content: "";
            position: absolute;
            inset: -30%;
            pointer-events: none;
            background: radial-gradient(circle, rgba(129, 140, 248, 0.15) 0%, rgba(129, 140, 248, 0) 60%);
            animation: vfDrift 14s linear infinite;
        }
        .vf-hero-box {
            position: relative;
            z-index: 1;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.35);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            padding: 1rem 1.1rem;
        }
        .vf-hero-box h1 {
            margin: 0 0 0.3rem 0;
            font-size: 1.9rem;
            font-weight: 700;
            font-family: "Sora", "Space Grotesk", sans-serif;
            letter-spacing: 0.2px;
        }
        .vf-hero-box p {
            margin: 0;
            opacity: 0.95;
            font-size: 1rem;
        }
        .vf-card {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            background: rgba(15, 23, 42, 0.02);
            margin-bottom: 0.6rem;
        }
        .vf-rank {
            font-weight: 700;
            color: #6366f1;
            margin-bottom: 0.25rem;
        }
        .vf-song-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .vf-artist {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 0.35rem;
        }
        .vf-explanation {
            font-size: 0.9rem;
            color: #334155;
            margin-top: 0.35rem;
        }
        .vf-log {
            border-left: 3px solid #6366f1;
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.5rem;
            background: rgba(99, 102, 241, 0.08);
        }
        .vf-log-title {
            font-size: 0.86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 0.15rem;
            color: #3730a3;
        }
        .vf-log-message {
            font-size: 0.93rem;
            color: #1f2937;
        }
        .vf-log-done {
            border-left-color: #10b981;
            background: rgba(16, 185, 129, 0.08);
        }
        .vf-log-done .vf-log-title {
            color: #047857;
        }
        .vf-log-retry {
            border-left-color: #f59e0b;
            background: rgba(245, 158, 11, 0.1);
        }
        .vf-log-retry .vf-log-title {
            color: #b45309;
        }
        .vf-log-error {
            border-left-color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }
        .vf-log-error .vf-log-title {
            color: #b91c1c;
        }
        @keyframes vfTwinkle {
            0%, 100% { opacity: 0.55; }
            50% { opacity: 1; }
        }
        @keyframes vfNebulaPulse {
            0%, 100% { filter: saturate(1) brightness(1); }
            50% { filter: saturate(1.2) brightness(1.08); }
        }
        @keyframes vfDrift {
            0% { transform: translateX(-8%) translateY(3%) rotate(0deg); }
            100% { transform: translateX(8%) translateY(-3%) rotate(360deg); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="vf-hero">
        <div class="vf-hero-box">
            <h1>🎵 VibeFinder</h1>
            <p>Describe your vibe in plain English. The agent classifies your intent, retrieves context, scores songs, and reflects before showing top picks.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load catalog once ─────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

@st.cache_data
def get_songs():
    return load_songs(DATA_PATH)

songs = get_songs()

# ── Input ─────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("What are you in the mood for?")
    query = st.text_input(
        "Describe your request",
        label_visibility="collapsed",
        placeholder="e.g. chill lofi music to study to from the 2020s",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search = st.button("Find Music", type="primary", use_container_width=True)
    with col2:
        st.caption("Try adding genre, mood, activity, and era for better matches.")

# ── Run agent ─────────────────────────────────────────────────────────────────
if search and query.strip():

    with st.spinner("Agent thinking..."):
        results, log = run_agent(query.strip(), songs)

    STEP_LABELS = {
        "CLASSIFY": "Intent Read",
        "RETRIEVE": "Context Pull",
        "PLAN": "Vibe Blueprint",
        "ACT": "Song Scoring",
        "REFLECT": "Quality Check",
        "RETRY": "Second Pass",
        "DONE": "Final Vibe Lock-In",
        "ERROR": "Route Blocked",
    }

    confidence_score = None
    timeline = []
    for step, message in log:
        if step == "CONFIDENCE":
            confidence_score = message
        else:
            timeline.append((step, message))

    # ── Results ───────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([2.2, 1], gap="large")

    with right_col:
        st.subheader("Session Insights")
        if confidence_score is not None:
            st.metric("Match Confidence", f"{confidence_score:.0%}")
            st.progress(min(max(confidence_score, 0.0), 1.0))
        else:
            st.info("Confidence score unavailable.")

        st.caption(f"Catalog size: {len(songs)} songs")
        with st.expander("Agent Timeline", expanded=True):
            for step, message in timeline:
                variant = ""
                if step == "DONE":
                    variant = " vf-log-done"
                elif step == "RETRY":
                    variant = " vf-log-retry"
                elif step == "ERROR":
                    variant = " vf-log-error"

                title = STEP_LABELS.get(step, step.title())
                safe_message = html.escape(str(message))
                st.markdown(
                    (
                        f'<div class="vf-log{variant}">'
                        f'<div class="vf-log-title">{title}</div>'
                        f'<div class="vf-log-message">{safe_message}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    with left_col:
        if results:
            st.subheader("Top Recommendations")

            for i, (song, score, explanation) in enumerate(results, start=1):
                tags = " · ".join(filter(None, [
                    song.get("genre", ""),
                    song.get("mood", ""),
                    f"energy {song.get('energy', '')}",
                    song.get("release_decade", ""),
                    song.get("detailed_mood", ""),
                ]))

                short_explanation = (
                    explanation[:140] + "..." if len(explanation) > 140 else explanation
                )

                rec_col, metric_col = st.columns([6, 1.4], gap="medium")
                with rec_col:
                    st.markdown(
                        (
                            '<div class="vf-card">'
                            f'<div class="vf-rank">#{i}</div>'
                            f'<div class="vf-song-title">{song["title"]}</div>'
                            f'<div class="vf-artist">{song["artist"]}</div>'
                            f"<div>{tags}</div>"
                            f'<div class="vf-explanation"><em>{short_explanation}</em></div>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                with metric_col:
                    st.metric("Score", f"{score:.2f}")

        else:
            st.warning("No recommendations found. Try rephrasing your request.")

elif search and not query.strip():
    st.warning("Please enter what you are in the mood for.")
